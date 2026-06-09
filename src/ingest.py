"""
ingest.py — Embed chunks and load into pgvector
================================================
    1. Load chunks from data/chunks/all_chunks.jsonl
    2. Run each chunk's text through a small neural net (MiniLM) → 384-dim vector
    3. Create a PostgreSQL table with a vector column (pgvector extension)
    4. Insert every chunk + its vector into that table
    5. Verify the counts look right

Run directly:
  python src/ingest.py

Or via master script:
  python pipeline.py --step ingest
"""

import json
import os
from pathlib import Path

import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer # embedding model wrapper
from tqdm import tqdm # progress bars
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNKS_PATH = Path("data/chunks/all_chunks.jsonl")
DATABASE_URL = os.getenv("DATABASE_URL") # postgresql://postgres:pw@localhost:5432/sec_rag

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
VECTOR_DIM = 384 # all-MiniLM-L6-v2 always outputs 384 numbers per input text

# ivfflat index sizing — computed at index-build time from the real row count,
# never hardcoded (see known_failures.md F1: a lists value sized for a much
# larger corpus silently broke vector recall on this one).
# pgvector heuristic: lists ~ rows/1000 for corpora under ~1M rows.
# MIN floors tiny corpora so the index isn't under-partitioned either;
# 16 was the value validated by the Day-4 eval on ~2.9k chunks.
IVFFLAT_MIN_LISTS = 16
IVFFLAT_ROWS_PER_LIST = 1000

# ---------------------------------------------------------------------------
# SQL table definition (DDL)
# ---------------------------------------------------------------------------
# executed once to set up the database schema
DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    -- TEXT PRIMARY KEY: chunk_id is a UUID string, unique per chunk.
    -- This is what makes our UPSERT safe (see UPSERT_SQL below).
    chunk_id       TEXT PRIMARY KEY,
    company        TEXT NOT NULL,
    filing_date    TEXT,          -- nullable: some filings may lack a date
    section        TEXT,          -- e.g. "business", "risk_factors", "mda"
    chunk_index    INTEGER,       -- position of this chunk within its section
    total_chunks   INTEGER,       -- how many chunks the section was split into
    text           TEXT NOT NULL, -- The actual text of the chunk

    -- THE KEY COLUMN
    -- `vector(384)` is a pgvector type
    -- It stores 384 float32 values efficiently and supports distance operators
    embedding      vector(384)
);
"""

# ---------------------------------------------------------------------------
# SQL upsert (DML)
# ---------------------------------------------------------------------------
UPSERT_SQL = """
INSERT INTO chunks
    (chunk_id, company, filing_date, section, chunk_index, total_chunks, text, embedding)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    -- EXCLUDED refers to the row we tried to INSERT (the new values).
    -- So: if chunk_id already exists, overwrite its embedding and text
    -- with the fresh values. Metadata (company, section, etc.) stays the same.
    embedding = EXCLUDED.embedding,
    text      = EXCLUDED.text;
"""

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
"""
Open and return a psycopg2 database connection

DATABASE_URL format: postgresql://user:password@host:port/dbname
"""
def get_connection():
  if not DATABASE_URL:
    raise ValueError(
      "DATABASE_URL not set in env\n"
      "Expected format: postgresql://postgres:yourpassword@localhost:5432/sec_rag"
    )
  return psycopg2.connect(DATABASE_URL)

"""
Execute the DDL string to create the table and index.
"""
def create_schema(conn):
  with conn.cursor() as cur:
    cur.execute(DDL)
  conn.commit()
  print("SCHEMA ready")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
"""
Read all_chunks.jsonl and return a list of chunk dicts.
Chunk:
  {
    "chunk_id": "c0e5c94a-...",
    "company": "Block (Square)",
    "filing_date": "2026-02-26",
    "section": "business",
    "chunk_index": 0,
    "total_chunks": 106,
    "text": "ITEM 1. BUSINESS Our Purpose ..."
  }

"""
def load_chunks(path: Path) -> list[dict]:
  if not path.exists():
    raise FileNotFoundError(
        f"FILE not found: {path}"
    )
  
  chunks = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        chunks.append(json.loads(line))
  print(f"LOADED {len(chunks):,} chunks from {path}")
  return chunks

# ---------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------- 
"""
Convert chunk texts into dense vectors using the MiniLM model

Returns a numpy array of shape (N, 384) where N = number of chunks
Each row is the embedding for the corresponding chunk
"""
def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
  texts = [c["text"] for c in chunks]
  all_embeddings = []

  print(f"EMBEDDING {len(texts):,} chunks (batch_size = {BATCH_SIZE})")
  for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
    batch = texts[i: i + BATCH_SIZE]
    vecs = model.encode(
      batch,
      convert_to_numpy=True, # return np.ndarray, not a torch.Tensor
      normalize_embeddings=True, # L2-normalize
      show_progress_bar=False,
    )

    # (batch_size, 384), 384-dim vector per text
    all_embeddings.append(vecs)
  
  #  stack list of (64,384) arrays to single (2932, 384) array
  return np.vstack(all_embeddings).astype("float32")

# ---------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------- 
"""
Upsert all chunks + their embedding vectors into PostgreSQL.

"""
def ingest(chunks: list[dict], embeddings: np.array, conn) -> None:
  rows = []
  for chunk, vec in zip(chunks, embeddings):
    rows.append((
      chunk["chunk_id"],
      chunk["company"],
      chunk.get("filing_date"),
      chunk.get("section"),
      chunk.get("chunk_index"),
      chunk.get("total_chunks"),
      chunk["text"],
      vec.tolist()
    ))

  print(f"UPSERTING {len(rows):,} rows into PostgreSQL …")

  with conn.cursor() as cur:
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Upserting"):
      batch = rows[i: i + BATCH_SIZE]
      cur.executemany(UPSERT_SQL, batch)
      conn.commit()

  print(f"INGESTION complete — {len(rows):,} chunks in DB")

# ---------------------------------------------------------------------------
# Vector index (built AFTER ingestion, sized from the real corpus)
# ---------------------------------------------------------------------------
"""
Drop and rebuild the ivfflat index with `lists` computed from the actual row
count. Runs after every ingest so the k-means cell centers are retrained on
the full, current corpus (an index created before/partway through ingestion
clusters on incomplete data).

Query-time counterpart: ivfflat.probes (session GUC) is set in
retriever.py at search time
"""
def create_vector_index(conn) -> int:
  with conn.cursor() as cur:
    cur.execute("SELECT count(*) FROM chunks;")
    n_rows = cur.fetchone()[0]

  lists = max(IVFFLAT_MIN_LISTS, n_rows // IVFFLAT_ROWS_PER_LIST)

  with conn.cursor() as cur:
    cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx;")
    cur.execute(
      f"CREATE INDEX chunks_embedding_idx ON chunks "
      f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists});"
    )
  conn.commit()
  print(f"IVFFLAT index built: lists={lists} (computed from {n_rows:,} rows)")
  return lists

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
"""
Run two queries to confirm the data landed correctly:
  1. Total row count (should match chunk count from JSONL)
  2. Breakdown by company + section (spot-check for missing data)
"""
def verify(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks;")
        count = cur.fetchone()[0]
        cur.execute("""
            SELECT company, section, COUNT(*)
            FROM chunks
            GROUP BY company, section
            ORDER BY company, section;
        """)
        rows = cur.fetchall()

    print(f"\nTOTAL rows in DB: {count:,}")
    print(f"{'Company':<25} {'Section':<25} {'Chunks':>6}")
    print("─" * 60)
    for company, section, n in rows:
        print(f"{company:<25} {section:<25} {n:>6}")


# ---------------------------------------------------------------------------
# MODULE ENTRY POINT
# --------------------------------------------------------------------------- 
"""
Orchestrates the full ingest pipeline in order
Called by pipeline.py --step ingest, or directly via __main__
Returns a dict so pipeline.py can log the result
"""
def run_ingest():
  # load raw chunk data
  chunks = load_chunks(CHUNKS_PATH)

  # load embedding model
  print(f"LOADING embedding model: {EMBED_MODEL}")
  model = SentenceTransformer(EMBED_MODEL)

  # embed chunks (2932, 384) numpy array
  embeddings = embed_chunks(chunks, model)

  # connect to Postgres
  print("\nCONNECTING to PostgreSQL …")
  conn = get_connection()
  create_schema(conn)

  # write to db
  ingest(chunks, embeddings, conn)

  # build the vector index on the full corpus, lists sized from real row count
  create_vector_index(conn)

  # verify
  verify(conn)

  conn.close()

  # return a status dict so pipeline.py can log/assert on it
  return {"status": "ok", "chunks_ingested": len(chunks)}

# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ingest()