"""
retriever.py — Hybrid BM25 + Vector Search with Reciprocal Rank Fusion
=======================================================================

Run:
  Not run directly — imported by rag_chain.py and the FastAPI app.
  But has a __main__ block for testing:
    python src/retriever.py
"""

import os
from pathlib import Path
import numpy as np
import psycopg2
import psycopg2.extras
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
EMBED_MODEL  = "all-MiniLM-L6-v2"   # must match what ingest.py used

CANDIDATE_K = 20 # before fusion
TOP_K       = 5 # after fuison
RRF_K = 60 # standard value

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

# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------
"""
Load all chunk texts from PostgreSQL and build an in-memory BM25 index.

Returns:
    bm25:      The BM25Okapi index object (used to score queries)
    chunk_ids: List of chunk_ids in the same order as the BM25 corpus
"""
def build_bm25_index(conn) -> tuple[BM25Okapi, list[str]]:
  # RealDictCursor returns rows as dicts: {"chunk_id": ..., "text": ...} instead of tuple
  with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    # chunk_id for lookup, text for indexing
    cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id;")
    rows = cur.fetchall()
  
  chunk_ids = [row["chunk_id"] for row in rows]

  # tokenize: lowercase and split on whitespace
  tokenized = [row["text"].lower().split() for row in rows]

  # compute IDF for every term
  bm25 = BM25Okapi(tokenized)

  print(f"BM25 index built: {len(chunk_ids):,} documents")
  return bm25, chunk_ids

# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------
"""
Score all chunks against the query using BM25, return top-k with metadata
"""
def bm25_search(query: str, bm25: BM25Okapi, chunk_ids: list[str], conn, k: int = CANDIDATE_K) -> list[dict]:
  # tokenize query same as documents
  query_tokens = query.lower().split()
  scores = bm25.get_scores(query_tokens)
  
  # top-k scores in desc order
  top_indecies = np.argsort(scores)[::-1][:k]

  # map indecies to chunk_ids
  top_chunk_ids = [chunk_ids[i] for i in top_indecies]
  top_scores = [float(scores[i]) for i in top_indecies]

  # fetch metadata
  with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
      cur.execute(
          """
          SELECT chunk_id, company, section, chunk_index, text
          FROM chunks
          WHERE chunk_id = ANY(%s);
          """,
          (top_chunk_ids,),
      )
      rows = {row["chunk_id"]: dict(row) for row in cur.fetchall()}
  
  # ranked order results
  results = []
  for chunk_id, score in zip(top_chunk_ids, top_scores):
    if chunk_id in rows:
      result = rows[chunk_id].copy()
      result["score"] = score
      results.append(result)
  
  return results

# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
"""
Embed the query and find the k most similar chunks using pgvector

Returns a list of dicts, each with:
    chunk_id, company, section, text, score (cosine distance, lower = more similar)
"""
def vector_search(query: str, model: SentenceTransformer, conn, k: int = CANDIDATE_K) -> list[dict]:
  # embed query
  query_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist() ## psycopg2 needs a Python list

  sql = """
      SELECT
          chunk_id,
          company,
          section,
          chunk_index,
          text,
          -- <=> pgvector's cosine distance operator
          -- Lower score = more similar (0.0 = identical vectors)
          embedding <=> %s::vector AS score
      FROM chunks
      ORDER BY embedding <=> %s::vector
      LIMIT %s;
  """

  with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    # query_vec twice since it appears twice in the SQL
    cur.execute(sql, (query_vec, query_vec, k))
    results = cur.fetchall()
  
  return [dict(row) for row in results]

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
"""
Merge BM25 and vector search results using Reciprocal Rank Fusion.

Args:
    bm25_results:   Ranked list from bm25_search() (rank 0 = best)
    vector_results: Ranked list from vector_search() (rank 0 = best)
    k:              Number of final results to return

Returns:
    Top-k chunks ranked by combined RRF score, highest first.
    Each dict includes the original metadata plus an "rrf_score" field.
"""
def reciprocal_rank_fusion(bm25_results: list[dict], vector_results: list[dict], k: int = TOP_K) -> list[dict]:
  rrf_scores: dict[str, float] = {}
  chunk_data: dict[str, dict] = {}

  # process BM25 results
  for rank, result in enumerate(bm25_results):
    cid = result["chunk_id"]
    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + 1 + RRF_K)
    chunk_data[cid] = result
  
  # process vector results
  for rank, result in enumerate(vector_results):
    cid = result["chunk_id"]
    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + 1 + RRF_K)
    # prefer the vector result's metadata if chunk appeared in both
    if cid not in chunk_data:
      chunk_data[cid] = result
    
  # sort chunks by RRF score desc
  top_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k] #top k

  results = []
  for cid in top_ids:
    result = chunk_data[cid].copy()
    result["rrf_score"] = round(rrf_scores[cid], 6)
    results.append(result)

  return results  

  

# ---------------------------------------------------------------------------
# Hybrid retriever (main interface)
# ---------------------------------------------------------------------------
"""
Main retriever class. Wraps BM25 + vector search + RRF into one object
"""
class HybridRetriever:
  """
  Initialize the retriever: load model, connect to DB, build BM25 index
  Called once at startup
  """
  def __init__(self):
    print(f"LOADING embedding model: {EMBED_MODEL}")
    self.model = SentenceTransformer(EMBED_MODEL)

    print(f"CONNECTING to PostgreSQL ...")
    self.conn = get_connection()

    print(f"BUILDING BM25 index ..")
    self.bm25, self.chunk_ids = build_bm25_index(self.conn)

    print("INITIALIZED HybridRetriever")

  """
  Run hybrid search for a query string.

  Args:
      query:       Natural language question or keyword query
      k:           Number of final chunks to return (default: TOP_K=5)
      candidate_k: How many candidates each search method fetches (default: 20)
                    Higher = more candidates for RRF to consider = slightly
                    better recall at cost of more DB/compute work

  Returns:
      List of k chunk dicts, sorted by RRF score (best first)
      Each dict has: chunk_id, company, section, chunk_index, text, rrf_score
  """
  def search(self, query: str, k: int = TOP_K, candidate_k: int = CANDIDATE_K) -> list[dict]:
    # BM25 keyword search
    bm25_results = bm25_search(query, self.bm25, self.chunk_ids, self.conn, k=candidate_k)

    # Vector semantic search
    vector_results = vector_search(query, self.model, self.conn, k=candidate_k)

    # fuse rankings
    results = reciprocal_rank_fusion(bm25_results, vector_results, k=k)

    return results

  def close(self):
      """Close the DB connection"""
      self.conn.close()

# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":

  retriever = HybridRetriever()

  test_queries = [
      "What are Coinbase's main risk factors?",
      "How does PayPal generate revenue?",
      "What is Block's strategy for Bitcoin?",
      "Goldman Sachs interest rate risk",
  ]

  for query in test_queries:
      print(f"\n{'─'*60}")
      print(f"QUERY: {query}")
      print(f"{'─'*60}")
      results = retriever.search(query, k=3)
      for i, r in enumerate(results, 1):
          print(f"\n  [{i}] {r['company']} — {r['section']} (chunk {r['chunk_index']})")
          print(f"       RRF score: {r['rrf_score']}")
          # Print first 200 chars of the chunk text as a preview
          preview = r["text"][:200].replace("\n", " ")
          print(f"       Preview: {preview}…")

  retriever.close()