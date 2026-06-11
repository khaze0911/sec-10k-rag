# SEC 10-K RAG Pipeline

> Hybrid BM25 + vector retrieval over SEC 10-K filings, with a measured evaluation loop. Built with LangChain, Mistral-7B (4-bit quantized), pgvector, and FastAPI; containerized with Docker and deployed on AWS EC2 (GPU).

**Status: Complete — deployed on AWS EC2 (`g5.xlarge`); instance run on-demand for demos.**

---

## Overview

A retrieval-augmented generation pipeline that answers natural-language questions over the SEC 10-K filings of 7 financial companies (Block, Coinbase, Goldman Sachs, JPMorgan Chase, Lemonade, PayPal, Visa). Filings are fetched from EDGAR, parsed into sections (Business, Risk Factors, MD&A, Market Risk, Financial Statements), chunked, embedded, and stored in pgvector. At query time a **hybrid retriever** runs BM25 keyword search and dense vector search in parallel and fuses the two rankings with Reciprocal Rank Fusion (RRF); the top chunks are passed to a 4-bit Mistral-7B for grounded generation.

The project is a real pipeline with a measured, honest evaluation loop: a hand-labeled gold set, a retrieval metric chosen deliberately, before/after measurement of an index change and a documented failure taxonomy including failures the headline metric cannot see.

Corpus: **7 companies, ~2,932 chunks**, embedded with `all-MiniLM-L6-v2` (384-dim, cosine).

---

## Results

Retrieval quality measured as **hit@5** on a 15-question hand-labeled gold set (12 normal questions + 3 deliberately unanswerable "retrieval-impossible" questions, scored separately). hit@5 = a hit if any of the top 5 fused chunks matches the target (company, section) pair.

The headline experiment: an over-partitioned `ivfflat` index was diagnosed and corrected, measuring retrieval **one variable at a time**.

| Config (normal questions) | BM25 | Vector | Fused |
|---|---|---|---|
| `lists=100, probes=1` (baseline) | 0.75 | 0.67 | 0.83 |
| `lists=16, probes=1` (rebuild only) | 0.75 | 0.58 | 0.83 |
| `lists=16, probes=4` (final) | 0.75 | **0.75** | **0.92** |

**Finding: the number of `probes` was the binding constraint, not the index rebuild.** Rebuilding the index alone was *net-negative* on vector recall (it relocated the index's blind spots rather than removing them); recovery came from raising `probes` so the search scans more of the index. This was only diagnosable because each step changed exactly one variable. BM25 is unaffected by the index and served as a built-in control (identical across all three runs, as expected).

**What this does and does not claim** (see [known_failures.md](known_failures.md)):
- **True:** diagnosed an over-partitioned `ivfflat` index; measured retrieval recall across a 3-step before/after; recall of the labeled target block recovered (one key question went from *not retrieved* to *rank 1* on the vector channel).
- **Not yet true:** "answers risk-factor questions correctly." A parser boundary bug (F6 below) means some retrieved chunks carry the right `(company, section)` label but the wrong *content*, so a perfect hit@5 can still sit on top of contaminated text. This is fixed in the post-sprint parser batch, then re-evaluated. The metric is honest about what it measures; it cannot see a label error in the data it grades against — only reading the retrieved text caught that.

Latency is not yet measured; the `probes=4` cost is acknowledged qualitatively only.

---

## Architecture

**Offline (one-time corpus build):**

```mermaid
flowchart LR
    EDGAR["SEC EDGAR"] --> FETCH["Fetch 10-Ks"]
    FETCH --> PARSE["Parse sections"]
    PARSE --> CHUNK["Chunk"]
    CHUNK --> EMBED["Embed (all-MiniLM-L6-v2)"]
    EMBED --> PG[("pgvector")]
```

**Query-time (the live path served by FastAPI):**

```mermaid
flowchart TD
    Q["Query"] --> R

    subgraph R["Hybrid retriever: BM25 + vector, fused with RRF"]
        BM25["BM25 keyword search (rank-bm25, in-memory)"]
        VEC["Vector search (pgvector, ivfflat, cosine)"]
        BM25 --> RRF["Reciprocal rank fusion"]
        VEC --> RRF
    end

    RRF --> CHAIN["RagChain · Mistral-7B 4-bit (top-k chunks as context)"]
    CHAIN --> OUT["Answer + sources"]
```

Retrieval design notes:
- **hit@5, not recall@k.** Risk-factor sections run to 178–227 chunks each; exhaustively labeling every relevant chunk is infeasible, and bootstrapping a relevant set from the retriever's own output is circular. hit@5 uses a target `(company, section)` pair with denominator 1.
- **Gold questions are derived from the corpus, never from the retriever's output.** Direction is corpus → questions → grade retriever, so the ground truth is independent of the system under test.
- **`ivfflat` index is built post-load** from the row count (lists sized to the corpus), not at schema-creation time on an empty table (which trains degenerate clusters).

---

## Stack

| Component | Technology |
|---|---|
| Data source | SEC EDGAR API |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector store | pgvector (PostgreSQL), cosine, `ivfflat` |
| Keyword search | BM25 (`rank-bm25`) |
| Retrieval | Hybrid BM25 + vector, Reciprocal Rank Fusion |
| Orchestration | LangChain |
| LLM | Mistral-7B-Instruct (4-bit quantized via bitsandbytes) |
| API | FastAPI (Uvicorn) |
| Container | Docker / docker-compose |
| Deployment | AWS EC2 (`g5.xlarge`, A10G GPU) |

---

## Project Structure

```
sec-10k-rag/
├── src/
│   ├── __init__.py
│   ├── edgar_client.py     # Fetches 10-K filings from SEC EDGAR
│   ├── parser.py           # Extracts sections (Business, Risk Factors, MD&A)
│   ├── chunker.py          # Splits sections into overlapping chunks
│   ├── ingest.py           # Embeds chunks and loads into pgvector
│   ├── retriever.py        # Hybrid BM25 + vector retriever (RRF)
│   ├── rag_chain.py        # LangChain RAG chain + Mistral-7B
│   └── eval.py             # Evaluation framework
├── scripts/
│   ├── __init__.py
│   └── diagnostic_retrieval.py   # Diagnostic script for the retrieval system
├── eval/
│   └── gold_qa.json        # Hand-labeled Q&A pairs for evaluation
├── app/
│   ├── main.py             # FastAPI app
│   └── models.py           # Pydantic schemas
├── docker/
│   ├── Dockerfile          # CPU image — local dev / fp32 fallback, slow
│   ├── Dockerfile.gpu      # GPU image — the EC2 deployment target (4-bit inference)
│   └── docker-compose.yml  # API + Postgres(pgvector); portable base (committed)
├── pipeline.py             # Offline orchestrator: fetch → parse → chunk → ingest
├── pyproject.toml          # Packaging + dependencies (editable install)
├── known_failures.md       # Failure taxonomy (F1–F6), canonical
└── .env.example
```
> **Note on the Docker labels:** the GPU image is the deployment target because 4-bit quantization needs CUDA at runtime. The CPU image runs the same app but falls back to fp32 Mistral-7B (~28 GB RAM, slow) and exists only for local dev without a GPU.

---

## Setup

**Prerequisites:** Python 3.10+, PostgreSQL with the `pgvector` extension, and (for fast generation) an NVIDIA GPU with CUDA. A `.env` file with `DATABASE_URL`.

```bash
# 1. Clone and install (editable install wires up the `src` and `app` packages)
git clone https://github.com/khaze0911/sec-10k-rag.git
cd sec-10k-rag
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Configure environment
cp .env.example .env
# edit .env:  DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/sec_rag

# 3. Database: create the db and enable pgvector
createdb sec_rag
psql sec_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## Usage

### Build the corpus (offline, one-time)

```bash
python pipeline.py                 # full pipeline: fetch → parse → chunk → ingest
python pipeline.py --step fetch    # individual steps also available
python pipeline.py --step parse
python pipeline.py --step chunk
```

> **Do not re-run the ingest step against the existing measured index.** Re-ingesting rebuilds the `ivfflat` index and re-clusters it, which invalidates the recorded before/after eval results. The corpus is intentionally frozen until the post-sprint parser batch (which has its own re-ingest + re-eval). Treat `python pipeline.py --step ingest` as a deliberate action, not a routine one.

### Query from the command line

```bash
python -m src.rag_chain "What does Visa identify as key risks?"
python -m src.retriever            # runs the retriever's built-in test queries
```

### Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# then:
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How does PayPal generate revenue?"}'
curl localhost:8000/health        # {"status":"ok","ready":true} once the model is loaded
```

### Run with Docker (GPU)

```bash
# from repo root; build context is the root, Dockerfile lives in docker/
docker build -f docker/Dockerfile.gpu -t sec-rag:gpu .

# run against a Postgres on the host machine (bridge networking):
docker run --rm --gpus all -p 8000:8000 \
    --add-host=host.docker.internal:host-gateway \
    -e DATABASE_URL='postgresql://USER:PASSWORD@host.docker.internal:5432/sec_rag' \
    -v /path/to/models:/models \
    sec-rag:gpu

# simpler alternative — share the host network stack (drop -p; use localhost):
docker run --rm --gpus all --network host \
    -e DATABASE_URL='postgresql://USER:PASSWORD@localhost:5432/sec_rag' \
    -v /path/to/models:/models \
    sec-rag:gpu

# or bring up API + Postgres together (no host-DB networking needed):
docker compose -f docker/docker-compose.yml up --build
```

Notes:
- Requires the NVIDIA Container Toolkit on the host (`--gpus all` fails without it).
- `host.docker.internal` needs the `--add-host` flag on Linux. For it to connect, the host Postgres must listen beyond `localhost` (`listen_addresses='*'`), allow the docker subnet in `pg_hba.conf` (`172.17.0.0/16`), and have the firewall permit it. The `--network host` form avoids all of that for local runs.
- `-v .../models:/models` mounts existing weights to skip a ~14 GB re-download; an empty mount downloads them from HuggingFace on first run and caches them after.

---

## Deployment (AWS EC2)

Deployed on a **`g5.xlarge`** GPU instance (single NVIDIA A10G, 24 GB VRAM), running 4-bit Mistral-7B. The instance is **run on-demand for demos, not left running 24/7** — started before a demo, stopped after — so GPU compute is billed only while in use (~$1/hr), with only the EBS volume billed while stopped. The deployment artifact (image + EC2 configuration) is the deliverable; uptime is not the goal for a portfolio demo.

**Instance:** Deep Learning OSS Nvidia Driver AMI (Ubuntu), which ships with the NVIDIA driver and Container Toolkit pre-installed, so GPU passthrough works out of the box. The security group restricts SSH (22) and the API (8000) to a single IP. The instance's EBS root volume persists the model weights and Postgres data across stop/start.

**Bring-up** is staged, because the database starts empty and the model weights download on first run:

```bash
# On the instance: clone the repo, copy up a pg_dump of the corpus, and create
# docker/.env with HF_TOKEN (authenticates the ~14 GB Mistral download) and
# POSTGRES_PASSWORD.

# 1. Start Postgres, restore the corpus (NOT ingest — that rebuilds the measured
#    ivfflat index), and verify the row count before starting the API:
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d db
docker compose -f docker/docker-compose.yml exec -T db \
    psql -U postgres -d sec_rag < sec_rag.dump
docker compose -f docker/docker-compose.yml exec db \
    psql -U postgres -d sec_rag -c "SELECT count(*) FROM chunks;"   # expect 2932

# 2. Start the API (first run builds the image + downloads weights):
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d --build api

# 3. Health-check, then query:
curl localhost:8000/health
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"How does PayPal generate revenue?"}'
```

The `HF_TOKEN` is required: an unauthenticated download of the Mistral weights gets rate-limited and stalls. The staged restore (rather than re-running ingest) preserves the exact `ivfflat` index the evaluation was measured against.

---

## Evaluation

```bash
# Each run records hit@k per channel (bm25 / vector / fused) plus first-hit rank.
# The filename encodes the index config (the harness can't read it from the DB).
python -m src.eval --probes 1 --out eval/results_lists100_probes1.json
python -m src.eval --probes 4 --out eval/results_lists16_probes4.json
```

The evaluation harness is retrieval-only (it never calls the LLM), uses the production fusion code, and scores the `retrieval-impossible` questions separately from the headline so that deliberately-unanswerable cases don't inflate or deflate the main number. See [known_failures.md](known_failures.md) for the full failure taxonomy (F1–F6), including the parser boundary bug (F6) that a metric defined over the data labels structurally cannot detect.

---

## Limitations & extensions

- **Scope:** 7 companies, ~2,932 chunks. Scaling to 30–50 filings is the natural next step, using the existing eval harness as the re-validation instrument.
- **Parser boundaries (F2, F6):** some sections are missing (start-boundary failures) and some `risk_factors` sections over-run into the next items (end-boundary overshoot), so a fraction of chunks are mislabeled. Fixed in the post-sprint parser batch, then re-evaluated as its own before/after.
- **Fusion (F5):** RRF can demote a strong single-channel hit relative to weaker chunks that appear in both channels. Candidate mitigations (larger candidate pool, weighted RRF, score normalization) are scoped, each to be measured with its own before/after.
- **Company routing:** embeddings encode topic far more strongly than company identity; query-side company detection / metadata filtering is future work.
- **Latency:** not yet measured.

---

## Related Projects

- [Banking77 Intent Classifier](https://github.com/khaze0911/banking77-intent-classifier) — Fine-tuned DistilBERT, 93% macro F1, deployed on AWS EC2