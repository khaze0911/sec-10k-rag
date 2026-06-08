# SEC 10-K RAG Pipeline

> RAG pipeline over SEC 10-K filings using LangChain, Mistral-7B (4-bit quantized), and pgvector — deployed as FastAPI + Docker on AWS EC2.

**Status: In Progress**

---

## Overview


---

## Results



---

## Architecture



---

## Stack

| Component | Technology |
|---|---|
| Data source | SEC EDGAR API |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | pgvector (PostgreSQL) |
| Keyword search | BM25 (rank-bm25) |
| Retrieval | Hybrid BM25 + vector (Reciprocal Rank Fusion) |
| Orchestration | LangChain |
| LLM | Mistral-7B-Instruct (4-bit quantized) |
| API | FastAPI + Docker |
| Deployment | AWS EC2 |

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
│   ├── Dockerfile          # CPU (EC2)
│   ├── Dockerfile.gpu      # GPU (local dev)
│   └── docker-compose.yml  # App + Postgres together
├── pipeline.py             # Master pipeline script
├── requirements.txt
├── known_failures.md
└── .env.example
```

---

## Setup



---

## Usage


---

## Evaluation



---

## Related Projects

- [Banking77 Intent Classifier](https://github.com/khaze0911/banking77-distilbert) — Fine-tuned DistilBERT, 93% macro F1, deployed on AWS EC2