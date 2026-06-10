"""
app/main.py — FastAPI application wiring the RAG pipeline behind HTTP

RUN (dev):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import anyio  # exposes the thread-pool limiter
from fastapi import FastAPI, HTTPException

# Our schemas (cheap import — pydantic only).
from app.models import AskRequest, AskResponse, HealthResponse, Source

# Pipeline: Importing this pulls in torch/transformers,
from src.rag_chain import RagChain, RagResult
from src.retriever import POOL_MAX_CONN, close_pool

# Basic logging so startup/shutdown and errors are visible in container logs.
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag.api")

# ---------------------------------------------------------------------------
# MAPPING: domain object -> response schema
# ---------------------------------------------------------------------------
"""Convert the chain's RagResult (a dataclass holding the answer + a list
of chunk dicts) into the Pydantic AskResponse
"""
def _to_response(result: RagResult) -> AskResponse:
    sources = [
        Source(
            chunk_id=c.get("chunk_id"),
            company=c.get("company"),
            section=c.get("section"),
            chunk_index=c.get("chunk_index"),
            rrf_score=c.get("rrf_score"),
        )
        for c in result.sources
    ]
    return AskResponse(answer=result.answer, sources=sources)

# ---------------------------------------------------------------------------
# APP STATE
# ---------------------------------------------------------------------------
class _State:
    chain: RagChain | None = None

state = _State()

# ---------------------------------------------------------------------------
# LIFESPAN: startup / shutdown hooks
# ---------------------------------------------------------------------------
# The `lifespan` async context manager runs setup code BEFORE `yield` (on
# startup) and teardown code AFTER `yield` (on shutdown). FastAPI calls it once
# per process. This replaces the older @app.on_event("startup"/"shutdown").
@asynccontextmanager
async def lifespan(app: FastAPI):

    # ---- STARTUP ----------------------------------------------------------
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = POOL_MAX_CONN
    log.info("Thread pool capped at %d (matches POOL_MAX_CONN).", POOL_MAX_CONN)

    log.info("Startup: building RagChain (loads Mistral-7B + BM25 index)...")
    state.chain = await anyio.to_thread.run_sync(RagChain)
    log.info("Startup complete: RagChain ready.")

    yield  # run application

    # ---- SHUTDOWN ---------------------------------------------------------
    log.info("Shutdown: closing DB connection pool...")
    close_pool() # release pooled Postgres connections cleanly
    state.chain = None
    log.info("Shutdown complete.")

# create application
app = FastAPI(
    title="SEC 10-K RAG API",
    version="0.1.0",
    description="Hybrid BM25 + vector RAG over SEC 10-K filings.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + readiness probe"""
    return HealthResponse(status="ok", ready=state.chain is not None)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if state.chain is None:
        raise HTTPException(status_code=503, detail="Model not ready.")
    try:
        result = state.chain.answer(req.question)
    except Exception as exc:
        # retriever DB errors or generation errors, returns a generic 500 to the clientzas
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Internal error.") from exc

    return _to_response(result)