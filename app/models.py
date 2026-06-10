"""
app/models.py — Pydantic schemas (the JSON contract of the API)
"""
from __future__ import annotations 
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# REQUEST
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question for 10-K corpus"
    )

# ---------------------------------------------------------------------------
# RESPONSE
# ---------------------------------------------------------------------------
class Source(BaseModel):
    chunk_id: str | None = None
    company: str | None = None
    section: str | None = None
    chunk_index: int | None = None
    rrf_score: float | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

class HealthResponse(BaseModel):
    """ Returned by /health. `ready` is False until the model finishes loading,
    so an orchestrator (Docker healthcheck, load balancer) can wait for warmup
    before sending real traffic."""
    status: str
    ready: bool
