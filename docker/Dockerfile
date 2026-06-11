# docker/Dockerfile  —  CPU image, deployment target (EC2).
# =============================================================================
# BUILD / RUN (note the -f and the trailing "." = build context = repo root):
#     docker build -f docker/Dockerfile -t sec-rag:cpu .
#     docker run --rm -p 8000:8000 \
#         -e DATABASE_URL=postgresql://postgres:pw@host:5432/sec_rag \
#         -v $HOME/.cache/models:/models \
#         sec-rag:cpu
#
# The build CONTEXT is the repo root (the "." at the end of build), NOT the
# docker/ folder. That's why COPY paths below are like `src/` and `app/`,
# relative to the repo root — not `../src/`.
#
# HARD TRADE-OFF ON CPU:
#   bitsandbytes 4-bit needs CUDA at runtime. With no GPU, rag_chain.py falls
#   back to fp32 (see its `torch.cuda.is_available()` guard). fp32 Mistral-7B
#   needs ~28 GB RAM and generates in tens of seconds per query. So this image
#   RUNS on EC2 but is slow and memory-hungry. If the deployed demo only needs
#   to prove the *pipeline* (retrieval + serving), that's acceptable; if it
#   needs fast generation, you want a GPU instance + Dockerfile.gpu instead.
# =============================================================================

# Slim Python base — no CUDA layers, keeps the CPU image small.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer (cached independently of source)
COPY pyproject.toml ./
COPY src/__init__.py ./src/__init__.py
COPY app/__init__.py ./app/__init__.py
RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

# Source layer
COPY src/ ./src/
COPY app/ ./app/

# Runtime hardening
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# HEALTHCHECK: container is "healthy" only once /health reports ready: true.
# start-period is long because the fp32 model load on CPU is SLOW; too short a
# grace period would flap the container as unhealthy during warmup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=urllib.request.urlopen('http://localhost:8000/health'); \
sys.exit(0 if json.load(r).get('ready') else 1)"

# One worker (one model copy). uvicorn imports `app.main` and serves `app`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
