FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/data/cache/huggingface \
    TERRIER_VERSION=5.11 \
    TERRIER_HELPER_VERSION=0.0.8 \
    USE_TF=0 \
    USE_FLAX=0 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --create-home --uid 1000 demo \
    && mkdir -p /app/data/cache \
    && chown -R demo:demo /app/data
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=demo:demo . .
USER demo
ARG IR_DEMO_CORPUS=scifact
ENV IR_DEMO_CORPUS=${IR_DEMO_CORPUS}

# A build is successful only after the data, indexes, and pinned models
# are available and the real search pipeline has returned results.
RUN python scripts/prepare_demo.py
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3).close()"]
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
