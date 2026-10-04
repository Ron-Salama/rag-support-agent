# Dockerfile = the recipe for the image: a sealed box holding Linux + Python + our packages + code +
# data + the embedding model + the built search index. Every instruction (FROM, RUN, COPY...) is one step; Docker caches
# each step and, on the next build, re-runs only the steps from the first one whose inputs changed.
# So the slow, rarely-changing steps (Python packages, the model) come first and our code comes last.
#
#   docker build -t rag-support-agent .                                  # build the image (first time ~10+ min)
#   docker run --env-file .env -p 8000:8000 rag-support-agent            # run it, then open http://localhost:8000/docs
#
# Step-by-step guide for Windows: docs/DOCKER_SETUP.md. Choices made here: DECISIONS.md C17.

# 1. Start from the official image that already contains Debian Linux + Python 3.13 ("slim" = no
#    compilers or extras; every package we need ships ready-made Linux "wheels" for Python 3.13).
FROM python:3.13-slim

# 2. Settings for every later step and for the running container:
#    PYTHONUNBUFFERED      print() shows up in `docker logs` immediately, not in delayed batches
#    PYTHONDONTWRITEBYTECODE  don't write .pyc cache files next to our code while it runs (cleaner container)
#    PIP_NO_CACHE_DIR      pip keeps no copy of what it downloaded -> a smaller image
#    PIP_DISABLE_PIP_VERSION_CHECK  pip doesn't go online to ask "is there a newer pip?" on every run
#    PIP_ROOT_USER_ACTION  installing as root is normal while BUILDING an image: skip pip's warning
#    HF_HOME               where Hugging Face models are stored inside the image (the same folder
#                          ragagent/config.py uses: <project>/.cache/huggingface)
#    HF_HUB_DISABLE_TELEMETRY  Hugging Face's library sends no usage statistics
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# 3. All our files live in /app inside the image (like `cd /app`, creating it).
WORKDIR /app

# 4. PyTorch FIRST, the CPU-only build from PyTorch's own package index. The normal Linux torch
#    bundles NVIDIA CUDA libraries (several GB) that we don't use: the embedding model runs on the CPU.
#    Same version as the tested .venv; keep it inside the range in requirements.txt.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 5. The other packages. requirements.txt is copied ALONE first, so this slow step is cached and only
#    re-runs when requirements.txt changes - not on every code edit. torch is already installed and
#    matches "torch>=2.14,<3", so pip leaves it alone (no CUDA download).
COPY requirements.txt .
RUN pip install -r requirements.txt

# 6. A normal user without admin rights to run the app (if someone ever broke into the web server,
#    they would not be root). /app becomes this user's, so the app can write its runtime files there.
RUN useradd --create-home --uid 10001 app && chown app:app /app
USER app

# 7. Download the embedding model INTO the image, so the container never needs the internet for it.
#    EMBED_MODEL must equal MODEL_NAME in ragagent/rag/embed.py. If they ever differ, step 10 fails
#    loudly, because of HF_HUB_OFFLINE below (better a failed build than a surprise download later).
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}', device='cpu')"
ENV HF_HUB_OFFLINE=1

# 8. Our code and the data the app reads. NOT copied: data/raw (the original files; the text below is
#    made from them), data/labels, outputs, .env (your API key!) - see .dockerignore.
#    --chown: the files belong to the "app" user (by default COPY makes them root's).
COPY --chown=app:app ragagent/ ragagent/
COPY --chown=app:app data/manifest.csv data/manifest.csv
COPY --chown=app:app data/text/ data/text/
COPY --chown=app:app data/fulltext/ data/fulltext/

# 9. Folders the app writes into while it runs: the LLM answer cache and the human review queue.
#    Made here by "app" so the app may write there (also when a Docker volume is mounted on them).
RUN mkdir -p .cache/llm outputs/review_queue

# 10. Build the search index (chunk -> embed -> Chroma) at BUILD time, so the container answers
#     right away. ~1.5 minutes on a laptop CPU. No LLM and no API key needed for this.
RUN python -m ragagent.rag.ingest

# 11. The eval scripts (last, because they change often and nothing above depends on them).
COPY --chown=app:app evals/ evals/

# 12. The API listens on port 8000 inside the container. `docker run -p 8000:8000` connects it to
#     port 8000 on your PC. EXPOSE itself only documents that; it opens nothing.
EXPOSE 8000

# 13. Docker asks the app every 30 s "are you alive?" (shown as "healthy" in `docker ps`).
#     python:3.13-slim has no curl, so we ask with Python's own urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

# 14. What runs when the container starts. --host 0.0.0.0 = accept connections from outside the
#     container (127.0.0.1 would only accept calls from inside it). Written as a JSON list
#     ("exec form") so Ctrl+C / `docker stop` reach uvicorn directly and it shuts down cleanly.
#     GEMINI_API_KEY is NOT in the image: pass it when you run it, with --env-file .env
CMD ["uvicorn", "ragagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
