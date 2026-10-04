"""Paths and settings in one place. Everything stays inside the project folder on D:."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Model downloads (sentence-transformers, later) go to D:, not C:\Users\...\.cache
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
# Windows without Developer Mode can't make symlinks; HF then copies files instead. Harmless, so hide the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATA = ROOT / "data"
RAW = DATA / "raw"            # original public documents (pdf / htm)
TEXT = DATA / "text"          # plain text made from them, one .txt per document
FULLTEXT = DATA / "fulltext"  # same, but EVERY page (RAG indexes whole documents)
LABELS = DATA / "labels"      # hand labels (the "answer key")
MANIFEST = DATA / "manifest.csv"

OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"     # what the model extracted
REVIEW_QUEUE = OUTPUTS / "review_queue"   # documents a human must look at
RESULTS = ROOT / "evals" / "results"      # scored eval runs (committed)
CHROMA = ROOT / "chroma"                  # vector index (rebuilt by `python -m ragagent.rag.ingest`)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
# Free tier (checked 2026-09-30): the 3.x Flash-Lite models give ~15 requests/min and ~500/day EACH,
# the bigger Flash models only ~20/day, and new keys can no longer use the 2.5 models.
# Quota is per model, so the judge (Part 4) uses a different model = a separate 500/day bucket.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "14"))          # stay just under the ~15/min free limit
LLM_CACHE = os.getenv("LLM_CACHE", "1") == "1"            # replay identical requests from .cache/llm
# Local fallback: fits a 6 GB consumer GPU, supports tools + JSON, no "thinking" mode.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507")
# Part 3: the agent's extract_fields tool shows what the model extracts from a document. The switch
# exists so that model output is not shown while a document is being hand-labeled (it would bias the
# label). All 15 labels are finished, so it is on by default; AGENT_EXTRACT_TOOL=0 in .env turns it off.
AGENT_EXTRACT_TOOL = os.getenv("AGENT_EXTRACT_TOOL", "1") == "1"
