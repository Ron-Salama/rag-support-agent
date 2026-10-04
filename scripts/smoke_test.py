"""Smoke test: is a running API alive and does it answer - without any API key or LLM call?

    python scripts/smoke_test.py                          # against http://localhost:8000
    python scripts/smoke_test.py http://localhost:8001    # another address

Run it after `docker run ... rag-support-agent` (or `uvicorn ragagent.api:app`). The CI workflow
(.github/workflows/docker.yml) runs exactly this against the freshly built image.

A "smoke test" is the quickest check that nothing is on fire: switch it on, does smoke come out?
It does not measure quality (that is evals/). It checks:
  1. GET /health      answers, and the search index was built (inside the image: at build time)
  2. GET /documents   lists every document in data/manifest.csv
  3. POST /ask        an off-topic question is refused by gate 1 (similarity score too low).
                      Gate 1 runs BEFORE any LLM call, so this needs no key - yet it proves the
                      embedding model and the Chroma index really work in there, offline.
Only the Python standard library is used, so it also runs outside the project's venv.
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFF_TOPIC = "What is a good recipe for chocolate cake?"   # best score 0.425 vs MIN_SCORE 0.60 (DECISIONS.md C8)


def call(base: str, path: str, body: dict | None = None, timeout: float = 120) -> dict | list:
    """GET (or POST, when body is given) one endpoint -> the parsed JSON reply."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_until_up(base: str, seconds: int = 90) -> bool:
    """The server needs a few seconds to start: ask /health every 2 s until it answers."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            call(base, "/health", timeout=5)
            return True
        except (OSError, ValueError):   # not listening yet (connection refused / reset) or half-started
            time.sleep(2)
    return False


def main(base: str) -> bool:
    if not wait_until_up(base):
        print(f"FAIL  no answer from {base}/health")
        return False
    with open(ROOT / "data" / "manifest.csv", newline="", encoding="utf-8-sig") as f:
        n_docs = len(list(csv.DictReader(f)))
    health = call(base, "/health")
    docs = call(base, "/documents")
    ask = call(base, "/ask", {"question": OFF_TOPIC})
    cases = [
        ("GET /health: status ok, index built", health.get("status") == "ok" and health.get("index_built") is True),
        (f"GET /documents: all {n_docs} documents", len(docs) == n_docs),
        ("POST /ask off-topic: refused by gate 1, no LLM", ask.get("found") is False
         and "best score" in (ask.get("refused_reason") or "") and len(ask.get("hits", [])) == 5),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n/health -> {health}\n/ask    -> refused_reason: {ask.get('refused_reason')!r}")
    print(f"\n{sum(ok for _, ok in cases)}/{len(cases)} passed")
    return all(ok for _, ok in cases)


if __name__ == "__main__":
    try:
        ok = main(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000")
    except urllib.error.HTTPError as e:   # e.g. 503 "No search index yet"
        print(f"FAIL  {e.code} from {e.url}: {e.read().decode(errors='replace')}")
        ok = False
    sys.exit(0 if ok else 1)
