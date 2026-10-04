"""The HTTP API: the project's functions, callable over the network (FastAPI).

    .\\.venv\\Scripts\\python.exe -m uvicorn ragagent.api:app --port 8000
    then open http://localhost:8000/docs   (an automatic page where you can try every endpoint)

Endpoints:
    GET  /health      is the server up? is the search index built?
    GET  /documents   the document list (from data/manifest.csv)
    POST /ask         {"question": "...", "k": 5}  -> cited answer or refusal       (Part 2, 0-1 LLM calls)
    POST /extract     {"doc_id": "inv_sammy"}      -> validated extracted fields   (Part 1, 1-3 LLM calls)
    POST /agent       {"question": "..."}            -> worker agent + verifier answer (Part 3, 2-14 LLM calls)

FastAPI in a few lines: each decorated function is one endpoint ("route"). The type hints are
the contract: FastAPI reads the JSON body into the Pydantic model (AskRequest), answers bad
input with a 422 error by itself, and turns the dict we return into JSON. Routes are plain
`def` (not `async def`), so FastAPI runs them in a worker thread: a slow LLM call does not
freeze the server for other requests.

Errors: unknown doc_id -> 404; index not built or LLM down / out of quota -> 503.
"""
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragagent import config
from ragagent.agent import orchestrate
from ragagent.extract import extract_with_retries
from ragagent.labels import manifest_rows
from ragagent.llm import LLMError, get_llm
from ragagent.rag import answer as rag_answer
from ragagent.rag import store

app = FastAPI(title="Financial document AI",
              description="Structured extraction + cited question answering over public financial documents.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What late charge applies if a payment is late?"])
    k: int = Field(default=5, ge=1, le=20, description="How many chunks to retrieve and show the model.")


class ExtractRequest(BaseModel):
    doc_id: str = Field(examples=["inv_sammy"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_built": store.index_exists(), "llm_provider": config.LLM_PROVIDER}


@app.get("/documents")
def documents() -> list[dict]:
    return [{k: row[k] for k in ("doc_id", "doc_type", "source_url", "publisher")} for row in manifest_rows()]


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not store.index_exists():
        raise HTTPException(503, "No search index yet - run: python -m ragagent.rag.ingest")
    try:
        return asdict(rag_answer.answer(req.question, req.k))   # dataclass (with its Hits) -> plain dict -> JSON
    except LLMError as e:
        raise HTTPException(503, f"LLM unavailable: {e}") from e


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    """Runs Part 1 on one document and returns the result. Nothing is saved to outputs/ (eval files stay untouched).

    When adding a new document to the eval set, hand-label it BEFORE looking at this output (it would bias the label).
    """
    row = next((r for r in manifest_rows() if r["doc_id"] == req.doc_id), None)
    if row is None:
        raise HTTPException(404, f"Unknown doc_id {req.doc_id!r} - see GET /documents")
    text = (config.TEXT / f"{row['doc_id']}.txt").read_text(encoding="utf-8")
    try:
        result = extract_with_retries(get_llm(), row["doc_type"], text)
    except LLMError as e:
        raise HTTPException(503, f"LLM unavailable: {e}") from e
    return {"doc_id": req.doc_id, **asdict(result)}


# ===================================================================================
# PART 3 - worker agent + verifier (ragagent/agent/orchestrate.py)
# ===================================================================================
class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What late charge applies if a loan payment is late?"])


@app.post("/agent")
def agent(req: AgentRequest) -> dict:
    """The model picks its own tools, a second model checks the answer, one retry, then human review.

    status: "verified" | "revised" (passed on the retry) | "needs_human_review" (answer withheld,
    case saved to outputs/review_queue/). The full trace (every tool call + verdict) is included.
    """
    if not store.index_exists():
        raise HTTPException(503, "No search index yet - run: python -m ragagent.rag.ingest")
    try:
        result = orchestrate.answer_with_verification(req.question)
    except LLMError as e:   # the LLM could not even be set up (e.g. no API key)
        raise HTTPException(503, f"LLM unavailable: {e}") from e
    if result["status"] == "error":   # same rule as /ask: LLM down / out of quota -> 503
        errors = [e.get("error") for e in result["trace"] if e.get("error")]
        raise HTTPException(503, f"LLM unavailable: {errors[-1] if errors else result['answer']}")
    return result
