"""The agent's tools: plain Python functions the model can ASK us to run, each with a spec.

    python -m ragagent.agent.tools --selftest    # check every tool and its error answers (no LLM calls)
    python -m ragagent.agent.tools               # print the tool specs exactly as the model sees them

A tool = a name + a description + a JSON schema of its arguments (together: the "spec", sent to
the model) + the Python function we run when the model asks for it. The description IS part of
the prompt: it is the only thing that tells the model when a tool is useful and how to call it.

Rules every tool follows:
  - It returns a plain dict that json.dumps can turn into text (the result goes back to the model as text).
  - Expected problems (unknown doc_id, page out of range) come back as {"error": "..."} with a hint
    instead of raising, so the model can read the error and fix its call (e.g. call list_documents
    first). The loop ALSO catches anything a tool raises: a second safety net, for bugs.
  - Long text is trimmed, so one tool result can't flood the model's context window.
  - Every page a tool returns has a source_id like "loan_note_jshighland p.2". The agent cites
    exactly that, as [loan_note_jshighland p.2], and the verifier checks those citations.
"""
import json
import re
import sys
from dataclasses import dataclass
from typing import Callable

from ragagent import config
from ragagent.labels import manifest_rows
from ragagent.rag.answer import MIN_SCORE
from ragagent.rag.chunk import split_pages
from ragagent.rag.retrieve import search

SEARCH_TEXT_CHARS = 700   # per search hit (a chunk is at most 800 characters)
PAGE_TEXT_CHARS = 4000    # per get_page call (9 in 10 pages are shorter than ~3,600 characters)
MAX_K = 10                # most chunks one search may return
CITE = re.compile(r"\[([^\[\]]+)\]")                  # one [...] block in an answer
SOURCE = re.compile(r"([A-Za-z0-9_\-]+) p\.\s?(\d+)")   # "doc_id p.N" inside such a block


def source_id(doc_id: str, page: int) -> str:
    return f"{doc_id} p.{page}"


def trim(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rstrip() + " [...]"


def _documents() -> dict[str, dict]:
    return {row["doc_id"]: row for row in manifest_rows()}


def _pages(doc_id: str) -> dict[int, str]:
    """{page number: text} of one document, from data/fulltext (the same text the search index was built from)."""
    return dict(split_pages((config.FULLTEXT / f"{doc_id}.txt").read_text(encoding="utf-8")))


def _unknown_doc(doc_id) -> dict:
    return {"error": f"unknown doc_id {doc_id!r} - call list_documents to see the valid ids"}


# ----------------------------------------------------------------- the tools
def search_docs(query: str, doc_id: str | None = None, k: int = 5) -> dict:
    """Part 2's search (Chroma), with the hits trimmed and labelled with the source_id to cite."""
    if doc_id and doc_id not in _documents():
        return _unknown_doc(doc_id)
    hits = search(query, max(1, min(int(k), MAX_K)), doc_id=doc_id or None)
    out = {"query": query, "sources": [
        {"source_id": source_id(h.doc_id, h.page), "doc_id": h.doc_id, "doc_type": h.doc_type, "page": h.page,
         "score": round(h.score, 3), "text": trim(h.text, SEARCH_TEXT_CHARS)} for h in hits]}
    best = hits[0].score if hits else 0.0
    if best < MIN_SCORE:   # the same threshold as Part 2's refusal gate 1 - here only a hint, the model decides
        out["note"] = (f"Nothing is very similar to this query (best score {best:.2f} < {MIN_SCORE}), so the "
                       "documents may not contain it. Try other words, or say you could not find it.")
    return out


def list_documents() -> dict:
    # The manifest's "notes" column is left out on purpose: those are labeling notes that hint at the answers.
    return {"documents": [{"doc_id": row["doc_id"], "doc_type": row["doc_type"], "pages": len(_pages(row["doc_id"])),
                           "description": f"{row['doc_type'].replace('_', ' ')} - {row['publisher']}"}
                          for row in manifest_rows()]}


def get_page(doc_id: str, page: int) -> dict:
    if doc_id not in _documents():
        return _unknown_doc(doc_id)
    pages, page = _pages(doc_id), int(page)
    if page not in pages:
        return {"error": f"{doc_id} has pages {min(pages)}-{max(pages)}, there is no page {page}"}
    text = pages[page]
    return {"source_id": source_id(doc_id, page), "doc_id": doc_id, "page": page,
            "text": trim(text, PAGE_TEXT_CHARS), "truncated": len(text) > PAGE_TEXT_CHARS}


def extract_fields(doc_id: str, llm=None) -> dict:
    """Part 1 on one document: 1-3 LLM calls. Nothing is saved to outputs/ (the eval files stay untouched)."""
    row = _documents().get(doc_id)
    if row is None:
        return _unknown_doc(doc_id)
    from ragagent.extract import extract_with_retries   # imported here: only needed when this tool runs
    from ragagent.llm import get_llm

    text = (config.TEXT / f"{doc_id}.txt").read_text(encoding="utf-8")
    result = extract_with_retries(llm or get_llm(), row["doc_type"], text)
    return {"doc_id": doc_id, "doc_type": row["doc_type"], "status": result.status,
            "data": result.data, "problems": result.problems}


# ----------------------------------------------------------------- the registry
@dataclass
class Tool:
    name: str
    description: str             # tells the model WHEN to use the tool - part of the prompt
    parameters: dict             # JSON schema of the arguments the model must send
    fn: Callable[..., dict]      # what we run: fn(**args) -> a JSON-serialisable dict

    @property
    def spec(self) -> dict:
        """The part the model sees (the neutral tool format of llm.chat)."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _schema(properties: dict, required: list[str]) -> dict:
    schema = {"type": "object", "properties": properties}
    return {**schema, "required": required} if required else schema


DOC_ID = {"type": "string", "description": "A doc_id from list_documents, e.g. 'loan_note_jshighland'."}

ALL_TOOLS = [
    Tool("search_docs",
         "Semantic search over every page of the documents. Returns the most similar text chunks, each with "
         "a source_id like 'loan_note_jshighland p.2' that you cite as [loan_note_jshighland p.2]. Use specific "
         "words; if nothing relevant comes back, try other wording or a doc_id filter.",
         _schema({"query": {"type": "string", "description": "What to look for, in plain words."},
                  "doc_id": {"type": "string", "description": "Optional: only search this document (see list_documents)."},
                  "k": {"type": "integer", "description": f"How many chunks to return, 1-{MAX_K} (default 5)."}},
                 ["query"]),
         search_docs),
    Tool("list_documents",
         "List every document: doc_id, document type, publisher and number of pages. Use it to find the "
         "doc_id for search_docs or get_page.",
         _schema({}, []), list_documents),
    Tool("get_page",
         f"Read the text of one page (first {PAGE_TEXT_CHARS:,} characters). Use it when a search hit looks "
         "relevant but is cut off, or to read the table or paragraph around it.",
         _schema({"doc_id": DOC_ID, "page": {"type": "integer", "description": "Page number, as in the source_id."}},
                 ["doc_id", "page"]),
         get_page),
    Tool("extract_fields",
         "Run the structured extractor on one document: its key fields (e.g. invoice total, loan principal, "
         "appraised value) as validated JSON with status ok / needs_review / failed. Slow (1-3 extra LLM calls). "
         "Its result has no page numbers: to cite a value, find its page with search_docs.",
         _schema({"doc_id": DOC_ID}, ["doc_id"]), extract_fields),
]


def registry(include_extract: bool | None = None) -> dict[str, Tool]:
    """The tools the agent gets, by name. extract_fields only if config.AGENT_EXTRACT_TOOL is on (see config.py)."""
    include = config.AGENT_EXTRACT_TOOL if include_extract is None else include_extract
    return {t.name: t for t in ALL_TOOLS if include or t.name != "extract_fields"}


# ----------------------------------------------------------------- citations (used by the loop and the verifier)
def cited_sources(answer: str) -> list[str]:
    """'4% [loan_x p.2][loan_y p.3]' or '4% [loan_x p.2; loan_y p.3]' -> ['loan_x p.2', 'loan_y p.3'] (each once)."""
    found = []
    for block in CITE.findall(answer):              # every [...] in the answer
        for doc, page in SOURCE.findall(block):     # every "doc_id p.N" inside it
            found.append(source_id(doc, int(page)))
    return list(dict.fromkeys(found))               # drop repeats, keep the first-seen order


def sources_in(steps: list[dict]) -> list[str]:
    """Every source_id the tools returned in these agent steps (each once, in order) = the pages the agent saw."""
    seen = []
    for step in steps:
        result = step["result"] if isinstance(step["result"], dict) else {}
        seen += [s["source_id"] for s in result.get("sources", [])]
        if "source_id" in result:   # a get_page result
            seen.append(result["source_id"])
    return list(dict.fromkeys(seen))


# ----------------------------------------------------------------- self-test (no LLM calls)
def selftest():
    from ragagent.extract import FakeLLM
    from ragagent.rag import store

    docs = list_documents()["documents"]
    long_page = get_page("fs_sfr_fy2025", 1)
    steps = [{"result": {"sources": [{"source_id": "a p.1"}, {"source_id": "b p.2"}]}},
             {"result": {"source_id": "a p.1", "text": "..."}}, {"result": {"error": "x"}}]
    cases = [
        ("list_documents: 15 docs, pages, no notes", lambda: len(docs) == 15 and all(d["pages"] >= 1 for d in docs)
         and all("notes" not in d for d in docs)),
        ("get_page: source_id + text", lambda: get_page("loan_note_jshighland", 2)["source_id"] == "loan_note_jshighland p.2"),
        ("get_page: long page trimmed", lambda: long_page["truncated"] and len(long_page["text"]) <= PAGE_TEXT_CHARS + 6),
        ("get_page: bad page -> error", lambda: "pages 1-3" in get_page("loan_note_jshighland", 99)["error"]),
        ("get_page: unknown doc -> error", lambda: "list_documents" in get_page("nope", 1)["error"]),
        ("search_docs: unknown doc -> error", lambda: "error" in search_docs("rate", doc_id="nope")),
        ("extract_fields: unknown doc -> error", lambda: "error" in extract_fields("nope", llm=FakeLLM([]))),
        # A fake LLM that never answers valid JSON: checks the plumbing, extracts nothing from the real document.
        ("extract_fields: fake LLM -> 'failed', no crash",
         lambda: extract_fields("inv_sammy", llm=FakeLLM(["oops"] * 3))["status"] == "failed"),
        ("registry: extract_fields only when switched on", lambda: "extract_fields" not in registry(False)
         and "extract_fields" in registry(True) and len(registry(True)) == 4),
        ("cited_sources: both citation styles", lambda: cited_sources("4% [a p.2][b p. 3] and [a p.2; c p.10] [S1]")
         == ["a p.2", "b p.3", "c p.10"]),
        ("sources_in: search + get_page, once", lambda: sources_in(steps) == ["a p.1", "b p.2"]),
        ("results are JSON-serialisable", lambda: bool(json.dumps([docs, long_page]))),
    ]
    if store.index_exists():
        cases += [
            ("search_docs: sources with source_id", lambda: all(
                s["source_id"] == f"{s['doc_id']} p.{s['page']}" and len(s["text"]) <= SEARCH_TEXT_CHARS + 6
                for s in search_docs("late charge on overdue payments", k=3)["sources"])),
            ("search_docs: k capped at MAX_K", lambda: len(search_docs("interest", k=50)["sources"]) == MAX_K),
            ("search_docs: doc_id filter", lambda: {s["doc_id"] for s in search_docs("interest", "loan_note_jshighland")["sources"]}
             == {"loan_note_jshighland"}),
            ("search_docs: off-topic -> note", lambda: "note" in search_docs("chocolate cake recipe")),
        ]
    else:
        print("SKIP  search_docs checks (no index yet - run: python -m ragagent.rag.ingest)")
    passed = 0
    for name, check in cases:
        try:
            ok, got = bool(check()), ""
        except Exception as e:  # noqa: BLE001 - show any crash as a failed case
            ok, got = False, f"crashed: {type(e).__name__}: {e}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<46} {got}")
    print(f"\n{passed}/{len(cases)} passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        print(json.dumps([t.spec for t in registry(True).values()], indent=2))
