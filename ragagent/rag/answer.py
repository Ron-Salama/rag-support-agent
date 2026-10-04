"""Cited answers: question -> retrieve -> LLM answers ONLY from the retrieved chunks, citing them.

    python -m ragagent.rag.answer "What late charge applies if a payment is late?"
    python -m ragagent.rag.answer --selftest      # test the refusal + citation rules with a fake LLM

The flow:
    question ──> search() ──> top-k chunks, each with a similarity score
                                   │
           best score < MIN_SCORE?  yes ──> REFUSE ("nothing in the documents is about this")
                                   │ no                         (gate 1: costs no LLM call)
           prompt = question + numbered sources [S1] (doc, page) ... [Sk]
                                   │
           LLM returns JSON {found, answer, cited}    (structured output, like Part 1)
                                   │
           found = false?           yes ──> REFUSE        (gate 2: the model read them, no answer)
           cites [S9] of 5 sources? drop it, say so       (citation check, done in code)
           no real citation left?   yes ──> REFUSE        (an answer we can't trace is not allowed)
                                   │
           Answer: text with [S1]-style citations + the doc / page / URL of each cited source

Why refuse at all? For a lender a confident WRONG answer is far worse than "I don't know".
Gate 1 catches clearly off-topic questions for free; gate 2 catches on-topic questions the
documents simply don't answer (similar words, no actual answer).
"""
import re
import sys
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from ragagent.rag.retrieve import Hit, search

MIN_SCORE = 0.60   # gate 1 threshold on the best chunk's cosine similarity - see DECISIONS.md
REFUSAL = "I could not find the answer to this in the documents."
CITATION = re.compile(r"\[S(\d+)\]")

SYSTEM = """You answer questions about financial documents (loan notes, appraisals, financial statements, invoices).
Rules:
- Use ONLY the numbered sources you are given. Never use outside knowledge and never guess.
- Cite the source of every fact inline, right after it, like [S2]. Several sources: [S1][S3].
- If the sources do not contain the answer, set found to false and leave answer empty.
- If sources from different documents give different answers, say which document says what, citing each.
- Answer in 1-4 short sentences. Copy numbers exactly as printed."""


class LLMAnswer(BaseModel):
    """The JSON shape the model must return (sent to it as the response schema)."""
    found: bool = Field(description="true only if the sources actually contain the answer to the question.")
    answer: str = Field(description="Short answer with inline citations like [S1]. Empty string if found is false.")
    cited: list[int] = Field(default_factory=list, description="Numbers of the sources the answer uses, e.g. [1, 3].")


@dataclass
class Answer:
    text: str
    found: bool
    refused_reason: str | None              # why we refused (None when found is True)
    citations: list[dict] = field(default_factory=list)   # {n, chunk_id, doc_id, page, source_url}
    hits: list[Hit] = field(default_factory=list)         # everything retrieved (for debugging / evals)


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(f"[S{n}] ({h.doc_id}, page {h.page})\n{h.text}" for n, h in enumerate(hits, start=1))
    return f"Question: {question}\n\nSources:\n{sources}"


def refuse(reason: str, hits: list[Hit]) -> Answer:
    return Answer(text=REFUSAL, found=False, refused_reason=reason, citations=[], hits=hits)


def check_citations(text: str, cited: list[int], n_sources: int) -> tuple[str, list[int]]:
    """Keep only citation numbers that exist (1..n_sources); remove the others from the text and say so."""
    used = set(cited) | {int(n) for n in CITATION.findall(text)}
    valid = sorted(n for n in used if 1 <= n <= n_sources)
    invalid = sorted(used - set(valid))
    for n in invalid:
        text = text.replace(f"[S{n}]", "")
    if invalid:
        text += f" (Removed citation(s) {', '.join(f'[S{n}]' for n in invalid)}: no such source.)"
    return text.strip(), valid


def answer_from_hits(question: str, hits: list[Hit], llm=None) -> Answer:
    """Everything after retrieval: gate 1, prompt, LLM, gate 2, citation check."""
    if not hits or hits[0].score < MIN_SCORE:
        best = f"{hits[0].score:.3f}" if hits else "none"
        return refuse(f"no source is similar enough to the question (best score {best} < {MIN_SCORE})", hits)
    if llm is None:  # created only now, so a gate-1 refusal needs no API key and no network
        from ragagent.llm import get_llm
        llm = get_llm()
    raw = llm.generate_json(SYSTEM, build_prompt(question, hits), LLMAnswer)
    try:
        out = LLMAnswer.model_validate_json(raw)
    except ValidationError:
        return refuse("the model's reply was not valid JSON", hits)
    if not out.found or not out.answer.strip():
        return refuse("the model found no answer in the retrieved sources", hits)
    text, valid = check_citations(out.answer, out.cited, len(hits))
    if not valid:
        return refuse("the model's answer did not cite any real source", hits)
    citations = [{"n": n, "chunk_id": hits[n - 1].chunk_id, "doc_id": hits[n - 1].doc_id,
                  "page": hits[n - 1].page, "source_url": hits[n - 1].source_url} for n in valid]
    return Answer(text=text, found=True, refused_reason=None, citations=citations, hits=hits)


def answer(question: str, k: int = 5, llm=None) -> Answer:
    """The public entry point (used by the API and the agent): retrieve k chunks, then answer."""
    return answer_from_hits(question, search(question, k), llm)


# ----------------------------------------------------------------- self-test with a fake LLM
def selftest():
    from ragagent.extract import FakeLLM   # plays back scripted answers, counts calls

    def hit(n: int, score: float) -> Hit:
        return Hit(f"doc{n}:p{n}:c1", f"doc{n}", "loan_agreement", n, f"text {n}", score, f"http://example/{n}")

    hits = [hit(1, 0.80), hit(2, 0.70)]

    def reply(found: bool, text: str, cited: list[int]) -> str:
        return LLMAnswer(found=found, answer=text, cited=cited).model_dump_json()

    cases = [  # name, hits, scripted LLM replies, check(answer, fake) -> bool
        ("low score -> refuse, no LLM call", [hit(1, 0.40)], [],
         lambda a, f: not a.found and f.calls == 0 and "best score" in a.refused_reason),
        ("no hits at all -> refuse", [], [], lambda a, f: not a.found and f.calls == 0),
        ("good answer -> cited", hits, [reply(True, "The late charge is 2% [S2].", [2])],
         lambda a, f: a.found and [c["n"] for c in a.citations] == [2] and a.citations[0]["page"] == 2),
        ("made-up [S7] dropped, noted", hits, [reply(True, "It is 2% [S1][S7].", [1, 7])],
         lambda a, f: a.found and [c["n"] for c in a.citations] == [1] and "[S7]" not in a.text.split("(")[0]
         and "Removed" in a.text),
        ("model says not found -> refuse", hits, [reply(False, "", [])],
         lambda a, f: not a.found and "no answer" in a.refused_reason),
        ("only fake citations -> refuse", hits, [reply(True, "It is 2% [S9].", [9])],
         lambda a, f: not a.found and "cite" in a.refused_reason),
        ("invalid JSON -> refuse", hits, ["Sure! The answer is 2%."],
         lambda a, f: not a.found and "JSON" in a.refused_reason),
    ]
    passed = 0
    for name, case_hits, replies, check in cases:
        fake = FakeLLM(replies)
        try:
            a = answer_from_hits("What is the late charge?", case_hits, fake)
            ok, got = check(a, fake), f"found={a.found} reason={a.refused_reason!r} text={a.text!r}"
        except Exception as e:  # noqa: BLE001 - show any crash as a failed case
            ok, got = False, f"crashed: {type(e).__name__}: {e}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<34} | {got}")
    print(f"\n{passed}/{len(cases)} passed")


def show(a: Answer):
    print(f"\n{a.text}\n")
    if not a.found:
        print(f"REFUSED: {a.refused_reason}")
    for c in a.citations:
        print(f"  [S{c['n']}] {c['doc_id']}, page {c['page']}  ({c['chunk_id']})  {c['source_url']}")
    print("\nRetrieved:")
    for n, h in enumerate(a.hits, start=1):
        print(f"  S{n}  score {h.score:.3f}  {h.chunk_id}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif args:
        show(answer(" ".join(args)))
    else:
        print(__doc__)
