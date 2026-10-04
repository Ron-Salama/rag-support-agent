"""The second agent: a strict checker that reads the worker's answer and ALL the evidence it saw.

    python -m ragagent.agent.verifier --selftest     # test the verdict rules with a fake judge (no API calls)

Why a second agent? The worker can still invent a number, cite the wrong page, or refuse when
the answer was right there in a search result. A second LLM call whose ONLY job is "is every
claim in this answer backed by this evidence?" catches many of those. It is the LLM-as-judge
idea (Part 4) used live, on every answer, instead of only in an offline eval.

Design points:
  - The judge sees exactly what the worker saw (its tool results), not the whole document set:
    "supported" means "backed by this evidence", which is something it can actually check.
  - It runs on a DIFFERENT model (config.GEMINI_JUDGE_MODEL): a model tends to rate its own
    answers generously, and on the free tier a second model has its own daily quota.
  - Structured output (generate_json + the Verdict schema), like Parts 1 and 2.
  - Code checks around the LLM, the same idea as Part 1 (Pydantic + check_rules):
        the worker called no tool          -> unsupported, no LLM call needed
        a citation no tool ever returned   -> unsupported, whatever the judge says
        an answer (not a refusal) with no [doc_id p.N] citation -> unsupported (it can't be traced)
        the judge's reply is not valid JSON -> unsupported (fail closed)
        "supported" is recomputed in code: true only if no unsupported claims AND citations ok
"""
import json
import sys

from pydantic import BaseModel, Field, ValidationError

from ragagent import config
from ragagent.agent.loop import NOT_FOUND
from ragagent.agent.tools import cited_sources, sources_in

SYSTEM = """You are a strict fact-checker for answers about financial documents. Another assistant answered a
question using search tools. You get the question, its answer, and ALL the evidence the tools returned to it.
Decide whether the answer is fully supported by that evidence.
Rules:
- Judge ONLY against the evidence. Do not use outside knowledge: a claim that may be true in the real world
  but is not in the evidence is unsupported.
- Check every number, date, percentage, name and amount: it must appear in the evidence with the same value
  (formatting such as $ signs or commas may differ). A claim that goes further than the evidence is unsupported.
- unsupported_claims: quote each unsupported claim from the answer. Empty list if there are none.
- citations_ok: true only if every citation like [doc_id p.N] names a source in the evidence whose text backs
  the claim it is attached to. A factual answer without citations is not ok.
- A refusal ("I could not find this in the documents.") is correct when the evidence does not contain the answer:
  then unsupported_claims is empty and citations_ok is true. If the evidence DOES contain the answer, the refusal
  is wrong: put "refused, but the evidence answers it: <which source>" in unsupported_claims.
- supported: true only if unsupported_claims is empty and citations_ok is true."""


class Verdict(BaseModel):
    """The verifier's answer - also the JSON schema the judge model must fill in.

    Field order matters: the model writes its JSON top to bottom, so it lists the problems
    BEFORE it commits to supported true/false (a small "think first, then decide").
    """
    unsupported_claims: list[str] = Field(default_factory=list, description="Each claim of the answer the evidence "
                                          "does not back, quoted from the answer. Empty list if none.")
    citations_ok: bool = Field(description="true if every [doc_id p.N] citation points to evidence that backs its claim.")
    reason: str = Field(description="One or two sentences explaining the verdict.")
    supported: bool = Field(description="true only if unsupported_claims is empty and citations_ok is true.")


def format_evidence(steps: list[dict]) -> str:
    """The worker's tool calls and results as readable text: sources as '[source_id] text', the rest as JSON."""
    blocks = []
    for n, step in enumerate(steps, start=1):
        result = step["result"]
        items = list(result.get("sources", []))   # search_docs hits
        if "source_id" in result:                 # a get_page result is itself one source
            items.append(result)
        if items:
            body = "\n".join(f"[{i['source_id']}] {i['text']}" for i in items)
        else:                                     # list_documents, errors, empty searches: show as JSON
            body = json.dumps(result, ensure_ascii=False, default=str)
        blocks.append(f"--- tool call {n}: {step['tool']}({json.dumps(step['args'], ensure_ascii=False)})\n{body}")
    return "\n\n".join(blocks)


def build_prompt(question: str, answer: str, steps: list[dict]) -> str:
    return (f"Question: {question}\n\nAnswer to check:\n{answer}\n\n"
            f"Evidence (everything the tools returned to the assistant):\n{format_evidence(steps)}")


def is_refusal(answer: str) -> bool:
    """Does the answer start with the worker's "not found" sentence? (A refusal has nothing to cite.)"""
    return answer.strip().lower().startswith(NOT_FOUND.rstrip(".").lower())


def verify(llm, question: str, answer: str, evidence: list[dict]) -> Verdict:
    """evidence = the worker's steps (AgentResult.steps): every tool call and what it returned. 0-1 LLM calls."""
    if not evidence:
        return Verdict(citations_ok=False, supported=False,
                       reason="The worker called no tool, so nothing from the documents backs this answer.")
    raw = llm.generate_json(SYSTEM, build_prompt(question, answer, evidence), Verdict)
    try:
        verdict = Verdict.model_validate_json(raw)
    except ValidationError:
        return Verdict(citations_ok=False, supported=False, reason="The verifier's reply was not valid JSON (fail closed).")

    unseen = [s for s in cited_sources(answer) if s not in sources_in(evidence)]
    if unseen:   # code knows exactly which pages the tools returned - no need to trust the judge on this
        verdict.unsupported_claims.append(f"cites {', '.join(unseen)}, which no tool returned")
        verdict.citations_ok = False
    if not cited_sources(answer) and not is_refusal(answer):   # a fact with no source: nobody can check it
        verdict.unsupported_claims.append("the answer cites no source like [doc_id p.N]")
        verdict.citations_ok = False
    verdict.supported = verdict.supported and verdict.citations_ok and not verdict.unsupported_claims
    return verdict


def judge_llm():
    """The verifier's model: Gemini's judge model (its own free quota), or the same local model on Ollama."""
    from ragagent.llm import GeminiLLM, get_llm

    if config.LLM_PROVIDER == "gemini":
        return GeminiLLM(config.GEMINI_JUDGE_MODEL)
    return get_llm()


# ----------------------------------------------------------------- self-test with a fake judge
def selftest():
    from ragagent.extract import FakeLLM   # plays back scripted answers, counts calls

    evidence = [{"step": 1, "tool": "search_docs", "args": {"query": "late charge"},
                 "result": {"sources": [{"source_id": "loan_x p.2", "text": "A late charge of 4% applies."}]}}]
    good = "The late charge is 4% [loan_x p.2]."

    def judge(supported: bool, claims: list[str] = (), citations_ok: bool = True) -> str:
        return Verdict(unsupported_claims=list(claims), citations_ok=citations_ok, reason="r",
                       supported=supported).model_dump_json()

    cases = [  # name, answer, evidence, judge replies, check(verdict, fake) -> bool
        ("supported answer -> supported", good, evidence, [judge(True)],
         lambda v, f: v.supported and f.calls == 1),
        ("wrong number -> unsupported", "It is 5% [loan_x p.2].", evidence, [judge(False, ["It is 5%"])],
         lambda v, f: not v.supported and v.unsupported_claims == ["It is 5%"]),
        ("no tool called -> unsupported, no LLM call", good, [], [],
         lambda v, f: not v.supported and f.calls == 0),
        ("cites a page no tool returned", "It is 4% [loan_x p.9].", evidence, [judge(True)],
         lambda v, f: not v.supported and not v.citations_ok and "loan_x p.9" in v.unsupported_claims[0]),
        ("judge contradicts itself -> unsupported", good, evidence, [judge(True, ["It is 4%"])],
         lambda v, f: not v.supported),
        ("invalid JSON from judge -> fail closed", good, evidence, ["Looks fine to me!"],
         lambda v, f: not v.supported and "JSON" in v.reason),
        ("justified refusal -> supported", "I could not find this in the documents.", evidence, [judge(True)],
         lambda v, f: v.supported),
        ("fact with no citation -> unsupported", "The late charge is 4%.", evidence, [judge(True)],
         lambda v, f: not v.supported and not v.citations_ok and "cites no source" in v.unsupported_claims[0]),
        ("prompt shows evidence as [source_id] text", good, evidence, [judge(True)],
         lambda v, f: "[loan_x p.2] A late charge of 4% applies." in build_prompt("q", good, evidence)),
    ]
    passed = 0
    for name, answer, steps, replies, check in cases:
        fake = FakeLLM(replies)
        try:
            v = verify(fake, "What late charge applies?", answer, steps)
            ok, got = check(v, fake), f"supported={v.supported} citations_ok={v.citations_ok} claims={v.unsupported_claims}"
        except Exception as e:  # noqa: BLE001 - show any crash as a failed case
            ok, got = False, f"crashed: {type(e).__name__}: {e}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<42} | {got}")
    print(f"\n{passed}/{len(cases)} passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        print(__doc__)
