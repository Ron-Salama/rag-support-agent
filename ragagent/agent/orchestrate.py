"""Worker + verifier: the multi-agent workflow behind POST /agent.

    python -m ragagent.agent.orchestrate "What late charge applies if a loan payment is late?"
    python -m ragagent.agent.orchestrate --selftest     # verified / revised / human review, with fakes

    question ──> WORKER (agent loop + tools) ──> answer + the evidence it saw
                     │  stopped at the step limit ──> human review     LLM failed ──> status "error"
                     ▼
                 VERIFIER (a second model) ──> supported? ── yes ──> status "verified"
                     │ no
                 WORKER again, ONCE, told exactly what the verifier rejected
                     ▼
                 VERIFIER ──> supported? ── yes ──> status "revised"
                     │ no
                 status "needs_human_review": the unverified answer is NOT shown; the whole case
                 is saved to outputs/review_queue/agent_<hash>.json for a person (human-in-the-loop)

Why only ONE retry? Each round costs about 3-5 LLM calls, and if the second try is still not
backed by the documents a third one rarely is: a person should look. Why hide an unverified
answer? Same rule as Part 2: for a lender, "not verified, a person will check" beats a
confident wrong number.

LLM calls per question: worker 1-6 + verifier 0-1, twice if it retries (at most 14) - plus 1-3 for
every extract_fields call (the tool is on by default; config.AGENT_EXTRACT_TOOL switches it off).
"""
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ragagent import config
from ragagent.agent.loop import MAX_STEPS, AgentResult, run_agent
from ragagent.agent.tools import cited_sources, registry
from ragagent.agent.verifier import Verdict, judge_llm, verify
from ragagent.labels import manifest_rows
from ragagent.llm import LLMError, get_llm

NEEDS_REVIEW = ("I could not verify an answer from the documents, so I am not giving one. "
                "The question has been sent to a person for review.")
LLM_DOWN = "The language model is not available right now, so this question could not be answered."


def worker_prompt(question: str, rejected: AgentResult | None = None, verdict: Verdict | None = None) -> str:
    """The question - and on the retry, what the verifier rejected (like Part 1's feedback on a retry)."""
    if verdict is None:
        return question
    claims = "\n".join(f"- {c}" for c in verdict.unsupported_claims) or "- (no single claim named)"
    return (f"{question}\n\nA reviewer checked your previous answer against your tool results and REJECTED it.\n"
            f"Previous answer: {rejected.answer}\nReviewer's reason: {verdict.reason}\nUnsupported claims:\n{claims}\n"
            "Search again and answer using only what the tools return, citing every fact - or reply "
            "'I could not find this in the documents.' if the evidence is not there.")


def citations(answer: str, seen: list[str]) -> list[dict]:
    """The [doc_id p.N] citations in the answer, with each document's source URL."""
    urls = {row["doc_id"]: row["source_url"] for row in manifest_rows()}
    out = []
    for sid in cited_sources(answer):
        doc_id, page = sid.rsplit(" p.", 1)
        out.append({"source_id": sid, "doc_id": doc_id, "page": int(page), "source_url": urls.get(doc_id),
                    "seen_by_worker": sid in seen})
    return out


def save_for_review(case: dict, folder: Path) -> Path:
    """Human-in-the-loop: one JSON file per question a person must look at (same idea as Part 1's queue)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"agent_{hashlib.sha256(case['question'].encode()).hexdigest()[:10]}.json"
    path.write_text(json.dumps(case, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _result(question: str, answer: str, status: str, trace: list, verdicts: list, cites: list | None = None) -> dict:
    return {"question": question, "answer": answer, "status": status, "citations": cites or [],
            "trace": trace, "verdicts": verdicts, "review_file": None}


def answer_with_verification(question: str, llm=None, judge=None, tools=None, max_steps: int = MAX_STEPS,
                             review_dir: Path = config.REVIEW_QUEUE) -> dict:
    """Worker -> verifier -> (one retry) -> verified / revised / needs_human_review / error. Returns the full trace."""
    llm = llm or get_llm()
    judge = judge or judge_llm()
    tools = registry() if tools is None else tools
    trace, verdicts = [], []
    worker, verdict = None, None
    for attempt in (1, 2):   # the first try + at most ONE retry with the verifier's feedback
        worker = run_agent(llm, worker_prompt(question, worker, verdict), tools, max_steps)
        trace.append({"agent": "worker", "attempt": attempt, **asdict(worker)})
        if worker.stopped_reason == "llm_error":
            return _result(question, LLM_DOWN, "error", trace, verdicts)
        if worker.stopped_reason == "step_limit":
            break   # no answer to verify, and a retry would likely run out of steps too: a person looks
        try:
            verdict = verify(judge, question, worker.answer, worker.steps)
        except LLMError as e:
            trace.append({"agent": "verifier", "attempt": attempt, "error": str(e)})
            return _result(question, LLM_DOWN, "error", trace, verdicts)
        verdicts.append(verdict.model_dump())
        trace.append({"agent": "verifier", "attempt": attempt, "verdict": verdict.model_dump()})
        if verdict.supported:
            return _result(question, worker.answer, "verified" if attempt == 1 else "revised", trace, verdicts,
                           citations(worker.answer, worker.sources_used))

    result = _result(question, NEEDS_REVIEW, "needs_human_review", trace, verdicts)
    case = {"question": question, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "worker_model": getattr(llm, "model", "?"), "judge_model": getattr(judge, "model", "?"),
            # why: the verifier's reason, or "step_limit" when the LAST try ran out of steps (even after a rejection)
            "draft_answer": worker.answer,
            "why": verdicts[-1]["reason"] if worker.stopped_reason == "answered" else worker.stopped_reason,
            "trace": trace, "verdicts": verdicts}
    result["review_file"] = str(save_for_review(case, review_dir))
    return result


# ----------------------------------------------------------------- self-test with fakes (no API calls)
class _FailingJudge:
    model = "fake"

    def generate_json(self, system, user, schema):
        raise LLMError("judge quota used up")


def selftest():
    import tempfile

    from ragagent.agent.loop import FakeChatLLM, fake_tools, final, tool_call
    from ragagent.extract import FakeLLM   # the judge only needs generate_json

    search = tool_call("search_docs", {"query": "late charge"})
    right, wrong = final("The late charge is 4% [loan_x p.2]."), final("The late charge is 5% [loan_x p.2].")

    def judge(supported: bool, claims: list[str] = ()) -> str:
        return Verdict(unsupported_claims=list(claims), citations_ok=True, reason="checked", supported=supported).model_dump_json()

    def review_file_ok(r: dict, why: str | None = None) -> bool:
        case = json.loads(Path(r["review_file"]).read_text(encoding="utf-8")) if r["review_file"] else {}
        return "draft_answer" in case and (why is None or case["why"] == why)

    cases = [  # name, worker replies, judge, max_steps, check(result, worker_fake, judge) -> bool
        ("supported 1st time -> verified", [search, right], FakeLLM([judge(True)]), 6,
         lambda r, w, j: r["status"] == "verified" and "4%" in r["answer"] and len(r["verdicts"]) == 1
         and r["citations"][0]["source_id"] == "loan_x p.2" and r["citations"][0]["seen_by_worker"] and r["review_file"] is None),
        ("rejected, fixed on retry -> revised", [search, wrong, search, right],
         FakeLLM([judge(False, ["5%"]), judge(True)]), 6,
         lambda r, w, j: r["status"] == "revised" and "4%" in r["answer"] and len(r["verdicts"]) == 2
         and "REJECTED" in w.requests[2][0]["content"] and "5%" in w.requests[2][0]["content"]),
        ("rejected twice -> needs_human_review", [search, wrong, search, wrong],
         FakeLLM([judge(False, ["5%"]), judge(False, ["5%"])]), 6,
         lambda r, w, j: r["status"] == "needs_human_review" and r["answer"] == NEEDS_REVIEW and review_file_ok(r)
         and j.calls == 2),
        ("step limit -> human review, no verifier call", [search] * 10, FakeLLM([]), 2,
         lambda r, w, j: r["status"] == "needs_human_review" and j.calls == 0 and review_file_ok(r, "step_limit")),
        ("rejected, then retry hits step limit", [search, wrong, search, search], FakeLLM([judge(False, ["5%"])]), 2,
         lambda r, w, j: r["status"] == "needs_human_review" and j.calls == 1 and review_file_ok(r, "step_limit")),
        ("worker LLM fails -> error", [LLMError("daily quota used up")], FakeLLM([]), 6,
         lambda r, w, j: r["status"] == "error" and r["answer"] == LLM_DOWN),
        ("verifier LLM fails -> error", [search, right], _FailingJudge(), 6,
         lambda r, w, j: r["status"] == "error" and "judge quota" in r["trace"][-1]["error"]),
    ]
    passed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, replies, judge_llm_, max_steps, check in cases:
            worker = FakeChatLLM(replies)
            try:
                r = answer_with_verification("What late charge applies?", worker, judge_llm_, fake_tools(), max_steps, Path(tmp))
                ok, got = check(r, worker, judge_llm_), f"status={r['status']} verdicts={len(r['verdicts'])}"
            except Exception as e:  # noqa: BLE001 - show any crash as a failed case
                ok, got = False, f"crashed: {type(e).__name__}: {e}"
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'}  {name:<46} | {got}")
    print(f"\n{passed}/{len(cases)} passed")


def show(result: dict):
    for event in result["trace"]:
        if event["agent"] == "worker":
            calls = " -> ".join(f"{s['tool']}({json.dumps(s['args'], ensure_ascii=False)})" for s in event["steps"])
            print(f"worker   #{event['attempt']}: {calls or '(no tool calls)'}\n"
                  f"             [{event['stopped_reason']}] {event['answer']}")
        elif "verdict" in event:
            v = event["verdict"]
            print(f"verifier #{event['attempt']}: supported={v['supported']} citations_ok={v['citations_ok']} "
                  f"unsupported={v['unsupported_claims']}\n             {v['reason']}")
        else:
            print(f"verifier #{event['attempt']}: ERROR {event['error']}")
    print(f"\nSTATUS: {result['status']}\n{result['answer']}")
    for c in result["citations"]:
        print(f"  [{c['source_id']}] {c['source_url']}")
    if result["review_file"]:
        print(f"Saved for human review: {result['review_file']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif args:
        show(answer_with_verification(" ".join(args)))
    else:
        print(__doc__)
