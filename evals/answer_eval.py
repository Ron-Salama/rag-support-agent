"""Answer eval: run the whole system on the golden questions and grade every answer.

    python -m evals.answer_eval                  # plain RAG (ragagent.rag.answer) on every golden question
    python -m evals.answer_eval --agent          # the Part 3 agent + verifier instead
    python -m evals.answer_eval --limit 5        # only the first 5 questions: a smoke test, NOT a result
    python -m evals.answer_eval --ids a03,t02    # only these questions
    python -m evals.answer_eval --selftest       # the scoring rules with fake answers + a fake judge (no API calls)

LLM cost per question, plain RAG: 0-1 call to answer (gate 1 refuses for free) + 0-1 judge call
(no judge when the system refused - code already knows what a refusal means). --agent costs
several calls per question. Identical requests come back from .cache/llm for free.

What is measured (code = no LLM needed, judge = see judge.py):
  ANSWERABLE questions
    false refusals     refused although the answer IS in the documents                       (code)
    correct            judge says the answer matches the expected one (a refusal is not correct)
    faithful           judge says every claim is backed by the cited sources   (of the answered)
    citation accuracy  a citation points at the expected doc AND page          (code, of the answered)
  TRAP questions
    refusal accuracy   refused, or answered only to say "not in the documents" (judge)
  ERRORS (LLM down, quota used up) are counted on their own, never silently as "refused".
  The counting rules and why: DECISIONS.md C16.

Plain RAG also records whether the right page was among the retrieved chunks. That splits a
failure into "retrieval never found it" vs "found it, but the answer step missed it" - two
different fixes (chunking/embedding vs prompt/threshold).

Output: evals/results/answers_<stamp>.json - every question, answer, citation and verdict;
evals/judge_agreement.py reads it to check the judge against a human grader.
"""
import argparse
import json
import sys
import time
from datetime import datetime

from evals import golden, judge
from ragagent import config
from ragagent.llm import CACHE_DIR, LLMError
from ragagent.rag import store


# ----------------------------------------------------------------- run one question (both systems -> one shape)
def run_rag(question: str, k: int, llm) -> dict:
    from ragagent.rag.answer import answer

    a = answer(question, k, llm)
    return {"text": a.text, "refused": not a.found, "why": a.refused_reason, "status": None, "error": None,
            "citations": [{key: c[key] for key in ("n", "chunk_id", "doc_id", "page")} for c in a.citations],
            "retrieved": [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "page": h.page, "score": round(h.score, 4)}
                          for h in a.hits]}


def from_agent_result(r: dict) -> dict:
    """answer_with_verification's dict -> the same shape as run_rag.

    The agent cites whole PAGES ([doc_id p.N] = source_id), not numbered chunks. An answer with no
    citation is not an answer (C9), so "needs_human_review" (the answer is hidden) counts as a refusal.
    "retrieved" = every page the worker's tools returned (its trace), to split retrieval vs answer failures.
    """
    citations = [{"n": None, "source_id": c.get("source_id"), "chunk_id": None, "doc_id": c.get("doc_id"),
                  "page": c.get("page")} for c in r.get("citations") or []]
    retrieved = []
    for step in r.get("trace") or []:
        if step.get("agent") != "worker":          # the verifier's entries hold no sources
            continue
        for sid in step.get("sources_used", []):   # source_id "loan_x p.2" -> doc "loan_x", page 2
            doc_id, page = sid.rsplit(" p.", 1)
            page_ref = {"doc_id": doc_id, "page": int(page)}
            if page_ref not in retrieved:          # each page once, in the order the worker saw it
                retrieved.append(page_ref)
    error = r.get("status") == "error"
    return {"text": r.get("answer") or "", "refused": not citations and not error, "why": None,
            "status": r.get("status"), "error": "agent status 'error'" if error else None,
            "citations": citations, "retrieved": retrieved}


def run_agent(question: str) -> dict:
    # Imported only here, so plain-RAG evals work before / without Part 3.
    from ragagent.agent.orchestrate import answer_with_verification

    return from_agent_result(answer_with_verification(question))


# ----------------------------------------------------------------- grading
def grade(item: dict, out: dict, judge_fn) -> dict:
    """Code decides what it can; the judge (judge_fn(item, out) -> Verdict | None) only sees real answers."""
    g = {"cited_right_doc": None, "cited_right_page": None, "retrieved_right_page": None,
         "correct": None, "faithful": None, "reasoning": None, "judged_by": "code"}
    if out["error"]:
        return g
    if item["type"] == "answerable":
        if out["retrieved"] is not None:
            g["retrieved_right_page"] = any(h["doc_id"] == item["doc_id"] and h["page"] in item["pages"]
                                            for h in out["retrieved"])
        if not out["refused"]:
            g["cited_right_doc"] = any(c["doc_id"] == item["doc_id"] for c in out["citations"])
            g["cited_right_page"] = any(c["doc_id"] == item["doc_id"] and c["page"] in item["pages"]
                                        for c in out["citations"])
    if out["refused"]:
        g["correct"] = item["type"] == "trap"   # refusing is right for a trap, wrong for an answerable question
        return g
    verdict = judge_fn(item, out)
    if verdict is None:
        g["judged_by"] = "judge_error"
        return g
    g.update(correct=verdict.correct, faithful=verdict.faithful, reasoning=verdict.reasoning, judged_by="llm")
    return g


def ratio(n: int, d: int) -> dict:
    return {"n": n, "of": d, "rate": round(n / d, 3) if d else None}


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] == "answerable"]
    traps = [r for r in rows if r["type"] == "trap"]
    answered = [r for r in ans if not r["system"]["refused"] and not r["system"]["error"]]
    refused = [r for r in ans if r["system"]["refused"]]
    judged = [r for r in answered if r["grade"]["judged_by"] == "llm"]
    return {
        "false_refusals": ratio(len(refused), len(ans)),
        "false_refusals_right_page_retrieved": sum(r["grade"]["retrieved_right_page"] is True for r in refused),
        "correct": ratio(sum(r["grade"]["correct"] is True for r in ans), len(ans)),
        "faithful": ratio(sum(r["grade"]["faithful"] is True for r in judged), len(judged)),
        "citation_right_page": ratio(sum(r["grade"]["cited_right_page"] is True for r in answered), len(answered)),
        "citation_right_doc": ratio(sum(r["grade"]["cited_right_doc"] is True for r in answered), len(answered)),
        "trap_refusal_accuracy": ratio(sum(r["grade"]["correct"] is True for r in traps), len(traps)),
        "traps_refused_by_system": sum(r["system"]["refused"] for r in traps),
        "flagged_for_human_review": sum(r["system"]["status"] == "needs_human_review" for r in rows),
        "errors": sum(r["system"]["error"] is not None for r in rows),
        "judge_errors": sum(r["grade"]["judged_by"] == "judge_error" for r in rows),
    }


def print_summary(s: dict):
    def line(label: str, key: str, note: str = ""):
        v = s[key]
        rate = f"{v['rate']:.0%}" if v["rate"] is not None else "-"
        print(f"  {label:<30} {v['n']:>3}/{v['of']:<3} = {rate:>4}   {note}")

    print("\nANSWERABLE")
    line("false refusals", "false_refusals", f"(right page WAS retrieved in {s['false_refusals_right_page_retrieved']})")
    line("correct (judge)", "correct", "(a refusal counts as not correct)")
    line("faithful (judge)", "faithful", "(of the answers the judge graded)")
    line("cites the expected page", "citation_right_page", "(of the answered)")
    line("cites the expected doc", "citation_right_doc", "(of the answered)")
    print("TRAPS")
    line("refusal accuracy", "trap_refusal_accuracy", f"({s['traps_refused_by_system']} refused by the system itself)")
    print(f"flagged for human review: {s['flagged_for_human_review']}   errors: {s['errors']}   "
          f"judge errors: {s['judge_errors']}")


# ----------------------------------------------------------------- the run
def source_text(c: dict, chunk_texts: dict[str, str]) -> str:
    """Plain RAG cites chunks (-> the chunk's text); the agent cites whole pages (-> the page from data/fulltext)."""
    if c.get("chunk_id"):
        return chunk_texts.get(c["chunk_id"], "(chunk not found in chroma/chunks.json)")
    if not (config.FULLTEXT / f"{c['doc_id']}.txt").exists():
        return f"(no document {c['doc_id']!r})"
    return golden.pages_of(c["doc_id"]).get(c["page"], f"(no page {c['page']} in {c['doc_id']})")


def make_judge_fn(chunk_texts: dict[str, str]):
    """judge_fn(item, out) for grade(): shows the judge the text of every CITED source."""
    judge_llm = judge.get_judge_llm()

    def judge_fn(item: dict, out: dict):
        sources = [{**c, "text": source_text(c, chunk_texts)} for c in out["citations"]]
        return judge.judge(judge_llm, item, out["text"], sources)
    return judge_fn, judge_llm.model


def pick(items: list[dict], ids: str | None, limit: int | None) -> list[dict]:
    if ids:
        wanted = [i.strip() for i in ids.split(",")]   # "a03, t02" works too
        items = [it for it in items if it["id"] in wanted]
    return items[:limit] if limit is not None else items   # careful: --limit 0 must mean 0, not "all"


def cache_size() -> int:
    """Each NEW (not replayed) LLM answer adds one file to .cache/llm, so before/after = live calls made."""
    return len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0


def main(args):
    items = pick(golden.load_checked(), args.ids, args.limit)
    partial = bool(args.ids or args.limit)
    if not items:
        print(f"No golden questions selected (--ids {args.ids!r}, --limit {args.limit!r}): nothing to run.")
        return False
    chunk_texts = {c["chunk_id"]: c["text"] for c in store.load_numpy()[1]}
    judge_fn, judge_model = make_judge_fn(chunk_texts)
    answer_llm = None
    if not args.agent:
        from ragagent.llm import get_llm
        answer_llm = get_llm()
    cache_before, rows, stopped = cache_size(), [], None

    for it in items:
        t0 = time.time()
        try:
            out = run_agent(it["question"]) if args.agent else run_rag(it["question"], args.k, answer_llm)
            g = grade(it, out, judge_fn)
        except LLMError as e:   # bad key / daily quota: every next question would fail too - stop, keep what we have
            stopped = str(e)
            print(f"{it['id']:<4} STOPPED: {e}")
            break
        rows.append({**{k: it.get(k) for k in ("id", "type", "question", "answer", "why_unanswerable", "doc_id", "pages")},
                     "system": out, "grade": g, "seconds": round(time.time() - t0, 1),
                     "sources": [{**c, "text": source_text(c, chunk_texts)} for c in out["citations"]]})
        what = "ERROR" if out["error"] else "REFUSED" if out["refused"] else "answered"
        print(f"{it['id']:<4} {it['type']:<10} {what:<8} cites page={g['cited_right_page']!s:<5} "
              f"correct={g['correct']!s:<5} faithful={g['faithful']!s:<5} {time.time() - t0:5.1f}s")

    summary = summarize(rows)
    # With LLM_CACHE=0 nothing is written to .cache/llm, so counting files would always say 0: unknown instead.
    live = cache_size() - cache_before if config.LLM_CACHE else None
    if partial:
        print(f"\nPARTIAL RUN ({len(rows)} of {len(golden.load())} questions): a smoke test, NOT a reportable result.")
    print_summary(summary)
    if live is None:
        print("live LLM calls: unknown (LLM_CACHE=0, so every LLM request was a live call)")
    else:
        print(f"live LLM calls: {live} (new files in .cache/llm during the run - replayed answers are free; "
              f"another program using the LLM at the same time is counted too)")
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS / f"answers_{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps({
        "date": f"{datetime.now():%Y-%m-%d %H:%M}", "command": "python -m evals.answer_eval " + " ".join(sys.argv[1:]),
        "mode": "agent" if args.agent else "rag", "golden": golden.default_path().name, "partial": partial,
        "stopped_early": stopped, "answer_model": answer_llm.model if answer_llm else "see ragagent/agent",
        "judge_model": judge_model, "k": None if args.agent else args.k, "live_llm_calls": live,
        "summary": summary, "questions": rows}, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")
    return stopped is None


# ----------------------------------------------------------------- self-test (fake answers, fake judge)
def selftest():
    def out(refused=False, cites=(), retrieved=(), error=None, status=None) -> dict:
        return {"text": "x", "refused": refused, "why": None, "status": status, "error": error,
                "citations": [{"n": 1, "chunk_id": f"{d}:p{p}:c1", "doc_id": d, "page": p} for d, p in cites],
                "retrieved": [{"chunk_id": "c", "doc_id": d, "page": p, "score": 0.7} for d, p in retrieved]}

    item = {"id": "a1", "type": "answerable", "doc_id": "doc", "pages": [2, 3]}
    trap = {"id": "t1", "type": "trap"}
    calls = []

    def fake_judge(verdict):
        def judge_fn(it, o):
            calls.append(it["id"])
            return verdict
        return judge_fn

    yes = judge.Verdict(reasoning="ok", correct=True, faithful=True)
    no = judge.Verdict(reasoning="wrong", correct=False, faithful=False)
    g_refused = grade(item, out(refused=True, retrieved=[("doc", 2)]), fake_judge(yes))
    judge_calls_after_refusal = len(calls)
    g_good = grade(item, out(cites=[("doc", 3)]), fake_judge(yes))
    g_wrong_doc = grade(item, out(cites=[("other", 3)]), fake_judge(no))
    g_trap_refused = grade(trap, out(refused=True), fake_judge(no))
    g_trap_answered = grade(trap, out(cites=[("doc", 1)]), fake_judge(no))
    g_error = grade(item, out(error="boom"), fake_judge(yes))
    g_judge_err = grade(item, out(cites=[("doc", 2)]), fake_judge(None))
    rows = [{"type": t, "system": o, "grade": g} for t, o, g in [
        ("answerable", out(refused=True), g_refused), ("answerable", out(cites=[("doc", 3)]), g_good),
        ("answerable", out(cites=[("other", 3)]), g_wrong_doc), ("trap", out(refused=True), g_trap_refused),
        ("trap", out(cites=[("doc", 1)]), g_trap_answered), ("answerable", out(error="boom"), g_error)]]
    s = summarize(rows)
    # the shape ragagent.agent.orchestrate returns: page citations + a trace of what the worker saw
    agent = from_agent_result({"answer": "It is 2% [d p.2].", "status": "verified",
                               "citations": [{"source_id": "d p.2", "doc_id": "d", "page": 2, "source_url": "u"}],
                               "trace": [{"agent": "worker", "sources_used": ["d p.2", "e p.1"]},
                                         {"agent": "verifier", "verdict": {}}]})
    review = from_agent_result({"answer": "I could not verify an answer...", "status": "needs_human_review",
                                "citations": [], "trace": []})
    cases = [
        ("refused answerable: not correct, no judge call", g_refused["correct"] is False
         and judge_calls_after_refusal == 0 and g_refused["retrieved_right_page"] is True),
        ("good answer: right page, judge verdict used", g_good["cited_right_page"] and g_good["correct"]
         and g_good["judged_by"] == "llm"),
        ("wrong document cited", g_wrong_doc["cited_right_page"] is False and g_wrong_doc["cited_right_doc"] is False),
        ("refused trap = correct, no judge call", g_trap_refused["correct"] is True and calls.count("t1") == 1),
        ("answered trap -> judge decides", g_trap_answered["correct"] is False),
        ("error is not a refusal", g_error["correct"] is None and s["errors"] == 1),
        ("judge error recorded", g_judge_err["judged_by"] == "judge_error" and g_judge_err["correct"] is None),
        ("false refusals 1/4 (error counts in the total)", s["false_refusals"] == {"n": 1, "of": 4, "rate": 0.25}),
        ("citation accuracy over the answered", s["citation_right_page"] == {"n": 1, "of": 2, "rate": 0.5}),
        ("trap refusal accuracy 1/2", s["trap_refusal_accuracy"]["rate"] == 0.5),
        ("agent result -> same shape", agent["refused"] is False and agent["citations"][0]["page"] == 2
         and agent["retrieved"] == [{"doc_id": "d", "page": 2}, {"doc_id": "e", "page": 1}]),
        ("agent needs_human_review = refused + flagged", review["refused"] and review["status"] == "needs_human_review"),
        ("agent without citations = refused", from_agent_result({"answer": "Not found.", "status": "verified"})["refused"]),
        ("agent page citation -> page text for the judge",
         source_text({"chunk_id": None, "doc_id": "inv_sammy", "page": 1}, {}).startswith("Sammy Maystone")),
        ("agent status 'error' = error", from_agent_result({"status": "error"})["error"] is not None),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in cases)}/{len(cases)} passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    p = argparse.ArgumentParser(description="Grade the system's answers on the golden set.")
    p.add_argument("--agent", action="store_true", help="evaluate the Part 3 agent instead of plain RAG")
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    p.add_argument("--ids", help="comma-separated question ids, e.g. a03,t02")
    p.add_argument("--k", type=int, default=5, help="chunks retrieved per question (plain RAG)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        sys.exit(0 if main(a) else 1)
