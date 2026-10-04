"""Calibrating the judge: how often does the LLM judge agree with a human grader?

    python -m evals.judge_agreement export             # latest evals/results/answers_*.json -> evals/judge_calibration.csv
    python -m evals.judge_agreement export <file>      # the same, from a specific answers_*.json
    python -m evals.judge_agreement score              # after a human filled in the human_verdict column
    python -m evals.judge_agreement --selftest         # the counting rules on made-up rows (no API calls)

Why calibrate? You check a kitchen scale with a known 1 kg weight before trusting it, and a
lab thermometer against a reference thermometer. The LLM judge (judge.py) is our measuring
instrument for "is this answer OK?", and a human grader is the reference. If they agree on a
sample, the judge's percentages over the whole golden set mean something. If they don't, those
percentages are noise: fix the judge prompt (or the golden answers) before reporting any number.

The two ways to disagree are NOT equally bad:
  judge OK, human WRONG   the judge lets bad answers through -> every reported score is too HIGH.
                          The dangerous one: it hides real failures behind a good-looking number.
  judge WRONG, human OK   the judge is too strict -> scores are too LOW. Annoying, but safe.

How to fill in the CSV (Excel is fine): for each row read the question, the expected answer, the
system's answer and its sources, then write ok or wrong in human_verdict (ok = correct AND every
claim is backed by the sources; for a trap, ok = it said the documents don't contain the answer
and invented nothing). Do it BEFORE looking at the judge_ columns - hide them. Grade blind:
seeing the judge's verdict first pulls the human verdict toward it, and the agreement then
measures nothing.

Only rows the LLM judge actually graded are exported (refusals are graded by code, not the judge).
With few rows the % is rough: 9 of 10 agreeing could just as well be 7 of 10 next time. Aim for 30+.
"""
import csv
import json
import sys
from pathlib import Path

from evals.judge import source_name
from ragagent.config import RESULTS, ROOT

CSV_PATH = ROOT / "evals" / "judge_calibration.csv"
COLUMNS = ["id", "type", "question", "expected", "system_answer", "sources", "human_verdict", "human_notes",
           "judge_verdict", "judge_correct", "judge_faithful", "judge_reasoning"]
OK_WORDS, WRONG_WORDS = {"ok", "y", "yes", "1", "true", "pass"}, {"wrong", "n", "no", "0", "false", "fail"}


def judge_ok(grade: dict) -> bool:
    """The judge's overall verdict, defined like the human one: correct AND faithful."""
    return bool(grade["correct"] and grade["faithful"])


def rows_from_answers(data: dict) -> list[dict]:
    """answers_*.json (made by answer_eval) -> one CSV row per answer the LLM judge graded."""
    rows = []
    for q in data["questions"]:
        g = q["grade"]
        if g["judged_by"] != "llm":
            continue
        sources = "\n\n".join(f"[{source_name(s)}] {s['doc_id']} p.{s['page']}: {s['text']}" for s in q["sources"])
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"],
                     "expected": q["answer"] if q["type"] == "answerable" else f"NOT ANSWERABLE: {q['why_unanswerable']}",
                     "system_answer": q["system"]["text"], "sources": sources, "human_verdict": "", "human_notes": "",
                     "judge_verdict": "ok" if judge_ok(g) else "wrong", "judge_correct": g["correct"],
                     "judge_faithful": g["faithful"], "judge_reasoning": g["reasoning"]})
    return rows


def read_csv(path: Path) -> list[dict]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:   # -sig: Excel adds a BOM when it saves
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        # Saved as plain "CSV (Comma delimited)", Excel writes the old Windows code page, not UTF-8
        # (a curly quote ’ becomes one byte that is invalid UTF-8). Read it the way Excel wrote it.
        with open(path, newline="", encoding="cp1252", errors="replace") as f:
            return list(csv.DictReader(f))


def export(answers_path: Path | None):
    runs = sorted(RESULTS.glob("answers_*.json"))   # names sort by time stamp: the last one is the newest
    if answers_path is None and not runs:
        print("No evals/results/answers_*.json yet - run first: python -m evals.answer_eval")
        return
    answers_path = answers_path or runs[-1]
    if CSV_PATH.exists() and any(r.get("human_verdict", "").strip() for r in read_csv(CSV_PATH)):
        print(f"{CSV_PATH} already has human verdicts - not overwriting it. Rename or delete it first.")
        return
    data = json.loads(answers_path.read_text(encoding="utf-8"))
    rows = rows_from_answers(data)
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:   # -sig so Excel shows $ and quotes right
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} judged answers from {answers_path.name} ({data['mode']}, {data['judge_model']}) to {CSV_PATH}")
    if data.get("partial"):
        print("NOTE: that run was partial (--limit/--ids): fine as a smoke test, too few rows to calibrate on.")
    print("Next: fill human_verdict with ok / wrong WITHOUT looking at the judge_ columns, then run: "
          "python -m evals.judge_agreement score")


def parse(verdict: str) -> bool | None:
    v = verdict.strip().lower()
    return True if v in OK_WORDS else False if v in WRONG_WORDS else None


def agreement(rows: list[dict]) -> dict:
    """Compare human_verdict with judge_verdict on the rows the human grader filled in."""
    filled = [r for r in rows if parse(r["human_verdict"]) is not None]
    too_lenient = [r["id"] for r in filled if parse(r["judge_verdict"]) and not parse(r["human_verdict"])]
    too_strict = [r["id"] for r in filled if not parse(r["judge_verdict"]) and parse(r["human_verdict"])]
    return {"rows": len(rows), "filled": len(filled), "agree": len(filled) - len(too_lenient) - len(too_strict),
            "judge_ok_human_wrong": too_lenient, "judge_wrong_human_ok": too_strict,
            "unreadable": [r["id"] for r in rows if r["human_verdict"].strip() and parse(r["human_verdict"]) is None]}


def score():
    if not CSV_PATH.exists():
        print(f"No {CSV_PATH} yet - run first: python -m evals.judge_agreement export")
        return
    a = agreement(read_csv(CSV_PATH))
    if not a["filled"]:
        print(f"No human_verdict filled in yet in {CSV_PATH} (write ok or wrong in that column).")
        return
    print(f"Human verdicts filled in: {a['filled']} of {a['rows']} rows.")
    print(f"AGREEMENT: {a['agree']}/{a['filled']} = {a['agree'] / a['filled']:.0%}")
    print(f"  judge OK, human WRONG (judge too lenient - scores look better than they are): "
          f"{len(a['judge_ok_human_wrong'])}  {a['judge_ok_human_wrong']}")
    print(f"  judge WRONG, human OK (judge too strict - scores look worse than they are):   "
          f"{len(a['judge_wrong_human_ok'])}  {a['judge_wrong_human_ok']}")
    if a["unreadable"]:
        print(f"  could not read human_verdict (use ok / wrong) for: {a['unreadable']}")


# ----------------------------------------------------------------- self-test
def selftest():
    def row(qid, judge_says, human_says):
        return {"id": qid, "judge_verdict": judge_says, "human_verdict": human_says}

    rows = [row("a1", "ok", "ok"), row("a2", "ok", "wrong"), row("a3", "wrong", "OK "), row("a4", "wrong", "no"),
            row("a5", "ok", ""), row("a6", "ok", "maybe")]
    a = agreement(rows)
    grade = {"judged_by": "llm", "correct": True, "faithful": False, "reasoning": "r"}
    fake_run = {"questions": [
        {"id": "a1", "type": "answerable", "question": "q", "answer": "A", "system": {"text": "t"}, "grade": grade,
         "sources": [{"n": 1, "doc_id": "d", "page": 2, "text": "src"}]},
        {"id": "t1", "type": "trap", "question": "q", "why_unanswerable": "w", "system": {"text": "t"},
         "grade": {**grade, "judged_by": "code"}, "sources": []}]}
    exported = rows_from_answers(fake_run)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:   # a CSV as Excel saves it with "CSV (Comma delimited)"
        excel_csv = Path(tmp) / "excel.csv"
        excel_csv.write_bytes("id,human_verdict,human_notes\na1,ok,Omega’s answer\n".encode("cp1252"))
        excel_rows = read_csv(excel_csv)
    cases = [
        ("agreement counts filled rows only", a["filled"] == 4 and a["agree"] == 2),
        ("judge OK / human wrong found", a["judge_ok_human_wrong"] == ["a2"]),
        ("judge wrong / human OK found", a["judge_wrong_human_ok"] == ["a3"]),
        ("unreadable verdict reported", a["unreadable"] == ["a6"]),
        ("export: only LLM-judged rows", [r["id"] for r in exported] == ["a1"]),
        ("export: correct but unfaithful = wrong", exported[0]["judge_verdict"] == "wrong"),
        ("export: sources shown with doc + page", exported[0]["sources"] == "[S1] d p.2: src"),
        ("CSV saved by Excel (Windows code page) still reads", excel_rows == [
            {"id": "a1", "human_verdict": "ok", "human_notes": "Omega’s answer"}]),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in cases)}/{len(cases)} passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    args = sys.argv[1:]
    if args[:1] == ["export"]:
        export(Path(args[1]) if len(args) > 1 else None)
    elif args == ["score"]:
        score()
    elif args == ["--selftest"]:
        selftest()
    else:
        print(__doc__)
