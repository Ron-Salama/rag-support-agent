"""Run-to-run variation of Part 1: extract the same documents N times with the disk cache OFF.

    python -m evals.extraction_variance 3 inv_sammy inv_capozzi_legal    # 3 runs each, 1-3 LLM calls per run

Gemini 3.5 ignores `temperature`, so the same prompt can get a different answer next time. One run that
gets a field right (or wrong) may just be luck: this shows, field by field, in how many of N runs the
answer matched the hand label, so noise is not mistaken for a pipeline fix (or a regression).
Nothing is written to outputs/ (the scored predictions stay untouched); the per-run values are saved to
evals/results/extraction_variance_<time>.json.
"""
import json
import sys
from collections import Counter
from datetime import datetime

from evals.score_extraction import outcome
from ragagent import config
from ragagent.extract import extract_with_retries
from ragagent.labels import manifest_rows
from ragagent.schemas import SCORED_FIELDS


def main(n_runs: int, doc_ids: list[str]):
    from ragagent.llm import get_llm

    config.LLM_CACHE = False   # a cached answer would only repeat the first run
    llm, details = get_llm(), []
    for row in [r for r in manifest_rows() if r["doc_id"] in doc_ids]:
        text = (config.TEXT / f"{row['doc_id']}.txt").read_text(encoding="utf-8")
        label = json.loads((config.LABELS / f"{row['doc_id']}.json").read_text(encoding="utf-8"))
        runs = [extract_with_retries(llm, row["doc_type"], text) for _ in range(n_runs)]
        print(f"\n{row['doc_id']}: status {dict(Counter(r.status for r in runs))}, attempts per run {[r.attempts for r in runs]}")
        for name in SCORED_FIELDS[row["doc_type"]]:
            preds = [(r.data or {}).get(name) for r in runs]
            right = sum(outcome(label.get(name), p, name) in ("correct", "true_null") for p in preds)
            details.append({"doc_id": row["doc_id"], "field": name, "label": label.get(name), "preds": preds,
                            "runs_right": right, "statuses": [r.status for r in runs]})
            if right < n_runs:   # only the fields that are not right every time
                print(f"  {name:<14} right in {right}/{n_runs} runs   label={label.get(name)!r}  preds={preds}")
    out = config.RESULTS / f"extraction_variance_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"model": llm.model, "runs": n_runs, "details": details}, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main(int(sys.argv[1]), sys.argv[2:])
