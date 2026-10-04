"""Score extracted fields against the hand labels.

    python -m evals.score_extraction                    # score outputs/predictions/ against data/labels/
    python -m evals.score_extraction --rescore FILE     # re-grade the label/prediction pairs saved in an
                                                        # older results FILE with today's grader (0 LLM calls)
    python -m evals.score_extraction --selftest         # check same() / outcome() on made-up pairs

For every scored field of every document, compare the prediction with the label:
    correct     same value (numbers within 0.5%, text compared loosely, addresses by their parts, see same())
    true_null   both empty - the model correctly said "not in the document"
    wrong       both have a value but they differ
    missed      label has a value, the model returned null
    false_fill  label is null but the model invented a value  <- hallucination, the worst kind

Accuracy = (correct + true_null) / all fields. We also report the false-fill rate
separately, because for a lender a confidently invented number is far worse than a blank.
"""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from ragagent.config import LABELS, PREDICTIONS, RESULTS
from ragagent.labels import manifest_rows
from ragagent.schemas import SCORED_FIELDS

COMPANY_WORDS = r"\b(inc|incorporated|llc|l\.l\.c|lp|l\.p|corp|corporation|co|company|ltd|the)\b"
STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
          "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
          "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
          "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
          "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
          "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
          "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
          "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
          "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
          "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy"}
STATE_PATTERN = "|".join(sorted([*STATES, *STATES.values()], key=len, reverse=True))   # longest first
STREET_WORDS = {"street": "st", "avenue": "ave", "road": "rd", "drive": "dr", "boulevard": "blvd", "parkway": "pkwy",
                "highway": "hwy", "lane": "ln", "court": "ct", "place": "pl", "suite": "ste",
                "north": "n", "south": "s", "east": "e", "west": "w"}


def norm_text(s: str) -> str:
    s = s.casefold()
    s = re.sub(COMPANY_WORDS, " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def address_parts(s: str) -> tuple[str, str, str, str]:
    """'1228 Rossmoor Parkway, Walnut Creek, Contra Costa County, California 94595'
    -> ('1228 rossmoor pkwy', 'walnut creek', 'ca', '94595'): street (number + name), city, state, ZIP.
    A part that is not printed comes back as ''. A county is dropped: it is extra detail, not another place."""
    parts = [" ".join(re.sub(r"[^a-z0-9 ]", " ", p.casefold()).split()) for p in s.split(",")]
    parts = [p for p in parts if p and not p.endswith(" county")]
    state = zip_code = ""
    m = re.fullmatch(rf"(.*?)\s*\b({STATE_PATTERN})\s*(\d{{5}})?(?:\s*\d{{4}})?", parts[-1]) if parts else None
    if m:   # the last part ends with a state (+ ZIP); whatever comes before it in that part is the city
        parts[-1:] = [m[1]] if m[1] else []
        state, zip_code = STATES.get(m[2], m[2]), m[3] or ""

    def short(p: str) -> str:   # Parkway = Pkwy, South = S
        return " ".join(STREET_WORDS.get(w, w) for w in p.split())
    return (short(parts[0]) if parts else "", short(parts[-1]) if len(parts) > 1 else "", state, zip_code)


def same_address(label: str, pred: str) -> bool:
    """Street number + street name must match; city, state and ZIP are compared only when BOTH sides print
    them. So "…, CA 94595" = "…, California 94595", and a county part or a missing ZIP is not an error."""
    a, b = address_parts(label), address_parts(pred)
    if " ".join(filter(None, a)) == " ".join(filter(None, b)):   # same words, only the commas differ
        return True
    return bool(a[0]) and a[0] == b[0] and all(x == y for x, y in zip(a[1:], b[1:]) if x and y)


def same(label, pred, field: str = "") -> bool:
    if isinstance(label, bool) or isinstance(pred, bool):
        return label == pred
    if isinstance(label, (int, float)) and isinstance(pred, (int, float)):
        return abs(label - pred) <= max(0.005 * abs(label), 0.01)
    if "address" in field:   # no fuzzy match here: the 85% rule below calls "1228 Rossmoor Pkwy, ..." and
        return same_address(str(label), str(pred))   # "1229 Rossmoor Pkwy, ..." the same (97% similar text)
    a, b = norm_text(str(label)), norm_text(str(pred))
    return a == b or (min(len(a), len(b)) >= 4 and (a in b or b in a)) or SequenceMatcher(None, a, b).ratio() >= 0.85


def outcome(label, pred, field: str = "") -> str:
    if label is None and pred is None:
        return "true_null"
    if label is None:
        return "false_fill"
    if pred is None:
        return "missed"
    return "correct" if same(label, pred, field) else "wrong"


def main():
    rows = [r for r in manifest_rows() if (LABELS / f"{r['doc_id']}.json").exists()]
    details, statuses, model = [], Counter(), None
    for row in rows:
        label = json.loads((LABELS / f"{row['doc_id']}.json").read_text(encoding="utf-8"))
        pred_path = PREDICTIONS / f"{row['doc_id']}.json"
        pred_file = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else {"status": "not_run"}
        statuses[pred_file["status"]] += 1
        model = model or pred_file.get("model")
        pred = pred_file.get("data") or {}
        for field in SCORED_FIELDS[row["doc_type"]]:
            details.append({"doc_id": row["doc_id"], "field": field, "label": label.get(field),
                            "pred": pred.get(field), "outcome": outcome(label.get(field), pred.get(field), field)})
    report(details, statuses, model, len(rows))


def rescore(path: Path):
    """Re-grade an older run with today's grader: same label/prediction pairs, new same(). This separates
    a GRADER fix (only the scoring changed) from a PIPELINE fix (the extraction itself got better)."""
    old = json.loads(path.read_text(encoding="utf-8"))
    details = [{**d, "outcome": outcome(d["label"], d["pred"], d["field"]), "old_outcome": d["outcome"]}
               for d in old["details"]]
    changed = [d for d in details if d["outcome"] != d["old_outcome"]]
    print(f"Re-graded {path.name}: {len(changed)} of {len(details)} fields changed outcome")
    for d in changed:
        print(f"  {d['old_outcome']} -> {d['outcome']:<10} {d['doc_id']:<24} {d['field']}")
    report(details, Counter(old["statuses"]), old["model"], len({d["doc_id"] for d in details}),
           rescored_from=path.name)


def report(details: list[dict], statuses: Counter, model, n_docs: int, rescored_from: str | None = None):
    """Print the summary and save every field's outcome to evals/results/extraction_<time>.json."""
    doc_types = {r["doc_id"]: r["doc_type"] for r in manifest_rows()}
    by_type, by_field = defaultdict(Counter), defaultdict(Counter)
    for d in details:
        by_type[doc_types[d["doc_id"]]][d["outcome"]] += 1
        by_field[f"{doc_types[d['doc_id']]}.{d['field']}"][d["outcome"]] += 1
    total = Counter(d["outcome"] for d in details)
    n = sum(total.values())
    good = total["correct"] + total["true_null"]
    nulls = total["true_null"] + total["false_fill"]
    print(f"\nDocuments scored: {n_docs}   extraction status: {dict(statuses)}   model: {model}")
    print(f"FIELD ACCURACY: {good}/{n} = {good / n:.1%}")
    if nulls:
        print(f"False-fill (hallucination) rate: {total['false_fill']}/{nulls} empty fields = {total['false_fill'] / nulls:.1%}")
    print("\nBy document type:")
    for t, c in by_type.items():
        k = sum(c.values())
        print(f"  {t:<22} {c['correct'] + c['true_null']:>3}/{k:<3} = {(c['correct'] + c['true_null']) / k:.0%}   {dict(c)}")
    print("\nErrors (fix the pipeline or the label, and write down WHY each happened):")
    for d in details:
        if d["outcome"] in ("wrong", "missed", "false_fill"):
            print(f"  {d['outcome']:<10} {d['doc_id']:<24} {d['field']:<20} label={d['label']!r}  pred={d['pred']!r}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = RESULTS / (f"extraction_{stamp}" + (f"_rescore_of_{Path(rescored_from).stem}" if rescored_from else "") + ".json")
    out.write_text(json.dumps({"model": model, "rescored_from": rescored_from, "field_accuracy": good / n,
                               "counts": total, "statuses": statuses, "by_type": by_type, "by_field": by_field,
                               "details": details}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")


def selftest():
    valley_label = "1228 Rossmoor Parkway , Walnut Creek , Contra Costa County , California 94595"
    cases = [   # (name, what same()/outcome() gave, what it must give)
        ("'Inc.' = 'Inc'", same("Deverick & Associates, Inc.", "Deverick & Associates, Inc"), True),
        ("CA = California, county ignored", same(valley_label, "1228 Rossmoor Parkway, Walnut Creek, CA 94595",
                                                 "property_address"), True),
        ("5 != 0.5", same(5, 0.5), False),
        ("street-only label vs full address", same("3201 South 23rd Street", "3201 South 23rd Street, Abilene, Texas",
                                                   "property_address"), True),
        ("no commas = with commas", same("605 N Dixie Hwy,Hallandale Beach, FL 33009",
                                         "605 N Dixie Hwy Hallandale Beach FL 33009", "property_address"), True),
        ("other house number", same("1228 Rossmoor Pkwy, Walnut Creek, CA", "1229 Rossmoor Pkwy, Walnut Creek, CA",
                                    "property_address"), False),
        ("other city", same(valley_label, "1228 Rossmoor Parkway, Concord, CA 94595", "property_address"), False),
        ("other state", same(valley_label, "1228 Rossmoor Parkway, Walnut Creek, Oregon", "property_address"), False),
        ("Ship To name is not the Bill To", same("Taylor Riddel", "Taylor's Store"), False),
        ("the 5 outcome buckets", [outcome(None, None), outcome(None, 1), outcome(1, None), outcome(1, 1), outcome(1, 2)],
         ["true_null", "false_fill", "missed", "correct", "wrong"]),
    ]
    for name, got, want in cases:
        print(f"{'PASS' if got == want else 'FAIL'}  {name:<36} want {want} | got {got}")
    print(f"\n{sum(got == want for _, got, want in cases)}/{len(cases)} passed")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see rag/retrieve.py)
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif len(args) == 2 and args[0] == "--rescore":
        rescore(Path(args[1]))
    else:
        main()
