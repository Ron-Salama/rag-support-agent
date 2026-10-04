"""Part 1 - document extraction: document text -> LLM -> validated JSON, with retries,
business-rule checks and a human review queue.

    python -m ragagent.extract --selftest     # test the retry loop with a fake LLM (no API calls)
    python -m ragagent.extract inv_01         # extract one document
    python -m ragagent.extract                # extract every document in data/manifest.csv

The flow for one document:
    text ──> prompt ──> LLM ──> raw JSON text
                                   │
             Pydantic: right shape and types?   no ──> tell the model what was wrong, retry
                                   │ yes
             business rules + evidence check:   problems ──> retry, re-asking ONLY the failing fields
                                   │ none                    (passed fields are kept); still wrong -> human review
                                 status "ok"
"""
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import islice
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from ragagent import config
from ragagent.labels import manifest_rows
from ragagent.schemas import SCHEMAS, SCORED_FIELDS

SYSTEM = """You extract data from financial documents for a commercial real-estate lender.
Rules:
- Use ONLY the document text you are given. Never guess and never use outside knowledge.
- If a field's value is not in the document, return null for it.
- All documents are from the United States: read numeric dates like 9/10/2020 as MONTH/day/year.
- Dates: YYYY-MM-DD. Numbers: plain numbers exactly as printed (no $ or commas; "(1,234)" means -1234).
- For every field you fill in, add one evidence item: a short verbatim quote from the document that contains the value."""


@dataclass
class Result:
    status: str                      # "ok" | "needs_review" | "failed"
    data: dict | None                # the validated fields (None if we never got valid JSON)
    problems: list[str] = field(default_factory=list)
    attempts: int = 0                # how many times the model was really asked (1 to max_attempts)
    last_raw: str = ""               # last raw model answer - kept for debugging / the reviewer


class Problem(NamedTuple):
    """One thing wrong with an answer, plus the field(s) a retry must re-read to fix it."""
    fields: tuple[str, ...]
    message: str


def build_prompt(doc_type: str, text: str, feedback: str | None = None, to_fix: set[str] | None = None) -> str:
    prompt = f"Document type: {doc_type}\n\n<document>\n{text}\n</document>\n\nExtract the fields defined by the JSON schema."
    if feedback:
        prompt += f"\n\nYOUR PREVIOUS ANSWER WAS REJECTED. Fix these problems and answer again:\n{feedback}"
    if to_fix:
        prompt += (f"\n\nFix ONLY these fields: {', '.join(sorted(to_fix))}. Every other field already passed all "
                   "checks and is final: we keep your previous value for it, whatever you answer now.")
    return prompt


# ----------------------------------------------------------------- is the value really printed in the document?
QUOTE_WINDOW = 300                    # characters: how far apart a quote's words may be (see _quote_in_text)
NOT_VALUE_CHECKED = {"period_months"}  # printed as words ("Three Months Ended"); rule 2 keeps it to 3/6/9/12
MONTHS = "january february march april may june july august september october november december".split()
SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
# One printed number: optional '(' or '-', optional '$', the digits (1,234.56), optional '%' and ')', and an
# optional scale word after it ("1.2 million"). It never starts inside another number, so "0.5" is not a 5.
_NUMBER = re.compile(r"(?P<open>\()?\s*(?P<minus>-)?\$?\s*(?<![\d.])(?P<digits>\d(?:[\d,]*\d)?(?:\.\d+)?)"
                     r"\s*%?\s*(?P<close>\))?(?:\s*(?P<scale>thousand|million|billion)\b)?", re.IGNORECASE)


def _squash(s: str) -> str:
    return re.sub(r"[\s$,|]+", "", s).casefold()


def _words(s: str) -> list[tuple[str, int, bool]]:
    """Lower-case words and numbers: (word, position, is it the first word on its line?).
    'Total revenues $ 328,246' -> [('total', 0, True), ('revenues', 6, False), ('328246', 17, False)].
    Commas inside a number go first, so a number is ONE word and can't be stitched together from pieces."""
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s.casefold())
    out, end = [], 0
    for m in re.finditer(r"\d+(?:\.\d+)?|[^\W\d_]+", s):
        out.append((m.group(), m.start(), not out or "\n" in s[end:m.start()]))
        end = m.end()
    return out


def _quote_in_text(quote: str, text: str) -> bool:
    """Is this evidence quote really in the document?

    1. Exact: the quote is in the text once spaces, '$', ',' and '|' are ignored. Most quotes pass here.
    2. Header-tolerant: the quote's words follow each other as in the text, except that the quote may
       JUMP to the first word of a later line (within QUOTE_WINDOW characters). Tables are flattened to
       text, so a column header "Three Months Ended" / "June 30," / "2026" sits on three lines with other
       columns in between, and the model quotes it as one line. Still rejected: invented or reordered
       words, and a number taken from another column of the same row ("Total revenues 282,506" when the
       row reads "Total revenues | 328,246 | 282,506").
    Known limit: it checks words and lines, not table columns, so a quote can still pass when another
    column's word happens to start a line.
    """
    if _squash(quote) in _squash(text):
        return True
    want = [w for w, _, _ in _words(quote)]
    doc = _words(text)
    for i, (word, start, _) in enumerate(doc):
        if not want or word != want[0]:
            continue
        found, last = 1, i
        for j, (word2, pos, line_start) in enumerate(islice(doc, i + 1, None), i + 1):
            if found == len(want) or pos - start > QUOTE_WINDOW:
                break
            if word2 == want[found] and (j == last + 1 or line_start):   # the next word, or a jump to a new line
                found, last = found + 1, j
        if found == len(want):
            return True
    return False


def _numbers_in(quote: str) -> list[float]:
    """Every number printed in a quote: '$ (25,202)' -> -25202 (brackets = negative), '7.5%' -> 7.5,
    '$1.2 million' -> 1.2 and 1200000. '0.5' is read as ONE number, so a 5 never matches it."""
    found = []
    for m in _NUMBER.finditer(quote):
        n = float(m["digits"].replace(",", ""))
        if m["minus"] or (m["open"] and m["close"]):
            n = -n
        found.append(n)
        if m["scale"]:
            found.append(n * SCALES[m["scale"].casefold()])
    return found


def _number_in_quote(value: float, quote: str, signed: bool) -> bool:
    """Is `value` one of the numbers printed in the quote? The sign only counts when `signed` (financial
    statements, where '(25,202)' means -25202). Elsewhere brackets are not a minus sign: a loan note
    writes 'ONE HUNDRED MILLION DOLLARS ($100,000,000.00)', and a discount printed '$50' is stored as -50."""
    for n in _numbers_in(quote):
        a, b = (n, value) if signed else (abs(n), abs(value))
        if abs(a - b) <= 1e-6 * max(1.0, abs(b)):
            return True
    return False


def _dates_in(quote: str) -> set[date]:
    """Every date printed in a quote, in the common US forms: 'June 30, 2026', 'Jun. 30 2026', '30 June 2026',
    '1st day of April, 2020', '6/30/2026', '6/30/26', '2026-06-30'. Table cells are joined first, so the
    header 'June 30, | 2026' reads 'June 30, 2026'."""
    q = re.sub(r"\s+,", ",", " ".join(quote.replace("|", " ").split())).casefold()
    parts = [(y, m, d) for m, d, y in re.findall(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})\b", q)]  # month first (R2)
    parts += re.findall(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", q)
    parts += [(y, m, d) for m, d, y in re.findall(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", q)]
    parts += [(y, m, d) for d, m, y in
              re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([a-z]{3,9})\.?,?\s+(\d{4})\b", q)]
    dates = set()
    for y, m, d in parts:
        month = int(m) if m.isdigit() else next((i for i, name in enumerate(MONTHS, 1) if name.startswith(m)), 0)
        year = int(y) + (2000 if int(y) < 70 else 1900) if len(y) == 2 else int(y)   # '03 -> 2003, like strptime
        try:
            dates.add(date(year, month, int(d)))
        except ValueError:   # not a real date: 'ended 30 2026', month 13, ...
            pass
    return dates


def check_rules(doc_type: str, obj: BaseModel, text: str) -> list[Problem]:
    """Checks Pydantic can't do: is every value backed by a real quote that CONTAINS it? do the numbers add up?
    Each problem names the field(s) a retry must re-read."""
    problems = []

    # 1a. Grounding: every quote must really be in the document (catches invented values) ...
    quotes = defaultdict(list)   # field -> its quotes that ARE in the document
    for ev in obj.evidence:
        if _quote_in_text(ev.quote, text):
            quotes[ev.field].append(ev.quote)
        else:
            problems.append(Problem((ev.field,), f"evidence quote for '{ev.field}' is not in the document: {ev.quote!r}"))
    quoted = {ev.field for ev in obj.evidence}
    for name in SCORED_FIELDS[doc_type]:
        if getattr(obj, name, None) is not None and name not in quoted and name not in ("units", "interest_rate_type", "property_type"):
            problems.append(Problem((name,), f"field '{name}' has a value but no evidence quote"))

    # 1b. ... and the quote must contain the value itself. Catches CALCULATED values: tax = 311.25 "backed"
    #     by the quote 'Sales Tax 3%' (a real quote, but 311.25 is not printed anywhere in it).
    signed = doc_type == "financial_statement"   # there '(25,202)' means -25202, see _number_in_quote
    for name, found in quotes.items():
        value = getattr(obj, name, None)
        if isinstance(value, date):
            ok = any(value in _dates_in(q) for q in found)
        elif isinstance(value, (int, float)) and name not in NOT_VALUE_CHECKED:
            ok = any(_number_in_quote(value, q, signed) for q in found)
        else:
            continue   # text and category fields: the quote being in the document is the check
        if not ok:
            problems.append(Problem((name,), f"'{name}' = {value} is not printed in its evidence quote {found[0]!r}: "
                                             "copy the value exactly as printed, never calculate it (null if not printed)"))

    # 2. Business rules per document type (a broken rule re-opens every field it involves)
    if doc_type == "invoice":
        if obj.subtotal is not None and obj.total is not None:
            expected = obj.subtotal + (obj.tax or 0) + (obj.other_charges or 0)
            if abs(expected - obj.total) > 0.01 * abs(obj.total) + 0.01:
                problems.append(Problem(("subtotal", "tax", "other_charges", "total"),
                                        f"subtotal + tax + other_charges = {expected:,.2f} but total = {obj.total:,.2f} "
                                        "(re-read these amounts; an amount that is not printed stays null, never calculate it)"))
        if obj.invoice_date and obj.due_date and obj.due_date < obj.invoice_date:
            problems.append(Problem(("invoice_date", "due_date"), "due_date is before invoice_date"))
    elif doc_type == "financial_statement":
        if obj.period_months not in (None, 3, 6, 9, 12):
            problems.append(Problem(("period_months",), f"period_months={obj.period_months} is not 3, 6, 9 or 12"))
    elif doc_type == "loan_agreement":
        if obj.interest_rate_type == "floating" and obj.spread_percent is None:
            problems.append(Problem(("interest_rate_type", "spread_percent"), "floating rate but no spread_percent"))
        if obj.interest_rate_type == "fixed" and obj.fixed_rate_percent is None:
            problems.append(Problem(("interest_rate_type", "fixed_rate_percent"), "fixed rate but no fixed_rate_percent"))
        if obj.agreement_date and obj.maturity_date and obj.maturity_date <= obj.agreement_date:
            problems.append(Problem(("agreement_date", "maturity_date"), "maturity_date is not after agreement_date"))
    elif doc_type == "appraisal":
        if obj.cap_rate_percent is not None and not 2 <= obj.cap_rate_percent <= 15:
            problems.append(Problem(("cap_rate_percent",),
                                    f"cap_rate_percent={obj.cap_rate_percent} is outside a plausible 2-15% range"))
    return problems


def keep_passed_fields(kept: BaseModel, new: BaseModel, to_fix: set[str]) -> BaseModel:
    """The kept answer, with ONLY the fields in `to_fix` (and their evidence quotes) taken from the new answer.
    Fields that already passed every check are final: a retry may not change them."""
    data, fresh = kept.model_dump(), new.model_dump()
    for name in to_fix & data.keys():
        data[name] = fresh[name]
    data["evidence"] = ([ev for ev in data["evidence"] if ev["field"] not in to_fix]
                        + [ev for ev in fresh["evidence"] if ev["field"] in to_fix])
    return type(kept).model_validate(data)


# ===================================================================================
# Core loop - kept small and framework-free on purpose; offline self-test:
#   python -m ragagent.extract --selftest
# ===================================================================================
def extract_with_retries(llm, doc_type: str, text: str, max_attempts: int = 3) -> Result:
    """Extract `doc_type` fields from `text`, validating and retrying up to `max_attempts` LLM calls.

    Returns Result "ok" (schema valid, every check_rules check passed), "needs_review" (a valid answer
    with problems left after the last attempt) or "failed" (never a valid answer); `attempts` = calls
    really made. A retry re-asks only the fields the problems name, and code keeps every field that
    already passed (keep_passed_fields). LLMError is not caught: a bad key or a used-up quota stops the
    run. DECISIONS.md C3, C4, C22-C24.
    """
    schema = SCHEMAS[doc_type]
    kept, to_fix, problems, feedback, raw = None, set(), [], None, ""
    for attempt in range(1, max_attempts + 1):
        raw = llm.generate_json(SYSTEM, build_prompt(doc_type, text, feedback, to_fix), schema)

        # Check 1: is it valid JSON with the right fields and types?
        try:
            obj = schema.model_validate_json(raw)
        except ValidationError as e:
            feedback = f"Your answer was:\n{raw[:2000]}\n\nIt failed validation:\n{e}"
            continue  # ask again, telling the model exactly what was wrong
        if kept is not None:  # a retry: fields that passed are final, take only the fixes
            obj = keep_passed_fields(kept, obj, to_fix)

        # Check 2: is every value backed by a real quote that contains it, and do the numbers add up?
        problems = check_rules(doc_type, obj, text)
        if not problems:
            return Result("ok", obj.model_dump(mode="json"), [], attempt, raw)
        kept, to_fix = obj, {name for p in problems for name in p.fields}
        feedback = "\n".join(f"- {p.message}" for p in problems)

    if kept is not None:  # out of attempts with problems left (even if the last answer wasn't valid JSON)
        return Result("needs_review", kept.model_dump(mode="json"), [p.message for p in problems], max_attempts, raw)
    return Result("failed", None, [feedback or "no valid answer"], max_attempts, raw)


# ----------------------------------------------------------------- running it
def save(doc_id: str, result: Result, model: str):
    config.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    payload = {"doc_id": doc_id, "model": model, **asdict(result)}
    (config.PREDICTIONS / f"{doc_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    review_file = config.REVIEW_QUEUE / f"{doc_id}.json"
    if result.status != "ok":  # human-in-the-loop: anything not clean goes to a person
        config.REVIEW_QUEUE.mkdir(parents=True, exist_ok=True)
        review_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif review_file.exists():  # clean now (e.g. after a fix): take it off the reviewer's pile
        review_file.unlink()


def run(doc_ids: list[str], seconds_between_calls: float = 0):
    from ragagent.llm import get_llm

    llm = get_llm()
    rows = [r for r in manifest_rows() if not doc_ids or r["doc_id"] in doc_ids]
    for row in rows:
        text = (config.TEXT / f"{row['doc_id']}.txt").read_text(encoding="utf-8")
        t0 = time.time()
        result = extract_with_retries(llm, row["doc_type"], text)
        save(row["doc_id"], result, llm.model)
        print(f"{row['doc_id']:<24} {result.status:<13} attempts={result.attempts}  {time.time() - t0:5.1f}s"
              + (f"  problems: {result.problems}" if result.problems else ""))
        time.sleep(seconds_between_calls)


# ----------------------------------------------------------------- self-test with a fake LLM
class FakeLLM:
    """Plays back scripted answers, so you can test the retry loop without any API calls."""
    model = "fake"

    def __init__(self, answers: list[str]):
        self.answers, self.calls = list(answers), 0

    def generate_json(self, system, user, schema):
        self.calls += 1
        return self.answers.pop(0)


def selftest():
    text = "=== Page 1 ===\nACME Supply Co. Invoice No. 1001 Date: 2024-03-05 Sales Tax 5% Total due: $150.00"
    good = json.dumps({"vendor_name": "ACME Supply Co.", "invoice_number": "1001", "invoice_date": "2024-03-05",
                       "total": 150.0, "evidence": [
                           {"field": "vendor_name", "quote": "ACME Supply Co."},
                           {"field": "invoice_number", "quote": "Invoice No. 1001"},
                           {"field": "invoice_date", "quote": "Date: 2024-03-05"},
                           {"field": "total", "quote": "Total due: $150.00"}]})
    bad_date = good.replace("2024-03-05", "March 5th 2024", 1)
    invented = good.replace('"quote": "Total due: $150.00"', '"quote": "Amount: $999"')
    vendor_changed = good.replace('"vendor_name": "ACME Supply Co."', '"vendor_name": "Evil Corp"')
    # tax 7.14 = 5% worked out by the model: the quote 'Sales Tax 5%' is real, but 7.14 is not printed in it
    calculated = good.replace('"total": 150.0,', '"tax": 7.14, "total": 150.0,').replace(
        '"evidence": [', '"evidence": [{"field": "tax", "quote": "Sales Tax 5%"}, ')
    cases = [   # (name, scripted answers, want status, want attempts, extra check on the Result or None)
        ("valid on 1st try", [good], "ok", 1, None),
        ("not JSON, then valid", ["Sure! Here is the JSON...", good], "ok", 2, None),
        ("bad date twice, then valid", [bad_date, bad_date, good], "ok", 3, None),
        ("never valid", ["oops", "oops", "oops"], "failed", 3, None),
        ("invented quote every time", [invented, invented, invented], "needs_review", 3, None),
        # the retry fixes 'total' but also changes the vendor, which had passed: the vendor must stay
        ("retry can't change a passed field", [invented, vendor_changed], "ok", 2,
         lambda r: r.data["vendor_name"] == "ACME Supply Co." and r.data["total"] == 150.0),
        ("calculated tax caught, then null", [calculated, good], "ok", 2, lambda r: r.data["tax"] is None),
        ("calculated tax every time", [calculated] * 3, "needs_review", 3,
         lambda r: any("'tax' = 7.14 is not printed" in p for p in r.problems)),
    ]
    passed = 0
    for name, answers, want_status, want_attempts, check in cases:
        fake = FakeLLM(answers)
        try:
            r = extract_with_retries(fake, "invoice", text, max_attempts=3)
            ok = (r.status == want_status and r.attempts == want_attempts == fake.calls
                  and (check is None or check(r)))
            got = f"status={r.status} attempts={r.attempts} llm_calls={fake.calls}"
        except Exception as e:  # noqa: BLE001 - show any crash as a failed case
            ok, got = False, f"crashed: {type(e).__name__}: {e}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<34} want status={want_status} attempts={want_attempts} | got {got}")

    # The proof helpers on their own: these must not raise false alarms on real documents.
    header = "Three Months Ended | Six Months Ended\nJune 30, | June 30,\n2026 | 2025 | 2026 | 2025\nTotal revenues | 121,319"
    checks = [
        ("5 is not in '0.5%'", not _number_in_quote(5, "Rate 0.5%", signed=False)),
        ("'(25,202)' is -25202 in a statement", _number_in_quote(-25202, "Net loss | $ | (25,202)", signed=True)
         and not _number_in_quote(25202, "Net loss | $ | (25,202)", signed=True)),
        ("'($100,000,000.00)' backs 100000000", _number_in_quote(100_000_000, "Dollars ($100,000,000.00)", signed=False)),
        ("'$1.2 million' backs 1200000", _number_in_quote(1_200_000, "valued at $1.2 million", signed=False)),
        ("date forms: 'June 30, 2026' '6/30/26'", {date(2026, 6, 30)} == _dates_in("June 30, 2026") == _dates_in("on 6/30/26")),
        ("date split over table cells", date(2026, 6, 30) in _dates_in("Three Months Ended | June 30, | 2026")),
        ("stitched header quote is found", _quote_in_text("Three Months Ended June 30, 2026", header)),
        ("invented / reordered quote is not", not _quote_in_text("Nine Months Ended June 30, 2026", header)
         and not _quote_in_text("2026 June 30 Three Months Ended", header)),
        ("number from the next column is not", not _quote_in_text("Total revenues 90,662", header + " | 90,662")),
        ("'121' is not a piece of '121,319'", not _quote_in_text("Three Months Ended Total revenues 121", header)),
    ]
    for name, ok in checks:
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(cases) + len(checks)} passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see rag/retrieve.py)
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    else:
        run(args)
