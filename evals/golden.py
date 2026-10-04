"""The golden set: hand-checked questions with known answers, used by every Part 4 eval.

    python -m evals.golden                          # check the golden file (default one, see below)
    python -m evals.golden evals\\golden.jsonl       # check a specific file

One JSON object per line (format and how to write good questions: evals/GOLDEN_README.md):
    answerable: {"id", "type": "answerable", "question", "answer", "doc_id", "pages": [..], "quote"}
    trap:       {"id", "type": "trap", "question", "why_unanswerable"}
A line holding {"_header": "..."} is a comment and is skipped.

Which file is used: evals/golden.jsonl (an extended, hand-written set) if it exists, otherwise
evals/golden_draft.jsonl (the 24-question draft set; what it is and how it was checked: evals/GOLDEN_README.md).

Why check the file at all? A wrong answer key gives a wrong score that LOOKS precise. The
check catches the cheap mistakes: a typo in a doc_id, a page number that does not exist, a
"verbatim" quote that is not actually in the document (or not on the pages you listed).
"""
import json
import sys
from pathlib import Path

from ragagent.config import FULLTEXT, ROOT
from ragagent.labels import manifest_rows
from ragagent.rag.chunk import split_pages

OWN = ROOT / "evals" / "golden.jsonl"
DRAFT = ROOT / "evals" / "golden_draft.jsonl"
REQUIRED = {"answerable": ["id", "question", "answer", "doc_id", "pages", "quote"],
            "trap": ["id", "question", "why_unanswerable"]}


def default_path() -> Path:
    return OWN if OWN.exists() else DRAFT


def load(path: Path | None = None) -> list[dict]:
    """Golden file -> list of question dicts (header/comment lines and blank lines skipped)."""
    path = path or default_path()
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:   # json's own message only says "line 1": tell WHICH line of the file
                raise ValueError(f"{path.name} line {line_no} is not valid JSON: {e}") from None
            if "_header" not in obj:
                items.append(obj)
    return items


def load_checked() -> list[dict]:
    """load() the default golden file, but STOP if it has problems: every eval starts with this.

    Without it, a typo in a doc_id or a page number would not show up as an error - just as a
    "retrieval miss", and the score would quietly be wrong.
    """
    path = default_path()
    items = load(path)
    errors = [p for p in problems(items) if not p.startswith("note:")]
    if errors:
        sys.exit(f"{path.name} has {len(errors)} problem(s) - fix them first (python -m evals.golden):\n  "
                 + "\n  ".join(errors))
    return items


def squash(text: str) -> str:
    """Whitespace-insensitive form for "is this quote in that text": the PDF/HTML text has odd line breaks."""
    return " ".join(text.split())


def pages_of(doc_id: str) -> dict[int, str]:
    """{page number: page text} for one document, from data/fulltext (the same pages RAG cites)."""
    return dict(split_pages((FULLTEXT / f"{doc_id}.txt").read_text(encoding="utf-8")))


def problems(items: list[dict]) -> list[str]:
    """Everything wrong with a golden set (empty list = OK). Warnings start with 'note:'."""
    out, seen = [], set()
    doc_ids = {row["doc_id"] for row in manifest_rows()}
    for it in items:
        qid = it.get("id", "?")
        if qid in seen:
            out.append(f"{qid}: duplicate id")
        seen.add(qid)
        missing = [k for k in REQUIRED.get(it.get("type"), []) if not it.get(k)]
        if it.get("type") not in REQUIRED:
            out.append(f"{qid}: type must be 'answerable' or 'trap', got {it.get('type')!r}")
        elif missing:
            out.append(f"{qid}: missing {missing}")
        if it.get("type") != "answerable" or missing:
            continue
        if it["doc_id"] not in doc_ids:
            out.append(f"{qid}: unknown doc_id {it['doc_id']!r} (see data/manifest.csv)")
            continue
        if not isinstance(it["pages"], list) or not all(type(p) is int for p in it["pages"]):
            out.append(f"{qid}: pages must be a list of numbers like [2] or [8, 11], got {it['pages']!r}")
            continue
        pages = pages_of(it["doc_id"])
        bad = [p for p in it["pages"] if p not in pages]
        if bad:
            out.append(f"{qid}: page(s) {bad} not in {it['doc_id']} (it has pages 1-{max(pages)})")
        found_on = [p for p, text in pages.items() if squash(it["quote"]) in squash(text)]
        if not set(found_on) & set(it["pages"]):
            out.append(f"{qid}: quote not found on pages {it['pages']} (found on: {found_on or 'no page'})")
        elif set(found_on) - set(it["pages"]):
            out.append(f"note: {qid}: quote also on pages {sorted(set(found_on) - set(it['pages']))} - add them if they answer it")
    return out


def main(path: Path):
    items = load(path)
    kinds = [it.get("type") for it in items]
    print(f"{path}: {kinds.count('answerable')} answerable, {kinds.count('trap')} traps, "
          f"{len({it.get('doc_id') for it in items if it.get('doc_id')})} documents covered")
    found = problems(items)
    for p in found:
        print(f"  {p}")
    errors = [p for p in found if not p.startswith("note:")]
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return not errors


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    sys.exit(0 if main(Path(sys.argv[1]) if sys.argv[1:] else default_path()) else 1)
