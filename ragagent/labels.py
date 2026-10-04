"""Hand labels = the answer key the extraction is scored against.

    python -m ragagent.labels init     # make an empty data/labels/<doc_id>.json for each document
    python -m ragagent.labels check    # validate every filled-in label with the same Pydantic schema

How to label (do it BEFORE you ever look at what the model extracted - otherwise you
copy the model's answer and the score means nothing):
  - open the original document (data/raw/...), find each field, type the value.
  - numbers: plain number exactly as printed, no $ or commas: 1234567.89 ; negatives as -123
  - dates: "YYYY-MM-DD"
  - not in the document -> leave it null
  - unsure -> write your doubt in "_notes" (ambiguous fields are an eval finding too)
"""
import csv
import json
import sys

from pydantic import ValidationError

from ragagent.config import LABELS, MANIFEST
from ragagent.schemas import SCHEMAS, SCORED_FIELDS


def manifest_rows() -> list[dict]:
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:  # -sig: tolerate a BOM (Excel/PowerShell add one)
        return list(csv.DictReader(f))


def init():
    LABELS.mkdir(parents=True, exist_ok=True)
    for row in manifest_rows():
        path = LABELS / f"{row['doc_id']}.json"
        if path.exists():
            print(f"keep     {path.name} (already exists)")
            continue
        template = {"_doc_type": row["doc_type"], "_file": row["file"], "_pages": row.get("pages", ""), "_notes": ""}
        template.update({field: None for field in SCORED_FIELDS[row["doc_type"]]})
        path.write_text(json.dumps(template, indent=2), encoding="utf-8")
        print(f"created  {path.name}")


def check() -> bool:
    ok = True
    for row in manifest_rows():
        path = LABELS / f"{row['doc_id']}.json"
        if not path.exists():
            print(f"MISSING  {path.name}")
            ok = False
            continue
        label = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}
        filled = sum(v is not None for v in label.values())
        if filled == 0:
            print(f"todo     {path.name:<32} (not labeled yet)")
            ok = False
            continue
        try:
            SCHEMAS[row["doc_type"]].model_validate(label)
            print(f"ok       {path.name:<32} {filled}/{len(label)} fields filled")
        except ValidationError as e:
            ok = False
            print(f"INVALID  {path.name}")
            for err in e.errors():
                print(f"         - {'.'.join(map(str, err['loc']))}: {err['msg']}")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "init":
        init()
    else:
        sys.exit(0 if check() else 1)
