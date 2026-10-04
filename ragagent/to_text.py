"""Step 1 of extraction: get plain text out of each document. No AI here, just parsing.

    python -m ragagent.to_text            # convert every document in data/manifest.csv
    python -m ragagent.to_text --full     # same, but EVERY page -> data/fulltext/ (for RAG, Part 2)

Output: data/text/<doc_id>.txt with a "=== Page N ===" line before each page, so later
steps (extraction evidence, RAG citations) can point at a page number.
PDF pages are real pages. An HTML filing has no pages, so a "page" there is the text between
two printed page breaks that EDGAR marks in the HTML (N = the N-th such piece, which may not
match the number printed in the footer); a filing without those marks is one long page 1.

Why two outputs: extraction (Part 1) only reads the pages listed in the manifest "pages"
column (the pages that hold the fields - short prompt, fewer tokens). RAG (Part 2) must be
able to answer questions about ANY page, so it indexes the full documents.

A scanned PDF (a picture of text) has no text layer and would come out empty here;
that needs OCR or a vision model, which is out of scope, so we only use digital PDFs.
"""
import csv
import re
import sys

import pdfplumber
from bs4 import BeautifulSoup, Comment

from ragagent.config import FULLTEXT, MANIFEST, RAW, TEXT


def parse_pages(spec: str, total: int) -> list[int]:
    """'1-5' or '2,7-9' or '' (all pages) -> 1-based page numbers."""
    if not spec.strip():
        return list(range(1, total + 1))
    pages = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        pages.extend(range(int(a), int(b or a) + 1))
    return [p for p in pages if 1 <= p <= total]


def pdf_pages(path, spec: str) -> list[tuple[int, str]]:
    with pdfplumber.open(path) as pdf:
        wanted = parse_pages(spec, len(pdf.pages))
        return [(n, pdf.pages[n - 1].extract_text() or "") for n in wanted]


BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "center"]


def html_pages(path, spec: str) -> list[tuple[int, str]]:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for bad in soup(["script", "style", "img"]):
        bad.decompose()
    # In HTML a newline inside text is just a space; real line breaks come from block tags.
    for s in soup.find_all(string=True):
        if not isinstance(s, Comment):
            s.replace_with(re.sub(r"\s+", " ", s))
    # Keep table structure readable: one row per line, cells separated by " | "
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        tr.replace_with(" | ".join(c for c in cells if c) + "\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    # EDGAR marks printed page breaks with a CSS page-break style; use them as "pages"
    marker = "@@PAGEBREAK@@"
    for tag in soup.find_all(style=re.compile(r"page-break-(before|after)", re.I)):
        tag.insert_before(f"\n{marker}\n")
    text = soup.get_text("")
    # \xa0 = non-breaking space, \u200b = zero-width space (both appear in EDGAR HTML)
    lines = (re.sub(r"[ \t\xa0\u200b]+", " ", line).strip() for line in text.split("\n"))
    text = "\n".join(line for line in lines if line)
    chunks = [c.strip() for c in text.split(marker) if c.strip()]
    wanted = parse_pages(spec, len(chunks))
    return [(n, chunks[n - 1]) for n in wanted]


def convert(row: dict, full: bool = False) -> str:
    path = RAW / row["file"]
    spec = "" if full else row.get("pages", "")  # "" means every page
    if path.suffix.lower() == ".pdf":
        pages = pdf_pages(path, spec)
    elif path.suffix.lower() in (".htm", ".html"):
        pages = html_pages(path, spec)
    else:
        pages = [(1, path.read_text(encoding="utf-8", errors="replace"))]
    return "\n".join(f"=== Page {n} ===\n{t.strip()}" for n, t in pages)


def main(full: bool = False):
    out_dir = FULLTEXT if full else TEXT
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:  # -sig: tolerate a BOM (Excel/PowerShell add one)
        for row in csv.DictReader(f):
            text = convert(row, full)
            (out_dir / f"{row['doc_id']}.txt").write_text(text, encoding="utf-8")
            words = len(text.split())
            pages = text.count("=== Page ")
            flag = "  <-- almost no text: scanned?" if words < 50 else ""
            print(f"{row['doc_id']:<28} {row['doc_type']:<20} {pages:>4} pages {words:>7} words{flag}")


if __name__ == "__main__":
    main(full="--full" in sys.argv[1:])
