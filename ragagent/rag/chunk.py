"""Chunking: cut each full document into small pieces that can each be found on their own.

    python -m ragagent.rag.chunk               # chunk every document, print counts + one example
    python -m ragagent.rag.chunk --selftest    # check the chunker's rules on a tiny fake document

Why chunk at all? An embedding (embed.py) squeezes a whole text into ONE vector. A vector
for a 190-page appraisal would be a blurry average of everything in it; a vector for ~800
characters is about one topic, so a question can land on exactly the right piece. And the
LLM later reads only the 5 best pieces, not 190 pages (cheaper, and less to get lost in).

Rules (see DECISIONS.md for the numbers):
  - A chunk NEVER crosses a page boundary, so every chunk cites exactly one page.
  - About CHUNK_SIZE characters; cut at a line break if possible, else at a sentence end,
    else between words (never mid-word unless a single "word" is longer than a chunk).
  - Consecutive chunks of a page share ~OVERLAP characters, so a fact that sits right on a
    cut is still whole in at least one chunk.
  - Chunks with almost no letters/digits (a lone page number, "| |") are skipped.

Each chunk is a plain dict (easy to save as JSON):
    {"chunk_id": "loan_note_jshighland:p2:c1", "doc_id": ..., "doc_type": ..., "page": 2,
     "source_url": ..., "text": ...}
"""
import re
import sys

from ragagent.config import FULLTEXT
from ragagent.labels import manifest_rows

CHUNK_SIZE = 800     # characters (~200 tokens; the embedding model reads up to 512 tokens)
OVERLAP = 150        # characters repeated from the end of the previous chunk
MIN_ALNUM = 30       # chunks with fewer letters+digits than this are noise, not content
SEPARATORS = ["\n", ". ", " "]   # preferred cut points, best first (our text has no blank lines)

PAGE_MARK = re.compile(r"^=== Page (\d+) ===$", re.M)


def split_pages(text: str) -> list[tuple[int, str]]:
    """'=== Page 3 ===\\n...' text (made by to_text.py) -> [(3, '...'), ...]."""
    parts = PAGE_MARK.split(text)   # ['', '1', 'page one text', '2', 'page two text', ...]
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def split_text(text: str, max_len: int, seps: list[str] = SEPARATORS) -> list[str]:
    """Cut text into pieces of at most max_len chars, using the best separator that works.

    Try the first separator; any piece still too long is cut again with the next one.
    The separator stays at the end of its piece, so joining the pieces gives the text back.
    """
    if len(text) <= max_len:
        return [text]
    if not seps:  # nothing left to cut on: hard cut
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]
    parts = text.split(seps[0])
    parts = [p + seps[0] for p in parts[:-1]] + [parts[-1]]
    pieces = []
    for part in parts:
        pieces.extend(split_text(part, max_len, seps[1:]))
    return [p for p in pieces if p]


def tail(text: str, n: int) -> str:
    """The last ~n characters of text, starting at a word boundary (the overlap)."""
    if n <= 0:   # careful: text[-0:] is the WHOLE text (same as text[0:]), not an empty string
        return ""
    t = text[-n:]
    cut = t.find(" ")
    return t[cut + 1:] if cut != -1 else t


def chunk_page(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """One page -> chunks of at most `size` chars that overlap by up to `overlap` chars.

    Pieces are at most (size - overlap), so "overlap + one piece" always fits in a chunk.
    Greedy packing: keep adding pieces; when the next one does not fit, close the chunk
    and start the next chunk with the tail of the closed one.
    """
    chunks, current = [], ""
    for piece in split_text(text, size - overlap):
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            current = tail(current, overlap)
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if sum(ch.isalnum() for ch in c) >= MIN_ALNUM]


def chunk_document(row: dict, text: str) -> list[dict]:
    """All chunks of one document, each with its metadata (row = its line in manifest.csv)."""
    out = []
    for page, page_text in split_pages(text):
        for i, chunk_text in enumerate(chunk_page(page_text), start=1):
            out.append({"chunk_id": f"{row['doc_id']}:p{page}:c{i}", "doc_id": row["doc_id"],
                        "doc_type": row["doc_type"], "page": page, "source_url": row["source_url"],
                        "text": chunk_text})
    return out


def chunk_all() -> list[dict]:
    """Chunk every document in data/manifest.csv (reads data/fulltext/, made by to_text --full)."""
    chunks = []
    for row in manifest_rows():
        path = FULLTEXT / f"{row['doc_id']}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python -m ragagent.to_text --full")
        chunks.extend(chunk_document(row, path.read_text(encoding="utf-8")))
    return chunks


# ----------------------------------------------------------------- self-test (no files needed)
def selftest():
    sentence = "The borrower shall pay monthly installments of principal and interest. "
    doc = "=== Page 1 ===\n" + sentence * 30 + "\n=== Page 2 ===\nShort page two text about the guarantor.\n=== Page 3 ===\n7"
    row = {"doc_id": "fake", "doc_type": "loan_agreement", "source_url": "http://example"}
    chunks = chunk_document(row, doc)
    p1 = [c for c in chunks if c["page"] == 1]
    cases = [
        ("pages split correctly", [p for p, _ in split_pages(doc)] == [1, 2, 3]),
        ("no chunk longer than CHUNK_SIZE", all(len(c["text"]) <= CHUNK_SIZE for c in chunks)),
        ("long page -> several chunks", len(p1) >= 3),
        ("chunks never mix pages", all("guarantor" not in c["text"] for c in p1)),
        ("neighbours overlap", all(a["text"][-40:] in b["text"] for a, b in zip(p1, p1[1:]))),
        ("cut at sentence ends", all(c["text"].endswith(".") for c in p1)),
        ("near-empty page skipped", all(c["page"] != 3 for c in chunks)),
        ("chunk_id format", chunks[0]["chunk_id"] == "fake:p1:c1" and chunks[-1]["chunk_id"] == "fake:p2:c1"),
        ("one huge word is hard-cut", all(len(c) <= 100 for c in chunk_page("x" * 250, size=100, overlap=20))),
        ("overlap 0 -> no repeats, size kept", all(len(c) <= 200 for c in chunk_page(sentence * 30, size=200, overlap=0))
         and sum(map(len, chunk_page(sentence * 30, size=200, overlap=0))) < len(sentence * 30)),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in cases)}/{len(cases)} passed")


def main():
    chunks = chunk_all()
    for row in manifest_rows():
        mine = [c for c in chunks if c["doc_id"] == row["doc_id"]]
        print(f"{row['doc_id']:<24} {len(mine):>5} chunks")
    lengths = sorted(len(c["text"]) for c in chunks)
    print(f"\nTOTAL {len(chunks)} chunks, length min/median/max = "
          f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]} chars")
    print(f"\nExample {chunks[0]['chunk_id']}:\n{chunks[0]['text']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main()
