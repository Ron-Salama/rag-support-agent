"""Retrieval eval: does search() put the right page in the top k? No LLM calls, so it is free.

    python -m evals.retrieval_eval               # the real index (Chroma): hit@1/3/5/10 + MRR
    python -m evals.retrieval_eval --exact       # same questions, exact numpy search (chroma/vectors.npy)
    python -m evals.retrieval_eval --ablation    # chunk size 400/800/1600 x overlap 0/~20%: which works best?
    python -m evals.retrieval_eval --selftest    # check the scoring code with made-up hits

Why measure retrieval on its own? The LLM only ever reads the k chunks we retrieve. If the
right page is not among them, no prompt can fix the answer. Scoring retrieval separately tells
you WHICH half is broken (finding vs answering), and it costs no LLM quota at all.

Definitions (see DECISIONS.md C14):
  hit      a retrieved chunk from the expected doc_id AND on one of the expected pages
  rank     position of the FIRST hit among the top 10 (1 = best); no hit in the top 10 = a miss
  hit@k    share of questions with rank <= k  ("was the right page in the k chunks the LLM sees?")
  MRR      mean of 1/rank (a miss counts 0). 1.0 = the right page always came first.
  quote@k  stricter: the hit chunk must also CONTAIN the golden quote (whitespace ignored).
           A page-level hit is easier for big chunks (one 1600-char chunk covers most of a page),
           so quote@k checks that the exact passage the LLM needs is really inside the chunk.
Only answerable questions are scored: a trap has no right page.

The ablation re-chunks and re-embeds the whole corpus for each setting and searches it IN
MEMORY with exact search (retrieve.search_bruteforce). chroma/ is never touched, and exact
search means the comparison is not blurred by HNSW's occasional misses (DECISIONS.md C10).
Vectors are cached in .cache/ablation/, so a re-run (e.g. after editing the golden set) is fast.
"""
import hashlib
import json
import sys
import time
from datetime import datetime

import numpy as np

from evals import golden
from ragagent import config
from ragagent.labels import manifest_rows
from ragagent.rag import chunk, embed, retrieve

KS = (1, 3, 5, 10)            # hit@k values reported; 3 / 5 / 10 are the k options in DECISIONS.md C9
MAX_K = max(KS)               # how deep we look for the first hit (MRR@10)
ABLATION = [(400, 0), (400, 80), (800, 0), (800, 150), (1600, 0), (1600, 320)]   # (size, overlap) in chars
ABLATION_CACHE = config.ROOT / ".cache" / "ablation"


def first_rank(hits: list, item: dict, need_quote: bool = False) -> int | None:
    """1-based position of the first hit (right doc AND page, + contains the quote if need_quote)."""
    for rank, h in enumerate(hits, start=1):
        if h.doc_id == item["doc_id"] and h.page in item["pages"]:
            if not need_quote or golden.squash(item["quote"]) in golden.squash(h.text):
                return rank
    return None


def metrics(ranks: list[int | None], quote_ranks: list[int | None]) -> dict:
    """One rank per question (None = miss) -> hit@k for every k, MRR and quote@5, as fractions of all questions.

    sum() over True/False counts the Trues: sum(r <= 3 for r in [1, 5, 2]) == 2.
    """
    n = len(ranks)
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    out["quote@5"] = sum(r is not None and r <= 5 for r in quote_ranks) / n
    return out


def evaluate(search_fn, items: list[dict]) -> tuple[dict, list[dict]]:
    """Run search_fn(question, k) for every answerable item -> (summary metrics, one row per question)."""
    rows = []
    for it in items:
        hits = search_fn(it["question"], MAX_K)
        rows.append({"id": it["id"], "doc_id": it["doc_id"], "pages": it["pages"],
                     "rank": first_rank(hits, it), "quote_rank": first_rank(hits, it, need_quote=True),
                     "top": [{"chunk_id": h.chunk_id, "score": round(h.score, 4)} for h in hits[:3]]})
    return metrics([r["rank"] for r in rows], [r["quote_rank"] for r in rows]), rows


def print_rows(rows: list[dict]):
    print(f"\n{'id':<5} {'rank':>4} {'quote':>5}  {'expected':<30} top-1 retrieved (score)")
    for r in rows:
        expected = f"{r['doc_id']} p{','.join(map(str, r['pages']))}"
        top = r["top"][0] if r["top"] else {"chunk_id": "-", "score": 0}
        print(f"{r['id']:<5} {r['rank'] or '-':>4} {r['quote_rank'] or '-':>5}  {expected:<30} "
              f"{top['chunk_id']} ({top['score']:.3f})")


def print_metrics(m: dict, n: int):
    print(f"\n{n} answerable questions:  " + "   ".join(f"{k} {v:.0%}" for k, v in m.items() if k != "mrr")
          + f"   MRR@{MAX_K} {m['mrr']:.3f}")


def save(name: str, payload: dict):
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


def command() -> str:
    return "python -m evals.retrieval_eval " + " ".join(sys.argv[1:])


def answerable() -> list[dict]:
    """The answerable golden questions (a trap has no right page, so it is not scored here)."""
    items = [it for it in golden.load_checked() if it["type"] == "answerable"]
    if not items:
        sys.exit(f"{golden.default_path().name} has no answerable questions yet - nothing to score.")
    return items


def main(exact: bool):
    items = answerable()
    search_fn = retrieve.search_numpy if exact else retrieve.search
    m, rows = evaluate(search_fn, items)
    print_rows(rows)
    print_metrics(m, len(items))
    save("retrieval", {"date": f"{datetime.now():%Y-%m-%d %H:%M}", "command": command(),
                       "golden": golden.default_path().name, "index": "exact numpy" if exact else "chroma",
                       "embedding_model": embed.MODEL_NAME, "chunk_size": chunk.CHUNK_SIZE,
                       "overlap": chunk.OVERLAP, "n_questions": len(items), "metrics": m, "questions": rows})


# ----------------------------------------------------------------- ablation: other chunk sizes
def chunk_corpus(size: int, overlap: int) -> list[dict]:
    """Same as chunk.chunk_all(), but with this size/overlap instead of the CHUNK_SIZE/OVERLAP constants."""
    chunks = []
    for row in manifest_rows():
        text = (config.FULLTEXT / f"{row['doc_id']}.txt").read_text(encoding="utf-8")
        for page, page_text in chunk.split_pages(text):
            for i, t in enumerate(chunk.chunk_page(page_text, size, overlap), start=1):
                chunks.append({"chunk_id": f"{row['doc_id']}:p{page}:c{i}", "doc_id": row["doc_id"],
                               "doc_type": row["doc_type"], "page": page, "source_url": row["source_url"], "text": t})
    return chunks


def embed_cached(texts: list[str]) -> np.ndarray:
    """Embed the chunks, or load the vectors from .cache/ablation/ if these exact texts were embedded before."""
    key = hashlib.sha256((embed.MODEL_NAME + "\n".join(texts)).encode()).hexdigest()[:16]
    path = ABLATION_CACHE / f"{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = embed.embed_passages(texts)
    ABLATION_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def ablation():
    items = answerable()
    query_vecs = {it["question"]: embed.embed_query(it["question"]) for it in items}   # same for every setting
    model = embed.get_model()
    results = []
    for size, overlap in ABLATION:
        t0 = time.time()
        chunks = chunk_corpus(size, overlap)
        texts = [c["text"] for c in chunks]
        vecs = embed_cached(texts)
        # The model reads at most max_seq_length tokens (512); anything after that is silently ignored.
        # (Counting them here makes transformers warn "longer than ... 512": harmless, it is what we measure.)
        too_long = sum(len(ids) > model.max_seq_length for ids in model.tokenizer(texts)["input_ids"])
        m, rows = evaluate(lambda q, k: retrieve.search_bruteforce(query_vecs[q], vecs, chunks, k), items)
        results.append({"size": size, "overlap": overlap, "n_chunks": len(chunks), "cut_off_chunks": too_long,
                        "seconds": round(time.time() - t0, 1), "metrics": m,
                        "ranks": {r["id"]: r["rank"] for r in rows}})
        print(f"  size {size:>4} overlap {overlap:>3}: {len(chunks):>5} chunks, done in {time.time() - t0:5.1f}s")

    print(f"\nABLATION on {len(items)} answerable questions ({golden.default_path().name}), exact search:")
    print(f"{'size':>5} {'overlap':>7} {'chunks':>6} {'cut':>4}  " + " ".join(f"{k:>7}" for k in results[0]["metrics"]))
    for r in results:
        print(f"{r['size']:>5} {r['overlap']:>7} {r['n_chunks']:>6} {r['cut_off_chunks']:>4}  "
              + " ".join(f"{v:>7.3f}" if k == "mrr" else f"{v:>7.0%}" for k, v in r["metrics"].items()))
    print(f"(cut = chunks longer than the model's {model.max_seq_length}-token input. Current setting: "
          f"{chunk.CHUNK_SIZE}/{chunk.OVERLAP}, which should match `--exact`.)")
    save("retrieval_ablation", {"date": f"{datetime.now():%Y-%m-%d %H:%M}", "command": command(),
                                "golden": golden.default_path().name, "embedding_model": embed.MODEL_NAME,
                                "search": "exact numpy, in memory", "n_questions": len(items), "results": results})


# ----------------------------------------------------------------- self-test (no index, no model)
def selftest():
    def hit(doc_id: str, page: int, text: str = "x") -> retrieve.Hit:
        return retrieve.Hit(f"{doc_id}:p{page}:c1", doc_id, "invoice", page, text, 0.5, "http://example")

    item = {"doc_id": "a", "pages": [2, 3], "quote": "late  charge of\n5%"}
    hits = [hit("b", 2), hit("a", 1), hit("a", 3, "no quote here"), hit("a", 2, "a late charge of 5% applies")]
    cases = [
        ("rank = first right doc+page", first_rank(hits, item) == 3),
        ("same page, other doc is no hit", first_rank([hit("b", 2)], item) is None),
        ("quote rank needs the quote", first_rank(hits, item, need_quote=True) == 4),
        ("no hits -> None", first_rank([], item) is None),
        ("hit@k and MRR", metrics([1, 3, None, 10], [1, None, None, None])
         == {"hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.5, "hit@10": 0.75, "mrr": (1 + 1 / 3 + 1 / 10) / 4, "quote@5": 0.25}),
        ("ablation chunks respect the size", all(len(c["text"]) <= 400 for c in chunk_corpus(400, 80))),
        ("ablation = chunk.py at the current setting", chunk_corpus(chunk.CHUNK_SIZE, chunk.OVERLAP) == chunk.chunk_all()),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in cases)}/{len(cases)} passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif args == ["--ablation"]:
        ablation()
    else:
        main(exact="--exact" in args)
