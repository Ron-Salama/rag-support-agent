"""Retrieval: question -> the k chunks whose meaning is closest to it.

    python -m ragagent.rag.retrieve --selftest              # test the search loop (offline fakes + Chroma check)
    python -m ragagent.rag.retrieve "who guarantees the loan?"   # show the top 5 chunks and their scores

Two ways to do the same search:
  - search_bruteforce: compare the question vector with EVERY chunk vector, by hand.
    1,860 chunks x 384 numbers is tiny, so this takes milliseconds. (The core loop.)
  - search: ask Chroma. Chroma uses an index (HNSW, a graph of "neighbour" links) so it stays
    fast with millions of chunks, and it can filter on metadata. The price: HNSW is APPROXIMATE.
    It almost always returns the same top k as the exact search, but now and then it misses
    one chunk that brute force finds.

Score = cosine similarity, higher = closer. Because every vector has length 1 (embed.py
normalizes them), cosine similarity is just the dot product: sum(q[i] * c[i]).
Chroma reports a DISTANCE instead (cosine distance = 1 - similarity), so we convert back.
"""
import sys
from dataclasses import dataclass

import numpy as np

from ragagent.rag import embed, store


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_type: str
    page: int
    text: str
    score: float        # cosine similarity: 1.0 = same meaning, ~0.3-0.5 = unrelated (for this model)
    source_url: str


# ===================================================================================
# Core loop - kept small and framework-free on purpose; offline self-test:
#   python -m ragagent.rag.retrieve --selftest
# ===================================================================================
def search_bruteforce(query_vec: np.ndarray, chunk_vecs: np.ndarray, chunks: list[dict], k: int = 5,
                      doc_id: str | None = None, doc_type: str | None = None) -> list[Hit]:
    """What a vector database does, by hand (~10 lines).

    Inputs: query_vec = the question's vector (length 1); chunk_vecs = matrix, row i is the
    vector of chunks[i]; chunks = dicts with chunk_id, doc_id, doc_type, page, text, source_url.

      1. For every chunk i (skip it if doc_id / doc_type is given and does not match):
             score = dot product of query_vec and chunk_vecs[i]   (np.dot; = cosine similarity)
         and keep (score, i).
      2. Sort the kept pairs by score, highest first. Equal scores keep their original order
         (Python's sort is "stable", so sort by score only).
      3. Return the first k as Hit objects: Hit(**chunks[i], score=float(score)).
         If fewer than k chunks are left, return them all (no crash).

    The fast numpy way to do step 1 for all chunks at once is  chunk_vecs @ query_vec
    - same numbers, written as one matrix-times-vector product.
    Test it with:  python -m ragagent.rag.retrieve --selftest
    """
    scored = []
    for i, vec in enumerate(chunk_vecs):
        if doc_id and chunks[i]["doc_id"] != doc_id:
            continue
        if doc_type and chunks[i]["doc_type"] != doc_type:
            continue
        scored.append((float(np.dot(query_vec, vec)), i))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [Hit(**chunks[i], score=score) for score, i in scored[:k]]


# ----------------------------------------------------------------- the Chroma way
def _where(doc_id: str | None, doc_type: str | None) -> dict | None:
    """Metadata filter in Chroma's syntax. Two conditions must be wrapped in "$and"."""
    conditions = []
    if doc_id:
        conditions.append({"doc_id": doc_id})
    if doc_type:
        conditions.append({"doc_type": doc_type})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(query: str, k: int = 5, doc_id: str | None = None, doc_type: str | None = None) -> list[Hit]:
    """Question text -> top-k Hits from Chroma, optionally only from one document / document type."""
    res = store.get_collection().query(
        query_embeddings=[embed.embed_query(query)], n_results=k,
        where=_where(doc_id, doc_type), include=["documents", "metadatas", "distances"],
    )
    # Chroma answers for a LIST of queries; we sent one, so take element [0] of each list.
    rows = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    return [Hit(chunk_id=cid, doc_id=m["doc_id"], doc_type=m["doc_type"], page=int(m["page"]), text=text,
                score=1.0 - float(dist), source_url=m["source_url"]) for cid, text, m, dist in rows]


def search_numpy(query: str, k: int = 5, doc_id: str | None = None, doc_type: str | None = None) -> list[Hit]:
    """Same as search(), but through the exact search_bruteforce over chroma/vectors.npy."""
    vecs, chunks = store.load_numpy()
    return search_bruteforce(embed.embed_query(query), vecs, chunks, k, doc_id, doc_type)


# ----------------------------------------------------------------- self-test
def _fake_chunks(n: int) -> list[dict]:
    types = ["invoice", "appraisal", "appraisal", "loan_agreement"]
    return [{"chunk_id": f"d{i}:p1:c1", "doc_id": f"d{i}", "doc_type": types[i % 4], "page": 1,
             "text": f"chunk {i}", "source_url": "http://example"} for i in range(n)]


def selftest():
    # float32, like the real chroma/vectors.npy (a numpy float32 is NOT a Python float - see the last case)
    q = np.array([1.0, 0.0], dtype=np.float32)
    vecs = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8], [0.6, 0.8]], dtype=np.float32)   # scores 0, 1, 0.6, 0.6
    chunks = _fake_chunks(4)   # doc types: d0 invoice, d1 appraisal, d2 appraisal, d3 loan_agreement
    cases = [
        ("ranking: best first", lambda r: [h.doc_id for h in r] == ["d1", "d2", "d3"], dict(k=3)),
        ("scores are the dot products", lambda r: [round(h.score, 6) for h in r] == [1.0, 0.6, 0.6, 0.0], dict(k=4)),
        ("k bigger than corpus -> all", lambda r: len(r) == 4, dict(k=10)),
        ("ties keep original order", lambda r: [h.doc_id for h in r[1:3]] == ["d2", "d3"], dict(k=4)),
        ("filter by doc_id", lambda r: [h.doc_id for h in r] == ["d3"], dict(k=5, doc_id="d3")),
        ("filter by doc_type", lambda r: [h.doc_id for h in r] == ["d1", "d2"], dict(k=5, doc_type="appraisal")),
        # filter FIRST, then take the top k (top-1 overall is d1, an appraisal: filtering after would give [])
        ("filter before top-k", lambda r: [h.doc_id for h in r] == ["d3"], dict(k=1, doc_type="loan_agreement")),
        ("filter matching nothing -> []", lambda r: r == [], dict(k=5, doc_id="nope")),
        ("score is a Python float", lambda r: isinstance(r[0], Hit) and type(r[0].score) is float, dict(k=1)),
    ]
    results = []
    for name, check, kwargs in cases:
        try:
            r = search_bruteforce(q, vecs, chunks, **kwargs)
            ok, got = check(r), [f"{h.doc_id}:{h.score:.2f}" for h in r]
        except Exception as e:  # noqa: BLE001 - show any crash as a failed case
            ok, got = False, f"crashed: {type(e).__name__}: {e}"
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name:<32} got {got}")

    if store.index_exists():   # real index: the exact loop vs Chroma's approximate index, top 5
        vecs, chunks = store.load_numpy()
        for question in ["Who is the guarantor of the loan?", "What is the occupancy rate of the property?",
                         "How did the company's funds from operations change compared to last year?"]:
            mine = search_bruteforce(embed.embed_query(question), vecs, chunks, 5)
            chroma = search(question, 5)
            # Compare SCORES rank by rank, not chunk ids: two chunks can have the exact same score (the
            # two HUD notes are the same printed form) and come back in either order, and Chroma's HNSW
            # may miss a chunk now and then. Exact search must score >= Chroma at every rank.
            ok = len(mine) == len(chroma) == 5 and all(m.score >= c.score - 1e-5 for m, c in zip(mine, chroma))
            results.append(ok)
            print(f"{'PASS' if ok else 'FAIL'}  top-5 as good as Chroma's: {question!r}")
            if [h.chunk_id for h in mine] != [h.chunk_id for h in chroma]:
                print(f"        (different ids - a tie or an HNSW miss{'' if ok else ', or a bug in the exact loop'})")
                print(f"        exact:  {[f'{h.chunk_id} {h.score:.4f}' for h in mine]}")
                print(f"        chroma: {[f'{h.chunk_id} {h.score:.4f}' for h in chroma]}")
    else:
        print("SKIP  Chroma comparison (no index yet - run: python -m ragagent.rag.ingest)")
    print(f"\n{sum(results)}/{len(results)} passed")


def show(question: str, k: int = 5):
    for h in search(question, k):
        preview = " ".join(h.text.split())[:110]
        print(f"{h.score:.3f}  {h.chunk_id:<32} {preview}")


if __name__ == "__main__":
    # Printed to a pipe or file (| Select-String, > out.txt), Windows uses an old code page that has
    # no '\x96' or '§' and print() would crash; errors="replace" shows '?' for those instead.
    sys.stdout.reconfigure(errors="replace")
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif args:
        show(" ".join(args))
    else:
        print(__doc__)
