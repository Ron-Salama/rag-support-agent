"""Ingestion: build the search index from scratch.

    python -m ragagent.to_text --full     # first: documents -> data/fulltext/*.txt (every page)
    python -m ragagent.rag.ingest         # then: chunk -> embed -> store in chroma/

    full text ──> chunk.py ──> ~1,900 chunks with metadata
                                   │
                  embed.py ──> one 384-number vector per chunk (local model, CPU)
                                   │
                  store.py ──> Chroma collection "docs"  +  chroma/vectors.npy + chunks.json

Re-run it whenever the documents, the chunking or the embedding model change: the old index
is deleted first, so there are never stale chunks left over.
"""
import time
from collections import Counter

from ragagent import config
from ragagent.rag import chunk, embed, store


def main():
    t0 = time.time()
    chunks = chunk.chunk_all()
    per_doc = Counter(c["doc_id"] for c in chunks)
    for doc_id, n in per_doc.items():
        print(f"{doc_id:<24} {n:>5} chunks")
    t1 = time.time()
    print(f"\nEmbedding {len(chunks)} chunks with {embed.MODEL_NAME} on CPU ...")
    vectors = embed.embed_passages([c["text"] for c in chunks])
    t2 = time.time()
    count = store.rebuild(chunks, vectors)
    t3 = time.time()
    print(f"\nStored {count} chunks from {len(per_doc)} documents in {config.CHROMA}")
    print(f"Time: chunk {t1 - t0:.1f}s + embed {t2 - t1:.1f}s + store {t3 - t2:.1f}s = {t3 - t0:.1f}s total")


if __name__ == "__main__":
    main()
