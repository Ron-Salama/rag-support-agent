"""Where the vectors live: a Chroma vector database, plus a plain numpy copy for exact search.

Chroma (chroma/ folder, a small database on disk) stores, per chunk: an id, the text, the
metadata (doc_id, doc_type, page, source_url) and the vector. Its job at question time:
"give me the k vectors closest to this one, optionally only where doc_type = 'appraisal'",
fast, using an index (HNSW) instead of comparing against every single vector.

The numpy copy (chroma/vectors.npy + chroma/chunks.json) holds the same vectors and chunks
so retrieve.search_bruteforce can do the same search by hand - it shows what a vector
database actually does, and checks that Chroma gives the same answer.

IMPORTANT: Chroma has a "default embedding function" that would download its own model to
C:\\Users\\...\\.cache. We never use it: we always pass our own vectors (embed.py) and create /
open the collection with embedding_function=None.
"""
import json

import numpy as np

from ragagent import config

COLLECTION = "docs"
VECTORS = config.CHROMA / "vectors.npy"   # shape (n_chunks, 384), row i = chunk i
CHUNKS = config.CHROMA / "chunks.json"    # list of chunk dicts, same order as the rows
BATCH = 500                               # Chroma limits how many items one call may add

_client = None


def client():
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings

        # anonymized_telemetry=False: Chroma would otherwise send usage pings over the internet.
        _client = chromadb.PersistentClient(path=str(config.CHROMA), settings=Settings(anonymized_telemetry=False))
    return _client


def get_collection():
    """Open the existing collection (made by ingest). Raises if ingest was never run."""
    return client().get_collection(COLLECTION, embedding_function=None)


def rebuild(chunks: list[dict], vectors: np.ndarray):
    """Throw away the old index and store these chunks + vectors (Chroma AND the numpy copy)."""
    # Delete the numpy copy FIRST and write it LAST: if ingest crashes half-way, index_exists()
    # then says "no index" (the API answers 503 "run ingest") instead of trusting a half-built one.
    VECTORS.unlink(missing_ok=True)
    CHUNKS.unlink(missing_ok=True)
    c = client()
    if COLLECTION in [col.name for col in c.list_collections()]:
        c.delete_collection(COLLECTION)
    # space "cosine": Chroma returns distance = 1 - cosine similarity (0 = identical direction).
    col = c.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
    for start in range(0, len(chunks), BATCH):
        part = chunks[start:start + BATCH]
        col.upsert(
            ids=[ch["chunk_id"] for ch in part],
            documents=[ch["text"] for ch in part],
            metadatas=[{k: ch[k] for k in ("doc_id", "doc_type", "page", "source_url")} for ch in part],
            embeddings=vectors[start:start + BATCH],
        )
    np.save(VECTORS, vectors)
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return col.count()


def load_numpy() -> tuple[np.ndarray, list[dict]]:
    """The plain copy for brute-force search: (vectors matrix, chunk dicts in the same order)."""
    return np.load(VECTORS), json.loads(CHUNKS.read_text(encoding="utf-8"))


def index_exists() -> bool:
    """True once ingest has finished (rebuild writes these two files last)."""
    return VECTORS.exists() and CHUNKS.exists()
