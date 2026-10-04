"""Embeddings: turn text into a vector (384 numbers) so that similar MEANING = nearby vectors.

    python -m ragagent.rag.embed      # embed 3 sentences and print their similarities (try it!)

The model (BAAI/bge-small-en-v1.5, ~130 MB) runs locally on the CPU: free, no API, and the
documents never leave the machine. First use downloads it to .cache/huggingface inside the project folder.

Two details that matter:
  - normalize_embeddings=True makes every vector length 1. Then the dot product of two
    vectors IS their cosine similarity (1 = same direction/meaning, ~0 = unrelated), which
    is what retrieve.py ranks by.
  - bge models were trained with an instruction in front of SEARCH QUERIES only (not in
    front of the passages). Adding it to the query improves retrieval; adding it to the
    passages too would make it worse. So there are two functions: embed_passages / embed_query.
"""
import numpy as np

from ragagent import config  # noqa: F401  (sets HF_HOME before sentence_transformers is imported)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None  # loaded once, on first use (loading takes a few seconds)


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:  # already downloaded? then load it without touching the internet
            _model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
        except Exception:  # noqa: BLE001 - first run: not on disk yet, so download it (~130 MB)
            _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    """Chunks -> matrix of shape (len(texts), 384), one normalized row per text."""
    vecs = get_model().encode(texts, batch_size=32, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 200, convert_to_numpy=True)
    return vecs.astype(np.float32)


def embed_query(question: str) -> np.ndarray:
    """A question -> one normalized vector of shape (384,), with the bge query instruction."""
    return embed_passages([QUERY_PREFIX + question])[0]


def compare(texts: list[str]):
    """Play with it: embed your own sentences, peek at the numbers, compare the 1st with the rest."""
    vecs = embed_passages(texts)
    print(f"each sentence -> {vecs.shape[1]} numbers. The first 6 of each:")
    for text, v in zip(texts, vecs):
        print(f"  {np.round(v[:6], 3)} ...  {text!r}")
    print(f"\nsimilarity to {texts[0]!r}  (1.0 = same meaning, ~0.3-0.5 = unrelated for this model):")
    for text, score in zip(texts[1:], vecs[1:] @ vecs[0]):
        print(f"  {score:.3f}  {text!r}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:]:  # python -m ragagent.rag.embed "sentence one" "sentence two" ...
        compare(sys.argv[1:])
        sys.exit()
    passages = ["The loan bears interest at a fixed rate of 4.5% per annum.",
                "The property is a 120-unit apartment complex built in 1985.",
                "Revenue increased due to new triple-net leases."]
    q = embed_query("What is the interest rate on the note?")
    p = embed_passages(passages)
    print(f"vector shape {p.shape}, length of each vector = {np.linalg.norm(p, axis=1).round(3)}")
    for text, score in zip(passages, p @ q):
        print(f"  similarity {score:.3f}  {text}")
