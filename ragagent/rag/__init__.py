"""Part 2 - RAG (retrieval-augmented generation): find the right pages, then answer from them.

    chunk.py     full document text -> small page-bounded pieces ("chunks") with metadata
    embed.py     text -> vector (a list of 384 numbers that captures the meaning)
    store.py     save / load the vectors (Chroma + a plain numpy copy)
    ingest.py    run the three steps above for every document:  python -m ragagent.rag.ingest
    retrieve.py  question -> the k most similar chunks           (the retrieval core loop)
    answer.py    question -> retrieved chunks -> LLM -> answer with [S1]-style citations, or a refusal
"""
# Imported first on purpose: config sets HF_HOME (model downloads go to the project's .cache/) and that
# must happen BEFORE sentence_transformers / huggingface_hub are imported anywhere.
from ragagent import config  # noqa: F401
