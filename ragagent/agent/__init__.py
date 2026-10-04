"""Part 3 - the agent: an LLM that decides for itself which tools to call, plus a second LLM that checks it.

    tools.py        the tools the model may call (search_docs, list_documents, get_page, extract_fields)
    loop.py         the tool-calling loop with a step limit                       (the agent's core loop)
    verifier.py     the second agent: is every claim backed by what the tools returned?
    orchestrate.py  worker -> verifier -> one retry with feedback -> human review queue

    python -m ragagent.agent.orchestrate "What late charge applies if a loan payment is late?"

Difference from Part 2 (rag/answer.py): there, OUR code decides the steps (always: search once,
then answer). Here the MODEL decides: search again with other words, read a whole page, list
the documents first... Our code only runs what it asks for, and stops it after a few steps.
"""
# Imported first on purpose (same as rag/__init__.py): config sets HF_HOME before any model loads.
from ragagent import config  # noqa: F401
