# CLAUDE.md

Working rules for coding agents (Claude Code) in this repository. People should start with
[`README.md`](README.md); every design choice and its evidence is in [`DECISIONS.md`](DECISIONS.md).

## Purpose
Financial document AI on 15 public documents: evidence-checked extraction into validated JSON
(Part 1), cited answers or refusals over the full text (Part 2), a tool-using agent with a verifier
(Part 3), evals (Part 4), a FastAPI service packaged for Docker (Part 5).

## Layout
```
ragagent/           the package: extract.py, schemas.py, llm.py, api.py, rag/, agent/
evals/              eval scripts; evals/results/ = dated result files (committed)
data/manifest.csv   the 15 documents + source URLs     data/labels/    hand labels (the answer key)
data/text/          key pages per document (Part 1)     data/fulltext/  every page (RAG)
data/raw/           original files (git-ignored; rebuilt by python -m ragagent.download)
outputs/            predictions/ (committed); review_queue/ (runtime, git-ignored)
chroma/             vector index (git-ignored; rebuilt by python -m ragagent.rag.ingest)
```

## Commands
Python 3.13; development is on Windows/PowerShell, CI runs Linux. From the repository root:
```powershell
py -3.13 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch first
python -m pip install -r requirements.txt
python -m ragagent.rag.ingest                  # build chroma/ (no LLM, ~1.5 min on a CPU)
python -m ragagent.extract --selftest          # offline self-test (fake LLM); the 15 modules are the
                                               # loop in .github/workflows/docker.yml and the README
python -m evals.golden                         # check the golden set against data/fulltext
python -m evals.score_extraction               # re-grade committed predictions vs hand labels (offline)
python -m evals.retrieval_eval                 # retrieval hit@k / MRR (no LLM)
python -m ragagent.extract [doc_id ...]        # LIVE: 1-3 LLM calls per document
python -m evals.answer_eval [--agent]          # LIVE: answers + LLM judge
python -m uvicorn ragagent.api:app --port 8000 # API; python scripts/smoke_test.py checks it, 0 LLM calls
```
Eval scripts write a new dated file to `evals/results/`; delete it when the run was only a check.

## Constraints
- No paid APIs required: Gemini free tier or a local Ollama model; embeddings run locally on the CPU.
- Secrets only in `.env` (git-ignored; template in `.env.example`): never in code, logs, commits or
  the Docker image. The key is passed at run time.
- Documents must be public and non-confidential, each with its source URL in `data/manifest.csv`.
  `data/raw/` is never committed (C20).
- Hugging Face and LLM caches stay inside the project folder (`.cache/`, set in `ragagent/config.py`).

## Conventions
- **Evidence over trust:** an extracted value needs a verbatim quote that code finds in the text and
  that contains the value (C3, C22, C23). The LLM never does arithmetic; numbers are extracted as
  printed and checked in code (C2).
- **Fail closed:** an answer without a checked citation is refused or withheld; a doubtful extraction
  goes to the human review queue, never downstream as `ok` (C4, C9, C12).
- **Numbers come only from saved eval result files** in `evals/results/`, named next to the number.
  Never invent, estimate or round a result, and say when a run replayed answers from the LLM cache.
- **Every bug fix gets a self-test case that fails on the old code.**
- **Record design choices in `DECISIONS.md`:** the options, the choice, and the evidence (command +
  results file). IDs are stable: never renumber or reuse one.
- **Never show model output while hand-labeling:** labels are written blind, before any model output
  for that document exists (set `AGENT_EXTRACT_TOOL=0` while labeling new documents, C13).
- **Keep the self-test contract CI relies on:** each module prints `PASS`/`FAIL` lines and ends with
  `N/N passed`; CI fails on any `FAIL` line or a missing all-passed line (C18). A new module goes
  into the `docker.yml` loop and the README list.
- **Don't tune on the test set silently:** rules and prompts were improved on the same 15 scored
  documents; any further change says so, and run-to-run variation is checked with cache-off re-runs
  (`python -m evals.extraction_variance`, C26).
