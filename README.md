# Financial document AI: evidence-checked extraction, cited Q&A and a verified agent

[![docker](https://github.com/Ron-Salama/rag-support-agent/actions/workflows/docker.yml/badge.svg)](https://github.com/Ron-Salama/rag-support-agent/actions/workflows/docker.yml)

Lending and finance workflows run on numbers buried in invoices, financial statements, loan
agreements and appraisals, and a wrong number that looks right is worse than no number. This project
reads 15 public financial documents and turns them into **validated structured data**: every value
must come with a verbatim quote that code finds in the document and that contains the value, business
rules check the numbers, and anything doubtful goes to a **human review queue** instead of downstream.
On the same documents it **answers questions with page-level citations, or refuses**, and runs a
tool-using **agent whose answer a second model verifies** before anyone sees it. Everything is
measured against hand-made labels. It runs on a laptop for $0: local embeddings and Chroma, with the
LLM on the Gemini free tier (cloud) or a local Ollama model, served by FastAPI and packaged for Docker.

## Results

Every number here comes from a dated file in [`evals/results/`](evals/results/), named under each table.

### Extraction vs hand labels: 15 documents, 131 fields

Model `gemini-3.5-flash-lite`; predictions of the current run are in `outputs/predictions/`. A field is
right when it matches the hand label, or when both are empty. A **false fill** is the dangerous error:
the label is empty (the document does not print that value) but the model returned one anyway. 18 of
the 131 labels are empty.

| | Baseline, 2026-10-02 | Same predictions, fixed grader | **Current, 2026-10-04** |
|---|---|---|---|
| Fields right | 126/131 (96.2%) | 127/131 (96.9%) | **128/131 (97.7%)** |
| False fills | 1 of 18 | 1 of 18 | 1 of 18 |
| Invoices (4 documents) | 34/36 | 34/36 | 35/36 |
| Financial statements (4) | 31/32 | 31/32 | 31/32 |
| Loan agreements (4) | 36/36 | 36/36 | 36/36 |
| Appraisals (3) | 25/27 | 26/27 | 26/27 |
| Document status | 13 `ok`, 2 `needs_review` | 13 `ok`, 2 `needs_review` | 14 `ok`, 1 `needs_review` |
| File | `extraction_20261002-1913.json` | `extraction_20261004-1529_rescore_of_extraction_20261002-1913.json` | `extraction_20261004-1530.json` |

- One of the two points gained is a grader fix, not a better pipeline (middle column, item 5 below).
- The current false fill is new: `inv_capozzi_legal` `amount_due` = 165.00. In 3 re-runs with the cache
  off it was right 2 of 3 times (`extraction_variance_20261004-1532.json`), so it is run-to-run
  variation. I did not tune the prompt for it, because tuning on the 15 test documents would overfit.
- Text is compared loosely (case, punctuation, "Inc"/"Ltd"; addresses by part). In this run that forgave
  "Ltd" vs "Ltd.", an address that adds the city and state, and one `bill_to` that deserves a stricter
  look (item 4). No number needed the 0.5% numeric tolerance.

### Retrieval: is the right page in the top k? (no LLM calls)

Measured on a **24-question draft golden set** (+ 8 refusal traps), drafted with AI assistance and
checked against the source pages by `python -m evals.golden` ([`evals/GOLDEN_README.md`](evals/GOLDEN_README.md)).
A hit = a retrieved chunk from the expected document AND page; quote@5 = that chunk also contains the
expected quote. One question = 4.2 points, so this is a coarse first measurement, not a benchmark.

| hit@1 | hit@3 | hit@5 | hit@10 | quote@5 | MRR@10 |
|---|---|---|---|---|---|
| 14/24 (58%) | 19/24 (79%) | 20/24 (83%) | 22/24 (92%) | 20/24 (83%) | 0.698 |

`retrieval_20261001-175001.json` (Chroma). Exact numpy search gives identical numbers
(`retrieval_20261001-175010.json`); both were reproduced the same day (`..-183436.json`, `..-183832.json`).

Chunk-size ablation, exact search, same questions (`retrieval_ablation_20261001-175020.json`):

| chunk size / overlap (chars) | chunks | cut at the model's 512 tokens | hit@1 | hit@5 | quote@5 | MRR@10 |
|---|---|---|---|---|---|---|
| 400 / 0 | 3,177 | 0 | 16/24 (67%) | 21/24 (88%) | 15/24 (62%) | 0.752 |
| 400 / 80 | 3,799 | 0 | 15/24 (62%) | 21/24 (88%) | 14/24 (58%) | 0.730 |
| 800 / 0 | 1,664 | 4 | 15/24 (62%) | 21/24 (88%) | 20/24 (83%) | 0.732 |
| **800 / 150 (used)** | 1,860 | 4 | 14/24 (58%) | 20/24 (83%) | 20/24 (83%) | 0.698 |
| 1600 / 0 | 916 | 120 | 15/24 (62%) | 20/24 (83%) | 20/24 (83%) | 0.699 |
| 1600 / 320 | 998 | 143 | 14/24 (58%) | 19/24 (79%) | 19/24 (79%) | 0.672 |

Most gaps are 1-2 questions: noise, not a winner. What is clear: 400-character chunks often split the
passage an answer needs (quote@5 58-62%), and 1,600-character chunks overflow the embedding model's
512-token input. Two zoning/parking questions in the long appraisals miss at every size, so chunk
size is not their fix.

### Answer quality and the agent: not measured yet

Only smoke runs exist (2-5 questions each, marked `"partial": true` in `evals/results/answers_*.json`).
They show the pipeline runs end to end; **they are not a result.** The full measurement is
`python -m evals.answer_eval` (all 32 questions, about 2 LLM calls each; `--agent` for the agent).

## What went wrong and what I changed

The baseline got 5 fields wrong. Each was traced to a root cause before anything changed (details
and evidence: [`DECISIONS.md`](DECISIONS.md) C22-C26).

1. **The model invented a tax and "proved" it with a real quote.** `inv_contoso_stmt6` prints only
   "Sales Tax 3%"; the model returned tax = 311.25 (3% of 10,375) with the quote "Sales Tax 3%". The
   quote existed, so the value passed as `ok`. **Fix:** the value must be printed *inside* its own
   quote (numbers with commas, `$`, `%`, "million", bracket negatives; dates in common printed forms) (C22).
   **After:** tax = null in the scored run and 3/3 re-runs. The invoice now goes to `needs_review`,
   because its sum can't be checked (open decision C5).
2. **A retry broke a correct field.** In a re-run, a retry asked to fix one date quote, and the model
   also swapped Omega's correct FFO (223,963) for AFFO (261,357): 125/131 (`extraction_20261002-1928.json`).
   **Fix:** a retry re-asks only the failing fields, and code merges back only those
   (`keep_passed_fields`); a self-test proves that a passed field can't change (C24).
3. **Flattened tables broke the quote check.** Omega and NHI went to review with the right values.
   Table cells are joined with "|" (fixed first), and the column header "Three Months Ended" / "June 30," /
   "2026" sits on three lines. A first tolerant matcher ("words in order within 300 characters") was too
   loose: it accepted "Total revenues 282,506", the prior-year column. **Fix:** words must follow each
   other as in the text, the only jump allowed is to the first word of a later line, and a number is
   one word (C23). **After:** Omega and NHI are `ok` on the first attempt.
4. **Layout trap: Bill To and Ship To side by side** (`inv_sammy`). The columns flatten to "Taylor
   Riddel Taylor's Store" on one line, and the rule said "the customer company, never a person". The
   customer is a person, so the model took the Ship To name: the rule was at fault, not the model.
   **Fix:** `bill_to` = whoever is billed (a person if the customer is one), never the Ship To; the same
   wording in the schema the model reads and in the [labeling guide](data/labels/LABELING_GUIDE.md) (R4).
   **After:** right in the scored run and 3/3 re-runs. **Still open:** after the rewording,
   `inv_contoso_100` returned the Bill To department line "Microsoft Finance" instead of the customer
   name "MICROSOFT CORPORATION" (exactly right in the baseline). The grader's substring rule counts it
   as right, so 128/131 includes one field a strict grader would reject.
5. **The grader was too strict.** `appr_valleyview_alf` "..., Walnut Creek, CA 94595" was marked wrong
   against the label "..., Contra Costa County, California 94595". **Fix:** addresses are compared by
   part (street, city, state, ZIP; CA = California; county ignored), with no fuzzy match: an 85%
   similarity rule calls "1228 Rossmoor Pkwy" and "1229 Rossmoor Pkwy" the same (C25). **After:**
   re-grading the baseline's saved predictions changed only that field, 126 to 127.

Still wrong: `fs_sfr_fy2025` `net_income` (the model took plain "Net income" 33,306; the schema asks
for the "attributable to common stockholders" line, 7,575) and `appr_hallandale_comm` `property_type`
(an auto-body shop: "industrial" vs the label "other", C21).

## Architecture

```
 data/manifest.csv (15 documents + source URLs) -> to_text.py -> data/text/     (key pages: Part 1)
                                                                 data/fulltext/ (every page: Parts 2-3)
 PART 1  EXTRACTION  extract.py, schemas.py
   text -> LLM -> JSON -> [1] Pydantic schema  [2] verbatim evidence quote found in the text
                          [3] value printed inside its quote  [4] business rules (sums, dates)
                       -> [5] retry with the problems, re-asking only failing fields (max 3)
   -> ok | needs_review -> [6] human review queue (outputs/review_queue/) | failed

 PART 2  RAG  rag/: page-bounded chunks -> bge-small (CPU) -> Chroma -> top-5 retrieval
   question -> [7] gate 1: best similarity >= 0.60 -> LLM -> [8] gate 2: model says "found"
            -> [9] every [S#] citation checked in code -> cited answer, or a refusal

 PART 3  AGENT  agent/: hand-written loop over search_docs, get_page, list_documents, extract_fields (max 6 steps)
   -> [10] verifier: a second model checks every claim against the tool results; code overrules it
   -> verified | revised (1 retry) | needs_human_review (answer withheld, case queued)

 PART 4  EVALS  score_extraction, retrieval_eval (+ ablation), refusal_threshold, answer_eval + LLM judge
 PART 5  SERVE  api.py (FastAPI), Dockerfile (model + index built in), GitHub Actions CI
```

## Quick start

Python 3.13, PowerShell, from the repository root (macOS/Linux: `python3.13 -m venv .venv` and
`source .venv/bin/activate`, then the same commands).

**1. Install and check everything: no API key, 0 LLM calls.**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only torch first (no CUDA download)
python -m pip install -r requirements.txt
python -m ragagent.rag.ingest      # build the index: chunk -> embed -> Chroma (about 1.5 min on a CPU)

# 15 self-test modules, 153 cases, fake LLMs: each line should end "N/N passed"
"ragagent.llm", "ragagent.extract", "ragagent.download", "ragagent.rag.chunk", "ragagent.rag.retrieve",
"ragagent.rag.answer", "ragagent.agent.tools", "ragagent.agent.loop", "ragagent.agent.verifier",
"ragagent.agent.orchestrate", "evals.retrieval_eval", "evals.answer_eval", "evals.judge",
"evals.judge_agreement", "evals.score_extraction" | ForEach-Object { "{0,-28} {1}" -f $_, (python -m $_ --selftest 2>$null | Select-Object -Last 1) }

python -m evals.golden              # validate the golden set against the document text
python -m evals.retrieval_eval      # reproduces the retrieval table (writes a new dated file to evals/results/)
python -m evals.score_extraction    # re-grades the committed predictions: 128/131 (also writes a new file)
python -m ragagent.rag.retrieve "What late charge applies if a loan payment is late?"   # top-5 chunks + scores
```

**2. Run the API.** For answers, paste a free Gemini API key (Google AI Studio) into `.env`; without
one, `/health`, `/documents` and the gate-1 refusals still work.
```powershell
Copy-Item .env.example .env
python -m uvicorn ragagent.api:app --port 8000     # interactive docs: http://localhost:8000/docs
python scripts/smoke_test.py                       # in a second window: /health, /documents, an off-topic /ask
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType application/json `
  -Body '{"question": "What late charge applies if a loan payment is late?"}'
```

| Endpoint | Returns | LLM calls |
|---|---|---|
| `GET /health` | server up, index built | 0 |
| `GET /documents` | the 15 documents with source URLs | 0 |
| `POST /ask` `{"question"}` | cited answer, or a refusal with the reason | 0-1 |
| `POST /extract` `{"doc_id"}` | validated fields + evidence + status (`ok` / `needs_review` / `failed`) | 1-3 |
| `POST /agent` `{"question"}` | verified answer + full trace (every tool call and verdict) | 2-14, +1-3 per `extract_fields` |

**3. Docker.**
```powershell
docker build -t rag-support-agent .
docker run --env-file .env -p 8000:8000 rag-support-agent    # the key is passed at run time, never stored in the image
python scripts/smoke_test.py
```
The image holds CPU-only PyTorch, the code, the document text, the embedding model and the built index,
runs as a non-root user, and needs the internet only for the LLM. **Status:** [CI](.github/workflows/docker.yml)
builds it on every push, smoke-tests it with no API key and runs all 15 self-test modules inside it with
the network off. The first run passed on 2026-10-04 (build 6 min, smoke test 3/3, 153/153 self-test cases,
image 1.96 GB: [run 37205328703](https://github.com/Ron-Salama/rag-support-agent/actions/runs/37205328703)).
It has not yet been run on a local Docker install. Windows install notes: [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md).

## Design decisions

Every real choice, with the options and the evidence, is in [`DECISIONS.md`](DECISIONS.md). The main ones:

- **Evidence over trust** (C3, C22, C23): no verbatim quote containing the value, no value. It is a cheap,
  code-only hallucination check, and reviewers see where every value came from.
- **The LLM never does arithmetic** (C2): numbers are extracted as printed, with their units; scaling
  and sum checks happen in code.
- **Retries can fix, not break** (C4, C24): at most 3 attempts, each re-asking only the failing fields;
  anything still doubtful goes to the human review queue.
- **Fail closed on questions** (C8, C9): gate 1's threshold, 0.60, was chosen from data (off-topic probes
  0.425-0.569, answerable 0.628-0.820; `python -m evals.refusal_threshold`, re-run 2026-10-04, prints
  only). For a lender, "I don't know" beats an untraceable answer.
- **Page-bounded chunks** (C6, C14): a chunk never crosses a page, so every citation is one page.
- **A hand-written agent loop, no framework** (C11, C12): a step limit, tool errors fed back to the
  model, a verifier on a different model, code checks that overrule it, unverified answers withheld.
- **One small LLM interface** (C1): `generate_json` and `chat` with tools, a disk cache, backoff on rate
  limits, a loud stop on a bad key or exhausted quota; swapping providers is one class.
- **A self-contained image** (C17): model and index built in at build time, the key only at run time.

## Data

15 public, non-confidential documents; every file's source URL is in [`data/manifest.csv`](data/manifest.csv).

| Type | # | Source | Terms |
|---|---|---|---|
| Invoices | 4 | 3 sample invoices with fictional companies from open-source repositories (Azure-Samples, invoice2data test corpus); 1 legal-services invoice from a US bankruptcy-court fee application (University of Florida archive) | samples: MIT licence; court exhibit: public record |
| Financial statements | 4 | quarterly / annual earnings releases of listed companies (SEC EDGAR, 8-K Exhibit 99.1) | public SEC filings |
| Loan agreements | 4 | promissory notes and loan agreements filed as SEC EDGAR exhibits | public SEC filings |
| Appraisals | 3 | 2 appraisal reports filed as SEC EDGAR exhibits; 1 from a US city's public meeting agenda packet | public records |

The documents belong to their authors and filers and are used unchanged, for research and
demonstration. The original files (`data/raw/`, about 16 MB) are **not committed**:
`python -m ragagent.download` re-fetches them from the manifest URLs (sec.gov requires a line
`SEC_USER_AGENT=Your Name you@example.com` in `.env`, sent to sec.gov only). The extracted text
(`data/text/`, `data/fulltext/`: 476 pages, 1,860 chunks) is committed, so the API, the evals and the
Docker image work without the originals. Hand labels: [`data/labels/`](data/labels/).

## Moving to AWS

A plan, **not built**, based on AWS documentation read in late September 2026. The LLM sits behind one
interface, so the rest of the pipeline would not change.

| Local piece | AWS counterpart |
|---|---|
| `llm.py` chat + tool calling | **Bedrock Runtime Converse API**: tool specs -> `toolConfig.tools[].toolSpec`, calls -> `toolUse`, results -> `toolResult`; an IAM role instead of a key |
| `extract.py` | Converse with a "record extraction" tool, keeping Pydantic + retries; or **Bedrock Data Automation** blueprints, whose per-field confidence would feed the same review queue |
| chunk + embed + Chroma | **Bedrock Knowledge Bases** (Titan Text Embeddings v2 or Cohere Embed; OpenSearch Serverless, Aurora pgvector or S3 Vectors) |
| refusal gates + verifier | **Bedrock Guardrails** contextual grounding check, next to our own checks |
| `outputs/` + review queue | S3 -> **Snowflake** (raw `VARIANT` table -> dbt models with tests, e.g. line items sum to the total) |

## Limitations and next steps

- **Small:** 15 documents and 131 labeled fields, and the extraction rules were improved on the same documents they are scored on.
- **Draft golden set:** 24 + 8 AI-drafted questions, checked by script; hand-written questions and a held-out split come next.
- **Answer quality not measured yet**, and the LLM judge is not yet calibrated against 30+ human verdicts (`evals.judge_agreement`).
- **Free-tier model:** Gemini 3.5 ignores `temperature`, so runs vary (hence the re-runs). The Ollama backend is tested with a fake client only.
- **Refusals:** on-topic questions no document answers pass the score gate; only gate 2 and the verifier stop them.
- **Text only:** no OCR for scanned PDFs, and tables are flattened to text (the source of problems 3 and 4).
- **Docker:** built and tested in CI only (1.96 GB image, mostly CPU PyTorch); not yet run on a local Docker install.
- **Not production:** no auth or rate limiting, shared state that is not thread-safe, a review queue that is a folder of JSON files.

Next: the full `answer_eval` (plain and `--agent`) on all 32 questions; hybrid keyword + vector search
or a reranker for the two zoning/parking misses; deduplicated `/ask` citations by (document, page);
payment cross-checks in code, such as an invoice's `amount_due` against its `total` plus any printed
previous balance (R1).

## How it was built

Scoped, hand-labeled and evaluated by the author, and implemented with AI-assisted development
(Claude Code). The 15 extraction answer keys were labeled by hand, blind to model output, and the
error analysis above decided what to change; the design decisions and their evidence are in
[`DECISIONS.md`](DECISIONS.md). Only the draft golden questions were AI-drafted, and a script checks
them against the documents.
