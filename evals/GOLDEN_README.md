# The golden set - how it works and how to extend it

A **golden set** is a list of questions whose right answer, and where it sits in the documents,
is known in advance. It is the answer key for the RAG system, the same way `data/labels/` is the
answer key for extraction. Every number Part 4 reports is "how well did the system do on THESE
questions", so the numbers are only as good as the questions.

> **`evals/golden_draft.jsonl`** is a draft golden set of **24 answerable questions + 8 refusal
> traps**, drafted with AI assistance (2026-10-01) and verified against the source text page by
> page: `python -m evals.golden` checks that every answer's quote is verbatim on its listed pages,
> and a review corrected two entries (a19, a21; see `DECISIONS.md` C14). It is to be extended with
> hand-written questions in **`evals/golden.jsonl`**; as soon as that file exists, every eval script
> uses it instead of the draft. With 24 answerable questions one question = 4.2 points, so treat
> numbers measured on the draft as coarse.

## Format: one JSON object per line (JSONL)

**Answerable question** (the documents contain the answer):
```json
{"id": "a01", "type": "answerable", "question": "What are the payment terms on the Sammy Maystone invoice?",
 "answer": "Net 30 days", "doc_id": "inv_sammy", "pages": [1], "quote": "Payment Terms: Net 30 days"}
```
(in the file it is ONE line; it is wrapped here only to fit the page)

| field | meaning |
|---|---|
| `id` | short and unique: `a01`, `a02` ... for answerable, `t01` ... for traps |
| `question` | asked the way a real user (a loan analyst) would ask it |
| `answer` | the short correct answer. The judge compares MEANING, so wording does not have to match |
| `doc_id` | the document that answers it (see `data/manifest.csv`) |
| `pages` | every page that answers it, as the `=== Page N ===` numbers in `data/fulltext/<doc_id>.txt` (NOT the number printed in the document's footer - for HTML filings they differ) |
| `quote` | a short snippet copied verbatim from that page (spaces/line breaks may differ). Proves the answer is really there |
| `notes` | optional: why the question is hard, what trap it sets |

**Trap** (sounds like it belongs here, but NO document answers it):
```json
{"id": "t01", "type": "trap", "question": "How many employees does Omega Healthcare Investors have?",
 "why_unanswerable": "The release gives no employee count, and no other document does."}
```
A line `{"_header": "..."}` is a comment and is skipped.

## How to write good questions

1. **Open the document and find the fact first**, then write the question. Copy the quote from
   `data/fulltext/<doc_id>.txt` (that is the exact text the system searches). To find the page,
   the easy way: put any page (e.g. `[1]`) and run `python -m evals.golden` - it answers
   `quote not found on pages [1] (found on: [48])`. It also tells you when the same quote is on
   more pages than you listed.
2. **Mix easy and hard.** Easy: a one-page invoice. Hard: the fact sits in a 190-page appraisal,
   or two documents look alike (the two HUD notes are the same printed form; three REITs all
   declared a dividend), or the page with the number never names the property.
3. **Cover every document**, at least one question each, more for the long ones.
4. **Ask about what a lender cares about**: guarantors, late charges, prepayment, payment terms,
   occupancy, year built, zoning, parking, dividends, guidance, number of facilities.
5. **Avoid the extraction fields** (`SCORED_FIELDS` in `ragagent/schemas.py`): the Part 1 eval
   already measures them against the hand labels, so the golden set covers the rest of each document.
6. **Traps**: on-topic and plausible. Before adding one, search the whole corpus
   (`Select-String -Path data\fulltext\*.txt -Pattern "salary"`) to make sure NO document
   answers it. The best traps have a near-miss: a similar fact in ANOTHER document (an LTV
   exists in the Chimney Square appraisal, but not for the Northwood loan).
7. **Size**: the target is 40 answerable + 15 traps; 30+ and 10+ is a good start. Always state the
   number of questions next to a percentage (3 misses out of 24 = 12.5%).
8. **Don't tune on everything.** If you try ten chunk sizes and keep the one that scores best on
   these questions, that score is a little optimistic. Keep ~5 questions you never look at while
   tuning, and check the winner on them at the end.

## Check the file, then run the evals

```powershell
python -m evals.golden                       # every doc_id / page / quote checked against data/fulltext
                                             # (the evals below run the same check first and stop on a problem)
python -m evals.retrieval_eval               # is the right page retrieved? (free, no LLM)
python -m evals.retrieval_eval --ablation    # same, for chunk size 400 / 800 / 1600, overlap 0 / ~20%
python -m evals.answer_eval                  # full answers + LLM judge (~2 LLM calls per question)
python -m evals.judge_agreement export       # then fill human_verdict, then: ... score
```
