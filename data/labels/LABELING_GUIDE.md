# Labeling guide (the answer key)

**Golden rule: label BEFORE you look at any model output.** If you peek, you copy its mistakes and
the accuracy number becomes meaningless.

## How
1. In VS Code, open `data/text/<doc_id>.txt` (exactly what the model will see) on the left and
   `data/labels/<doc_id>.json` on the right. If the text looks confusing, open the original in
   `data/raw/` (PDFs in any viewer, .htm files in a browser).
2. Use Ctrl+F with the search words below. Type each value into the JSON.
3. Run `python -m ragagent.labels check` to validate everything with the same Pydantic schemas.

## Format rules (same as the model's)
- Numbers: plain, exactly as printed, no `$` or commas: `1234567.89`. Brackets mean negative: `(25,202)` -> `-25202`.
- Dates: `"YYYY-MM-DD"` in quotes. All documents are American, so `9/10/2020` = month/day = `"2020-09-10"`
  (`DECISIONS.md` R2; the model gets the exact same rule in its instructions).
- Not in the document -> leave `null`. A placeholder like `XXX` or `[***]` also counts as null.
- Percent fields: the number only, `3.97` for 3.97%.
- Unsure? Write the doubt in `"_notes"`. Ambiguous fields are an eval finding too.

## Per document type
**Invoices** (search: `invoice`, `date`, `due`, `bill to`, `subtotal`, `tax`, `total`)
- `vendor_name` = the company that SENT the invoice.
- `bill_to` = the customer being billed ("Bill To", "Sold To" or "Customer name"). A company -> the company name,
  not an "attention" line or a department. An individual customer -> that person's name. NEVER the "Ship To" /
  delivery name (`DECISIONS.md` R4, refined after the first eval).
- `total` = this invoice's own charges. Do NOT include a "previous balance".
- `amount_due` = the overall "Total Due / Balance Due / Amount Due" line, if printed (may include an old balance). Not printed -> null.
- `tax` = the tax AMOUNT. If only a rate like "3%" is printed, write the amount only if it is printed; otherwise null.

**Financial statements** (search: `Three Months Ended`, `Total revenues`, `Net income`, `Funds from operations` / `FFO`, `Total assets`, `in thousands`)
- Use the MOST RECENT quarter column. If the document only has annual numbers, use the latest year and `period_months: 12`.
- `units`: look for "(in thousands...)" above the tables -> `"thousands"`.
- `net_income` = the line "attributable/available to common stockholders" if it exists; otherwise plain net income.
- `ffo` = the TOTAL FFO line (NAREIT FFO), not the per-share figure.
- `total_assets` only from a balance sheet. No balance-sheet total -> null.

**Loan agreements / notes** (search: `Borrower`, `Lender`, `promise to pay`, `per annum`, `%`, `matur`, `dated`)
- `borrower_name`: if page 1 only says "the undersigned", look at the signature page at the end.
- `principal_amount`: the loan amount in dollars (often written in words AND digits; use the digits).
- If the rate or maturity is only "as defined in the Loan Agreement" (another document) -> null, and `interest_rate_type: "unknown"`.

**Appraisals** (search: `Salient`, `Final Value`, `As Is`, `Date of Value`, `Net Operating Income`, `Capitalization Rate`, `Units`, `Beds`, `square feet`)
- Use `data/text/` (only the 4-7 summary pages), NOT `data/fulltext/` (all 150+ pages, that one is for the search part).
- What it is: an expert's report saying "this building is worth $X on date Y". The lender uses it to decide how much it is safe to lend.
- The appraiser values the building 3 ways (cost to rebuild / what similar buildings sold for / the income it makes),
  then picks ONE final number. "Salient facts" just means "the key facts".
- `property_address` = where the BUILDING is. Not the client's address on the cover, not the appraiser's office.
- `net_operating_income` (NOI) = the building's yearly income minus its running costs (staff, taxes, repairs). "Its yearly profit."
- `cap_rate_percent` = the yearly return investors expect, in %. Value = NOI / cap rate. Example: NOI $100,000 at 8% -> $1,250,000.
- `as_is_market_value` = the FINAL value conclusion, not an individual approach (cost / sales / income) value.
  "As is" = in its condition today (not "after renovation").
- `effective_date` = the date OF value, not the date the report was written.
- `property_type`: pick the closest: `multifamily`, `assisted_living`, `skilled_nursing`, `senior_housing`, `office`, `retail`, `industrial`, `land`, `mixed_use`, `other`.
- `appraiser_firm` = the company (not the person).
