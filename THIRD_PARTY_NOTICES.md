# Third-party notices

This repository contains text extracted from third-party documents (`data/text/`, `data/fulltext/`)
and short excerpts of that text in eval and output files (`evals/golden_draft.jsonl`,
`evals/judge_calibration.csv`, `evals/results/`, `outputs/predictions/`). The source URL of every
document is in [`data/manifest.csv`](data/manifest.csv). The original files are not committed.

## Data notice

The documents belong to their authors and filers. They were obtained from public sources (SEC EDGAR
filings, a US bankruptcy-court exhibit in a university archive, a city's public meeting packet, and
open-source repositories) and are reproduced here as extracted text, unchanged in content, for
non-commercial research and demonstration. Some carry their authors' own use conditions; the
appraisals, for example, state that possession of the report does not carry the right of
publication. Rights holders can open an issue on this repository to have a document removed.

Two of the sample invoices come from open-source repositories under the MIT License. Their licence
texts follow.

## Azure-Samples/cognitive-services-REST-api-samples (Microsoft Corporation)

Source: https://github.com/Azure-Samples/cognitive-services-REST-api-samples (`LICENSE.md`)

Covers the fictional "Contoso" sample invoices:
- `inv_contoso_100`: `data/text/inv_contoso_100.txt`, `data/fulltext/inv_contoso_100.txt`
  (from `curl/form-recognizer/sample-invoice.pdf`)
- `inv_contoso_stmt6`: `data/text/inv_contoso_stmt6.txt`, `data/fulltext/inv_contoso_stmt6.txt`
  (from `curl/form-recognizer/Invoice-6.pdf`)

```
MIT License

Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE

Third Party Programs: The software may include third party programs that Microsoft,
not the third party, licenses to you under this agreement. Notices, if any, for the
third party programs are included for your information only.
```

## invoice-x/invoice2data (Manuel Riel)

Source: https://github.com/invoice-x/invoice2data (`LICENSE.md`)

Covers the fictitious "Sammy Maystone" test invoice:
- `inv_sammy`: `data/text/inv_sammy.txt`, `data/fulltext/inv_sammy.txt`
  (from `tests/compare/SammyMaystoneLinesTest.pdf`)

```
The MIT License (MIT)

Copyright (c) 2015 Manuel Riel

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
