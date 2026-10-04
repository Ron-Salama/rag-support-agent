"""Re-download the original documents (data/raw/) from the source URLs in data/manifest.csv.

    python -m ragagent.download              # fetch every document that is missing from data/raw/
    python -m ragagent.download --dry-run    # only list what would happen (no network, no files written)
    python -m ragagent.download --selftest   # check the rules with a fake server (no network)

Why this exists: data/raw/ holds other people's files (~16 MB of PDFs and SEC filings). Because
manifest.csv keeps the source URL of every document, anyone can rebuild data/raw/ with this
script - so the git repo does not HAVE to carry those files (still open: DECISIONS.md C20).
The text made from them (data/text/, data/fulltext/) is already in the repo; run
`python -m ragagent.to_text` (and `--full`) only if you want to remake it from fresh downloads.

SEC fair-access rule ("Accessing EDGAR Data" on sec.gov:
https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data):
  1. Say who you are: every automated request needs a User-Agent header with a name (or company)
     AND a contact email, e.g. "Jane Doe jane.doe@example.com". Without one, SEC answers 403 Forbidden.
  2. Stay under 10 requests per second.
The header comes from the environment variable SEC_USER_AGENT - put a line in .env:
    SEC_USER_AGENT=Jane Doe jane.doe@example.com
or set it for one PowerShell window:  $env:SEC_USER_AGENT = "Jane Doe jane.doe@example.com"
It is never written in the code: whoever runs the script declares their OWN identity.
It is sent to sec.gov ONLY: the other sites (GitHub, a university, a city) get GENERIC_AGENT,
because they don't ask for a name and email, so there is no reason to hand them yours.
We also wait DELAY_SECONDS before every download, far below SEC's limit.

Safe to re-run: files that already exist are skipped (no request at all). A download is written
to "<file>.part" first and renamed only when it is complete, so a crash never leaves a half file
that the next run would skip as "already there".
"""
import http.client
import os
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from ragagent.config import RAW   # importing config also loads .env (so SEC_USER_AGENT can live there)
from ragagent.labels import manifest_rows

DELAY_SECONDS = 1.0     # pause before each download (SEC allows 10 requests/second; we make 1)
TIMEOUT_SECONDS = 60    # give up on a server that stops answering
GENERIC_AGENT = "rag-support-agent document downloader (Python urllib)"   # for non-SEC sites


def is_sec(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == "sec.gov" or host.endswith(".sec.gov")


def user_agent_for(url: str, sec_agent: str | None) -> str | None:
    """The User-Agent to send. None = not allowed: an SEC URL, but SEC_USER_AGENT is missing or has no email.

    Your name + email go to sec.gov only; every other site gets the generic one.
    """
    if is_sec(url):
        return sec_agent if sec_agent and "@" in sec_agent else None
    return GENERIC_AGENT


def plan(row: dict, raw_dir: Path, sec_agent: str | None) -> str:
    """What to do with one document: 'exists' (skip it) | 'download' | 'no_agent' (can't ask SEC politely)."""
    if (raw_dir / row["file"]).exists():
        return "exists"
    return "download" if user_agent_for(row["source_url"], sec_agent) else "no_agent"


def fetch(url: str, user_agent: str) -> bytes:
    """GET one URL -> its bytes. Raises on 403/404 (HTTPError), no network (URLError) or a timeout."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def check_content(file_name: str, data: bytes) -> str | None:
    """A cheap sanity check on what came back: a problem, or None if it looks fine.

    Some servers answer "200 OK" with an HTML error page ("rate limit exceeded"). Saved as a .pdf,
    that would only show up much later, as a document with no text in it.
    """
    if not data:
        return "the server sent an empty file"
    if file_name.lower().endswith(".pdf") and not data.startswith(b"%PDF"):
        return "not a PDF (the server probably sent an error page instead)"
    return None


def download_one(row: dict, raw_dir: Path, sec_agent: str | None, fetch_fn=fetch, sleep_fn=time.sleep) -> str:
    """One manifest row -> 'exists' | 'downloaded' | 'error: <why>'. Never raises for a network problem."""
    todo = plan(row, raw_dir, sec_agent)
    if todo == "exists":
        return "exists"
    if todo == "no_agent":
        return "error: SEC_USER_AGENT is not set or has no email (see the SEC rule at the top of download.py)"
    url = row["source_url"]
    sleep_fn(DELAY_SECONDS)   # be polite: one request at a time, with a pause
    try:
        data = fetch_fn(url, user_agent_for(url, sec_agent))
    except (OSError, http.client.HTTPException) as e:   # HTTPError, URLError and timeouts are all OSErrors
        hint = " - SEC wants SEC_USER_AGENT='Name email@example.com'" if is_sec(url) and getattr(e, "code", 0) == 403 else ""
        return f"error: {e}{hint}"
    problem = check_content(row["file"], data)
    if problem:
        return f"error: {problem}"
    path = raw_dir / row["file"]
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)   # a rename is all-or-nothing: the real file name only ever holds a complete file
    return "downloaded"


def download_all(rows: list[dict], raw_dir: Path, sec_agent: str | None,
                 fetch_fn=fetch, sleep_fn=time.sleep, log=print) -> dict[str, str]:
    """Every manifest row -> its outcome, logged as we go. One failed document does not stop the others."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for row in rows:
        outcomes[row["doc_id"]] = download_one(row, raw_dir, sec_agent, fetch_fn, sleep_fn)
        log(f"{row['doc_id']:<24} {outcomes[row['doc_id']]}")
    return outcomes


def dry_run(rows: list[dict], raw_dir: Path, sec_agent: str | None, log=print) -> dict[str, str]:
    """Show what download_all WOULD do. Sends no request and creates no file or folder."""
    plans = {}
    for row in rows:
        todo = plan(row, raw_dir, sec_agent)
        plans[row["doc_id"]] = todo
        if todo == "exists":
            what = f"exists ({(raw_dir / row['file']).stat().st_size:,} bytes) - skip"
        elif todo == "download":
            what = f"would download {row['source_url']}"
        else:
            what = "would FAIL: sec.gov URL, but SEC_USER_AGENT is not set (or has no email)"
        log(f"{row['doc_id']:<24} {what}")
    return plans


# ----------------------------------------------------------------- self-test with a fake server (no network)
class FakeServer:
    """Stands in for fetch(): answers from a dict {url: bytes or an exception}, records every request."""

    def __init__(self, pages: dict):
        self.pages, self.requests = pages, []

    def fetch(self, url: str, user_agent: str) -> bytes:
        self.requests.append((url, user_agent))
        page = self.pages[url]
        if isinstance(page, Exception):   # a scripted failure, e.g. no internet
            raise page
        return page


def selftest():
    import tempfile
    import urllib.error

    sec_url, pdf_url = "https://www.sec.gov/Archives/x.htm", "https://example.org/a.pdf"
    rows = [{"doc_id": "sec_doc", "file": "sec_doc.htm", "source_url": sec_url},
            {"doc_id": "pdf_doc", "file": "pdf_doc.pdf", "source_url": pdf_url}]
    agent = "Jane Doe jane.doe@example.com"
    good = {sec_url: b"<html>filing</html>", pdf_url: b"%PDF-1.7 ..."}
    quiet = lambda line: None   # noqa: E731 - the self-test only checks results, it doesn't print progress

    def run(pages: dict, sec_agent: str | None, existing: tuple = ()) -> tuple:
        """Download both rows into a fresh temp folder -> (outcomes, files left there, the server, the pauses)."""
        server, pauses = FakeServer(pages), []
        with tempfile.TemporaryDirectory() as tmp:
            for name in existing:
                (Path(tmp) / name).write_bytes(b"old copy")
            outcomes = download_all(rows, Path(tmp), sec_agent, server.fetch, pauses.append, quiet)
            return outcomes, sorted(p.name for p in Path(tmp).iterdir()), server, pauses

    both = run(good, agent)
    skip = run(good, agent, existing=("sec_doc.htm",))
    no_agent = run(good, None)
    no_email = run(good, "Jane Doe")
    error_page = run({**good, pdf_url: b"<html>Too many requests</html>"}, agent)
    offline = run({**good, sec_url: urllib.error.URLError("no route to host")}, agent)
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "raw"
        plans = dry_run(rows, missing, agent, quiet)
        plans_no_agent = dry_run(rows, missing, None, quiet)
        dry_made_folder = missing.exists()
    manifest = manifest_rows()

    cases = [
        ("missing files downloaded, no .part left", both[0] == {"sec_doc": "downloaded", "pdf_doc": "downloaded"}
         and both[1] == ["pdf_doc.pdf", "sec_doc.htm"]),
        ("SEC URL is sent SEC_USER_AGENT", both[2].requests[0] == (sec_url, agent)),
        ("other sites never get your SEC name + email", both[2].requests[1] == (pdf_url, GENERIC_AGENT)),
        ("existing file skipped: no request for it", skip[0]["sec_doc"] == "exists"
         and skip[2].requests == [(pdf_url, GENERIC_AGENT)]),
        ("one pause before each request, none for skips", both[3] == [DELAY_SECONDS] * 2 and skip[3] == [DELAY_SECONDS]),
        ("SEC URL without SEC_USER_AGENT -> refused", no_agent[0]["sec_doc"].startswith("error: SEC_USER_AGENT")
         and sec_url not in [url for url, _ in no_agent[2].requests]),
        ("SEC_USER_AGENT without an email -> refused", no_email[0]["sec_doc"].startswith("error: SEC_USER_AGENT")),
        ("other sites work without SEC_USER_AGENT", no_agent[0]["pdf_doc"] == "downloaded"
         and no_agent[2].requests == [(pdf_url, GENERIC_AGENT)]),
        ("error page saved as .pdf -> rejected, no file", "not a PDF" in error_page[0]["pdf_doc"]
         and error_page[1] == ["sec_doc.htm"]),
        ("network error -> reported, next document runs", offline[0]["sec_doc"].startswith("error:")
         and offline[0]["pdf_doc"] == "downloaded" and offline[1] == ["pdf_doc.pdf"]),
        ("dry run: plans only, no folder created", plans == {"sec_doc": "download", "pdf_doc": "download"}
         and not dry_made_folder),
        ("dry run without agent: SEC row flagged", plans_no_agent == {"sec_doc": "no_agent", "pdf_doc": "download"}),
        ("real manifest: unique file names, http(s) URLs", len({r["file"] for r in manifest}) == len(manifest)
         and all(urlparse(r["source_url"]).scheme in ("http", "https") for r in manifest)),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in cases)}/{len(cases)} passed")


def main(dry: bool) -> bool:
    rows, sec_agent = manifest_rows(), os.getenv("SEC_USER_AGENT")
    print(f"SEC_USER_AGENT: {'set' if sec_agent else 'NOT SET (needed for the sec.gov documents)'}\n")
    if dry:
        dry_run(rows, RAW, sec_agent)
        return True
    outcomes = list(download_all(rows, RAW, sec_agent).values())
    errors = sum(o.startswith("error") for o in outcomes)
    print(f"\n{outcomes.count('downloaded')} downloaded, {outcomes.count('exists')} already there, "
          f"{errors} failed  ->  {RAW}")
    return errors == 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif args in ([], ["--dry-run"]):
        sys.exit(0 if main(dry=args == ["--dry-run"]) else 1)
    else:
        print(__doc__)
