"""
edgar_client.py — EDGAR API Client
====================================
Fetches SEC 10-K annual report filings from EDGAR (Electronic Data Gathering,
Analysis, and Retrieval) — the SEC's public database of all company filings.
"""

import requests
import time
import json
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------------------------------------------------------
# EDGAR API AUTHENTICATION
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": os.getenv("EDGAR_USER_AGENT", "Your Name your@email.com"),
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# Base URLs for EDGAR's two different servers:
EDGAR_BASE = "https://data.sec.gov"
SEC_BASE   = "https://www.sec.gov"

# ---------------------------------------------------------------------------
# TARGET COMPANIES
# ---------------------------------------------------------------------------
# CIK numbers are the stable IDs for each company

COMPANIES = {
    "JPMorgan Chase":  "0000019617",
    "PayPal":          "0001633917",
    "Coinbase":        "0001679788",
    "Lemonade":        "0001691936",
    "Block (Square)":  "0001512673",
    "Goldman Sachs":   "0000886982",
    "Visa":            "0001403161",
}

# ---------------------------------------------------------------------------
# Get filing metadata (WHERE is the 10-K?)
# ---------------------------------------------------------------------------
""" 
Most recent 10-K filing for a company.

Finds metadata:
when it was filed, what the accession number is, and what the main
document filename is. We use that info in fetch_10k_text() to download

Args:
    cik:          Company CIK, e.g. "0001633917" (PayPal)
    company_name: Human-readable name, used for logging only

Returns:
    dict with filing metadata, or None if not found / request failed
"""
def get_recent_10k_filing(cik: str, company_name: str) -> Optional[dict]:
    #EDGAR requires CIKs to be exactly 10 digits, zero-padded
    # cik_padded = cik.lstrip("0".zfill(10))
    cik_padded = cik.zfill(10)

    # submissions endpoint returns the full filing history as JSON
    # example: https://data.sec.gov/submissions/CIK0001633917.json
    url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"

    try:
        resp = requests.get(url, headers={**HEADERS, "Host": "data.sec.gov"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"ERROR Failed to fetch submissions for {company_name}: {e}")
        return None
    
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])

    # scan through the forms array to find the first "10-K".
    # EDGAR returns filings in reverse chronological order, so the FIRST
    for i, form in enumerate(forms):
        if form == "10-K":
            # remove accession numbers dashes
            accession_dashed = accessions[i]
            accession_flat = accessions[i].replace("-", "")
            filing_date = dates[i]
            primary_doc = primary_docs[i]

            # direct URL to the filing document
            # URL pattern: /Archives/edgar/full-index/data/{CIK}/{ACCESSION}/{FILENAME}
            cik_short = cik.lstrip("0") # get rid of leading zeros
            doc_url = (
                f"{SEC_BASE}/Archives/edgar/full-index"
                f"data/{cik_short}/{accession_flat}/{primary_doc}"
            )

            return {
                "company": company_name,
                "cik": cik,
                "accession_number": accession_dashed,
                "accession_flat": accession_flat,
                "filing_date": filing_date,
                "primary_doc": primary_doc,
                "doc_url": doc_url,
            }
    
    print(f"WARNING no 10-k found for {company_name}")
    return None

# ---------------------------------------------------------------------------
# Download filing document
# ---------------------------------------------------------------------------
"""
Download the raw HTML content of the 10-K filing.

Fallback strategy:
- Try the primary_doc URL directly.
- If that fails or is too small, fetch the filing's INDEX page
            (a directory listing) and try each .htm file we find there.

The 10,000 character minimum check is because sometimes the primary doc
is a tiny XBRL wrapper — real 10-Ks are hundreds of thousands of chars.

Returns:
    Raw HTML string, or None if all attempts failed
"""
def fetch_10k_text(filing_meta: dict) -> Optional[str]:
    company = filing_meta["company"]
    cik = filing_meta["cik"].zfill(10)
    accession = filing_meta["accession_flat"]
    primary_doc = filing_meta["primary_doc"]

    # both www.sec.gov endpoints need a different Host header than data.sec.gov
    headers = {**HEADERS, "Host": "www.sec.gov"}

    # --- Direct URL ---
    doc_url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
    try:
        resp = requests.get(doc_url, headers=headers, timeout=30)
        # Check both: successful status AND the file is actually big enough (not a stub/wrapper document)
        if resp.status_code == 200 and len(resp.text) > 10_000:
            return resp.text
    except Exception:
        pass

    # --- fallback scrape the filing index directory --- 
    index_url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession}/"
    try:
        resp = requests.get(index_url, headers=headers, timeout=15)
        resp.raise_for_status()

        # parse the directory listing HTML to find .htm links.
        # re.findall returns all matches — we'll try each one.
        # .htm and .html (hence htm[l]? — the 'l' is optional)
        links = re.findall(r'href="([^"]+\.htm[l]?)"', resp.text, re.IGNORECASE)

        # if no .htm files, fall back to .txt (older filings use plain text)
        if not links:
            links = re.findall(r'href="([^"]+\.txt)"', resp.text, re.IGNORECASE)

        # try each link until we get a real document
        for link in links:
            # links might be relative ("/Archives/...") or absolute ("https://...")
            full_url = link if link.startswith("http") else f"{SEC_BASE}{link}"
            try:
                doc_resp = requests.get(full_url, headers=headers, timeout=30)
                if doc_resp.status_code == 200 and len(doc_resp.text) > 10_000:
                    return doc_resp.text
            except Exception:
                continue 

    except Exception as e:
        print(f"ERROR index fetch failed for {company}: {e}")

    return None
    
# ---------------------------------------------------------------------------
# Fetch all companies
# ---------------------------------------------------------------------------

"""
Main pipeline: loop over all COMPANIES, fetch their 10-Ks, save to disk.

RATE LIMITING:
The SEC allows max 10 requests/second, if you hammer EDGAR, they'll ban your IP.

THE MANIFEST:
We save a manifest.json that maps company name → filing metadata + local
file path. This is the "contract" between Day 1 and Day 2 — the parser
reads the manifest to know which files to process.

Args:
    output_dir:      Where to save raw HTML files and manifest.json
    rate_limit_sec:  Seconds to sleep between API requests

Returns:
    The manifest dict (also saved to disk as manifest.json)
"""
def run_pipeline(output_dir: str = "data/raw", rate_limit_sec: float = 0.5) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    
    for company_name, cik in COMPANIES.items():
        print(f"\n{'='*60}") 
        print(f"Processing: {company_name} (CIK: {cik})")

        # Step 1: get metadata
        meta = get_recent_10k_filing(cik, company_name)
        time.sleep(rate_limit_sec) # respect SEC rate limits between requests

        if not meta:
            continue # skip company
        
        print(f"Found 10-k filed: {meta['filing_date']}")
        print(f"Accession: {meta['accession_number']}")
        # Step 2: download document
        print(f" Downloading document...")
        raw_text = fetch_10k_text(meta)
        time.sleep(rate_limit_sec)  # rate limit after the document download too

        if not raw_text:
            print(f"SKIP could not download document for {company_name}")
            continue
        
        # Step 3: save raw HTML to disk
        safe_name = company_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = out / f"{safe_name}_10k_{meta['filing_date']}.html"
        out_path.write_text(raw_text, encoding="utf-8")
        size_kb = len(raw_text) / 1024
        print(f"SAVED {out_path.name} ({size_kb:.0f} KB)")

        # merge the metadata dict with local file info
        manifest[company_name] = {
            **meta,
            "local_file": str(out_path),
            "size_bytes": len(raw_text),
        }

    # save manifest
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*60}")
    print(f"FETCHED {len(manifest)}/{len(COMPANIES)} filings.")
    print(f"SAVED {manifest_path}")

    return manifest

# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__== "__main__":
    run_pipeline()
