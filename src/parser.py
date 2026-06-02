"""
parser.py — 10-K Section Parser
=================================
Takes raw HTML from EDGAR and extracts the meaningful text sections.
"""

import re
import json
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup # HTML parsing library

# SECTION DETECTION PATTERNS
# ---------------------------------------------------------------------------
# 10-K filings are divided into "Items" by SEC regulation. The structure is:
#   Item 1   — Business (what does the company do?)
#   Item 1A  — Risk Factors (what could go wrong?)
#   Item 7   — MD&A (management's narrative on financial results)
#   Item 7A  — Quantitative Disclosures About Market Risk
#   Item 8   — Financial Statements (the actual numbers)
# Pattern syntax:
#   \s*  = zero or more whitespace characters
#   [\.\s]+ = one or more periods or spaces (separator between number and name)
#   .{0,10} = any 0–10 characters (catches "Discussion AND Analysis" etc.)
#   (?!\s*a) = negative lookahead: "1" not followed by "a" (to avoid matching
#              "Item 1A" when looking for "Item 1")
SECTION_PATTERNS = {
    "business": [
        r"item\s*1(?![\.\s]*\d)[\.\s\-]{1,10}business",
        r"item\s*1(?![\.\s]*\d)[^\n]{0,20}\n\s{0,10}business",
    ],
    "risk_factors": [
        r"item\s*1a[\.\s]+risk\s*factors",   # "Item 1A. Risk Factors"
        r"item\s*1\s*a[\.\s]+risk\s*factors", # "Item 1 A. Risk Factors" (with space)
        r"risk\s*factors",                    # Plain "Risk Factors" (fallback)
    ],
    "mda": [
        r"item\s*7[\.\s]+management.{0,10}discussion",  # "Item 7. Management's Discussion"
        r"management.{0,10}discussion\s*and\s*analysis", # Plain "MD&A" header (fallback)
    ],
    "market_risk": [
        r"item\s*7a[\.\s]+quantitative",      # "Item 7A. Quantitative..."
        r"item\s*7\s*a[\.\s]+quantitative",   # "Item 7 A. Quantitative..."
        r"quantitative\s*and\s*qualitative\s*disclosures\s*about\s*market\s*risk",
    ],
    "financial_statements": [
        r"item\s*8[\.\s]+financial\s*statements",
        r"financial\s*statements\s*and\s*supplementary\s*data",
    ],
}

# order matters, determines where each section ENDS
SECTION_ORDER = ["business", "risk_factors", "mda", "market_risk", "financial_statements"]

# Character limits per section
SECTION_CHAR_LIMITS = {
    "business": 80_000,   # ~20 pages of text
    "risk_factors": 120_000,   # Risk sections are often very long
    "mda": 120_000,   # MD&A is the richest section for RAG
    "market_risk": 40_000,   # Shorter, more quantitative
    "financial_statements": 60_000,   # We only take the intro, not all tables
}

# ---------------------------------------------------------------------------
# HTML to Plain Text
# ---------------------------------------------------------------------------
"""
Convert raw EDGAR HTML into clean plain text

Args:
    html: Raw HTML string from EDGAR

Returns:
    Cleaned plain text string
"""
def html_to_text(html: str) -> str:
    # BeautifulSoup parses the HTML
    soup = BeautifulSoup(html, "html.parser")
    
    # filter tags
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    
    # insert newlines before block-level elements to preserve structure
    for tag in soup.find_all(["p", "div", "br", "tr"]):
        tag.insert_before("\n")
    
    # returns all remaining text, with separator between tags
    text = soup.get_text(separator=" ")
    
    # clean whitespaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)

    return text.strip()
    
# ---------------------------------------------------------------------------
# Find Section Boundaries
# ---------------------------------------------------------------------------
"""
Locate where each 10-K section starts and ends within the plain text.

DETERMINING END BOUNDARIES:
    A section ends where the next section begins. So if "business" starts
    at char 500 and "risk_factors" starts at char 8,000, then "business"
    spans chars 500–8,000.

Args:
    text: Plain text of the entire 10-K (from html_to_text)

Returns:
    Dict mapping section name → (start_char, end_char)
"""
def find_section_boundaries(text: str) -> dict[str, tuple[int, int]]:
    TOC_SKIP_CHARS = len(text) // 20  # 5%
    text_lower = text.lower()
    found: dict[str, int] = {}

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text_lower, re.DOTALL):
                pos = m.start()
                if pos < TOC_SKIP_CHARS:
                    continue
                # take the EARLIEST match after the TOC zone
                if section not in found or pos < found[section]:
                    found[section] = pos
    
    # enforce correct section ordering — a section can't start after
    # a later section's detected position
    for i, section in enumerate(SECTION_ORDER):
        if section not in found:
            continue
        for j in range(i + 1, len(SECTION_ORDER)):
            next_sec = SECTION_ORDER[j]
            if next_sec in found and found[next_sec] <= found[section]:
                del found[next_sec]  # detected out of order — drop it

    # build (start, end) pairs
    boundaries: dict[str, tuple[int, int]] = {}
    for i, section in enumerate(SECTION_ORDER):
        if section not in found:
            continue
        start = found[section]
        end = len(text)
        for j in range(i + 1, len(SECTION_ORDER)):
            next_sec = SECTION_ORDER[j]
            if next_sec in found and found[next_sec] > start:
                end = found[next_sec]
                break
        boundaries[section] = (start, end)

    return boundaries

# ---------------------------------------------------------------------------
# Extract Section Text
# ---------------------------------------------------------------------------
"""
Slice the plain text into per-section strings and apply quality filters.

CHAR LIMITS:
    We apply per-section character limits (defined in SECTION_CHAR_LIMITS)
    to prevent pathologically large sections from blowing up memory or
    producing thousands of chunks. Financial statements especially can be
    enormous, we cap them at 60K chars

MINIMUM LENGTH FILTER:
    We discard sections shorter than 200 chars. A section that short is
    almost certainly a parsing artifact (e.g., the regex matched a header
    but the actual content is in an unrecognized format)

Args:
    text: Plain text of the entire 10-K

Returns:
    Dict mapping section name → section text (only sections that passed
    the quality filter)
"""
def extract_sections(text: str) -> dict[str, str]:
    boundaries = find_section_boundaries(text)
    sections: dict[str, str] = {}

    for section, (start, end) in boundaries.items():
        limit = SECTION_CHAR_LIMITS.get(section, 80_000)
        
        # min(end - start, limit) prevents from exceeding actual or artifical limit
        content = text[start : start + min(end - start, limit)].strip()
        
        # skip short sections
        if len(content) > 200:
            sections[section] = content

    return sections
    
# ---------------------------------------------------------------------------
# Parse a single filing
# ---------------------------------------------------------------------------
"""
Full parse pipeline for one 10-K file

Handles two input formats:
    - HTML
    - Plain text (older EDGAR filings, or pre-processed files)

Args:
    raw_path:     Path to the saved HTML file (from edgar_client.py)
    company_name: Human-readable company name (for metadata)
    filing_date:  Filing date string e.g. "2024-02-07" (for metadata)

Returns:
    Structured dict with company info + extracted sections, or None on failure
"""
def parse_filing(raw_path: Path, company_name: str, filing_date: str) -> Optional[dict]:
    # errors="ignore" silently drops any bytes that aren't valid UTF-8
    # SEC filings occasionally contain encoding artifacts from old systems  
    try:
        raw = raw_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"ERROR could not read {raw_path}: {e}")
        return None
    
    # detect HTML vs plain text by checking for common HTML tags
    if "<html" in raw.lower() or "<body" in raw.lower() or "<div" in raw.lower():
        text = html_to_text(raw)
    else:
        text = raw
    
    sections = extract_sections(text)
    if not sections:
        print(f"WARNING no sections found in {raw_path.name}")
    
    found_sections = list(sections.keys())
    total_chars = sum(len(value) for value in sections.values())

    print(f"Sections found: {found_sections}")
    print(f"Total extracted: {total_chars:,} chars")

    return {
        "company": company_name,
        "filing_date": filing_date,
        "source_file": str(raw_path),
        "sections": sections,
        "sections_found": found_sections,
        "total_chars": total_chars
    }
    
# ---------------------------------------------------------------------------
# Parse all filings from manifest
# ---------------------------------------------------------------------------
"""
Parse every filing listed in the EDGAR manifest.json

The manifest is the output of edgar_client.run_pipeline()

We save each parsed result as its own JSON file (one per company)
The chunker (chunker.py) reads these JSON files as its input

Args:
    manifest_path: Path to manifest.json (output of edgar_client.py)
    output_dir:    Where to save per-company parsed JSON files

Returns:
    List of parsed filing dicts (same as what parse_filing returns)
"""
def parse_all_from_manifest(manifest_path: str, output_dir: str = "data/parsed") -> list[dict]:
    manifest_path = Path(manifest_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    results = []

    for company_name, meta in manifest.items():
        print(f"\n{'='*60}")
        print(f"PARSING {company_name}")

        raw_path = Path(meta["local_file"])
        if not raw_path.exists():
            print(f"SKIP file not found: {raw_path}")
            continue
        
        parsed = parse_filing(raw_path, company_name, meta["filing_date"])
        if not parsed:
            continue
        
        # Save to disk as JSON 
        safe_name = company_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = out / f"{safe_name}_parsed.json"
        out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
        print(f"SAVED {out_path.name}")
        results.append(parsed)
    
    print(f"\nPARSED {len(results)}/{len(manifest)} filings")
    return results
    
# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parse_all_from_manifest("data/raw/manifest.json")