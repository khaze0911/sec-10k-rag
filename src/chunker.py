"""
chunker.py — Text Chunker
===========================
Splits parsed 10-K sections into small, overlapping chunks ready for embedding
"""

import json
import re
import uuid # For generating unique chunk IDs
from pathlib import Path

# ---------------------------------------------------------------------------
# CHUNKING PARAMETERS
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 800 # target max characters per chunk
DEFAULT_OVERLAP = 150 # overlap between adjacent chunks
MIN_CHUNK_SIZE = 100

# ---------------------------------------------------------------------------
# Split text into sentences
# ---------------------------------------------------------------------------
"""
Split a block of text into individual sentences

PLACEHOLDER SUBSTITUTION:
    Step 1: Replace periods in known abbreviations with "__DOT__"
            "U.S." → "U__DOT__S__DOT__"
    Step 2: Run the sentence splitter (now "U__DOT__S__DOT__" won't
            trigger a false split)
    Step 3: Restore "__DOT__" → "." in the output sentences

SPLIT PATTERN:
    We split on: period/question/exclamation followed by whitespace
    followed by a capital letter. The (?<=[.!?]) and (?=[A-Z]) are
    "lookbehind" and "lookahead" — they match positions WITHOUT consuming
    the characters, so the period stays with the sentence before it

Args:
    text: A block of text (one section of a 10-K)

Returns:
    List of sentence strings
"""
def split_into_sentances(text: str) -> list[str]:
    # abbreviations to protect from false sentence splits
    abbreviations = [
        r"U\.S\.", r"U\.K\.", r"Inc\.", r"Corp\.", r"Ltd\.", r"Co\.",
        r"No\.",   r"Sec\.",  r"Dept\.", r"i\.e\.", r"e\.g\.", r"vs\.",
        r"approx\.", r"est\.",
        r"Jan\.", r"Feb\.", r"Mar\.", r"Apr\.", r"Jun\.", r"Jul\.",
        r"Aug\.", r"Sep\.", r"Oct\.", r"Nov\.", r"Dec\.",
    ]
    # replace periods in abbreviations with placeholder
    # re.subn returns (new_string, number_of_replacements)
    protected = text
    for abbr in abbreviations:
        safe = abbr.replace("\\.", "__DOT__")
        protected, _ = re.subn(abbr, safe, protected)
    # split pattern:
    #   (?<=[.!?])   = lookbehind: preceded by sentence-ending punctuation
    #   \s+          = one or more whitespace characters (the space between sentences)
    #   (?=[A-Z])    = lookahead: followed by a capital letter (new sentence start)
    #
    # second alternative (?<=\n)\n+ splits on paragraph breaks
    # (double newlines from the HTML parser) — these are always safe to split on
    sentence_pattern = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+'
    )
    raw_sentences = sentence_pattern.split(protected)

    # restore placeholders and clean up
    sentences = []
    for s in raw_sentences:
        s = s.replace("__DOT__", ".") # restore abbreviation periods
        s = s.strip() # remove leading/trailing whitespace
        if s:
            sentences.append(s)
    
    return sentences

# ---------------------------------------------------------------------------
# Pack sentences into chunks with overlap
# ---------------------------------------------------------------------------
"""
Pack sentences into fixed-size chunks with overlap between chunks.

Args:
    text:       The section text to chunk
    chunk_size: Target max characters per chunk
    overlap:    Characters of overlap between consecutive chunks
    min_size:   Minimum chunk size — shorter chunks are discarded

Returns:
    List of chunk strings
"""
def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP, min_size: int = MIN_CHUNK_SIZE) -> list[str]:
    sentences = split_into_sentances(text)
    chunks = []
    current_chunk: list[str] = [] # sentences in chunk
    current_len: int = 0 # total chars in chunk

    for sentence in sentences:
        sentence_len = len(sentence)

        # edge case: sentence longer than chunk size
        if sentence_len > chunk_size:
            # flush
            if current_chunk:
                flushed = " ".join(current_chunk).strip()
                if len(flushed) >= min_size:
                    chunks.append(flushed)
                current_chunk = []
                current_len = 0
            # hard-split the long sentence using a sliding window
            for i in range(0, sentence_len, chunk_size - overlap):
                part = sentence[i : i + chunk_size].strip()
                if len(part) >= min_size:
                    chunks.append(part)
            continue

        # normal case
        if current_len + sentence_len + 1 > chunk_size and current_chunk:
            # flush
            flushed = " ".join(current_chunk).strip()
            if len(flushed) >= min_size:
                chunks.append(flushed)
        
            # walk backwards until overlap chars
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) + 1 <= overlap:
                    overlap_sentences.insert(0, s) # prepend to maintain order
                    overlap_len += len(s) + 1
                else:
                    break
            
            current_chunk = overlap_sentences
            current_len = overlap
    
        # add sentence to chunk
        current_chunk.append(sentence)
        current_len += sentence_len + 1
    
    # flush final chunk
    if current_chunk:
        final = " ".join(current_chunk).strip()
        if len(final) >= min_size:
            chunks.append(final)
        
    return chunks

# ---------------------------------------------------------------------------
# Add Metadata to Each Chunk
# ---------------------------------------------------------------------------
"""
Chunk all sections of a parsed 10-K and attach metadata to each chunk.

THE chunk_id FIELD:
    uuid.uuid4() generates a random universally unique identifier like:
    "550e8400-e29b-41d4-a716-446655440000"
    This becomes the primary key when we insert into PostgreSQL (pgvector).
    UUIDs are preferred over sequential integers for distributed systems
    (no risk of collision across multiple processes or machines).

Args:
    parsed:     Output of parser.parse_filing() — dict with sections
    chunk_size: Passed through to chunk_text()
    overlap:    Passed through to chunk_text()

Returns:
    List of chunk dicts, each with text + metadata
"""
def chunk_parsed_filing( parsed: dict, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[dict]:
    company = parsed["company"]
    filing_date = parsed["filing_date"]
    all_chunks  = []

    for section_name, section_text in parsed["sections"].items():
        raw_chunks = chunk_text(section_text, chunk_size=chunk_size, overlap=overlap)
        total = len(raw_chunks)  # Total chunks in section
        for i, chunk_str in enumerate(raw_chunks):
            chunk = {
                # Primary key for the database — unique per chunk
                "chunk_id": str(uuid.uuid4()),

                # Who and when — enables metadata filtering at query time
                "company": company,
                "filing_date": filing_date,

                # Which part of the 10-K this came from
                "section": section_name,

                # Position within the section
                "chunk_index":  i,
                "total_chunks": total,

                # The actual text that gets embedded and stored
                "text": chunk_str,

                # Convenience field
                "char_count": len(chunk_str),
            }
            all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# Chunk all parsed filings
# ---------------------------------------------------------------------------
"""
Chunk every parsed 10-K JSON and write output to JSONL files.

OUTPUT FORMAT — JSONL:

TWO OUTPUT FILES:
    1. Per-company files: coinbase_chunks.jsonl, paypal_chunks.jsonl, etc.
        → Useful for loading/re-loading a single company
    2. all_chunks.jsonl: every chunk from every company
        → This is the input to Day 2's embedding + pgvector ingest

Args:
    parsed_dir:  Directory containing *_parsed.json files
    output_dir:  Where to write JSONL chunk files
    chunk_size:  Passed through to chunk_text()
    overlap:     Passed through to chunk_text()

Returns:
    Summary stats dict (also saved as chunk_stats.json)
"""
def chunk_all_from_parsed_dir( parsed_dir: str = "data/parsed", output_dir: str = "data/chunks", chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> dict:
    parsed_path = Path(parsed_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_chunks_path = out / "all_chunks.jsonl"
    stats = {}
    total_chunks = 0

    with open(all_chunks_path, "w", encoding="utf-8") as all_f:
        # glob("*_parsed.json") finds all parsed company files
        # sorted() ensures consistent ordering across runs
        for parsed_file in sorted(parsed_path.glob("*_parsed.json")):
            print(f"\nChunking: {parsed_file.name}")
            parsed = json.loads(parsed_file.read_text())
            company = parsed["company"]

            chunks = chunk_parsed_filing(parsed, chunk_size=chunk_size, overlap=overlap)
            # --- Write per-company JSONL ---
            safe_name = company.lower().replace(" ", "_").replace("(", "").replace(")", "")
            company_out = out / f"{safe_name}_chunks.jsonl"
            with open(company_out, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    # json.dumps converts the dict to a JSON string
                    # "\n" at the end is the JSONL record separator
                    f.write(json.dumps(chunk) + "\n")

            # --- Write to combined file ---
            for chunk in chunks:
                all_f.write(json.dumps(chunk) + "\n")

            # --- Collect stats ---
            section_counts: dict[str, int] = {}
            for c in chunks:
                section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1

            stats[company] = {
                "total_chunks": len(chunks),
                "by_section":   section_counts,
                "filing_date":  parsed["filing_date"],
            }
            total_chunks += len(chunks)

            print(f"  {len(chunks)} chunks → {company_out.name}")
            for sec, cnt in section_counts.items():
                print(f"    {sec}: {cnt} chunks")

    print(f"\n{'='*60}")
    print(f"Total chunks across all companies: {total_chunks}")
    print(f"Combined JSONL: {all_chunks_path}")

    # Save stats JSON — useful for monitoring and README writeup
    stats_path = out / "chunk_stats.json"
    stats_path.write_text(
        json.dumps({"total_chunks": total_chunks, "companies": stats}, indent=2)
    )
    print(f"Stats: {stats_path}")

    return stats

# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chunk_all_from_parsed_dir()