#!/usr/bin/env python3
"""
SEC 10-K RAG pipeline
Executes all three steps:
  1. Fetch 10-K filings from EDGAR
  2. Parse and extract sections
  3. Chunk sections for embedding

Usage:
    python pipeline.py                     # Full pipeline
    python pipeline.py --step fetch        # Only fetch from EDGAR
    python pipeline.py --step parse        # Only parse (requires fetched data)
    python pipeline.py --step chunk        # Only chunk (requires parsed data)
    python pipeline.py --chunk-size 800    # Custom chunk size (default: 800 chars)
    python pipeline.py --skip-fetch        # Skip fetch if data already downloaded
    python pipeline.py --step ingest       # Only ingest (requires chunked data)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.edgar_client import run_pipeline as fetch_filings, COMPANIES
from src.parser import parse_all_from_manifest
from src.chunker import chunk_all_from_parsed_dir


def print_banner(step: str):
    print(f"\n{'#'*60}")
    print(f"# {step}")
    print(f"{'#'*60}")


def step_fetch(args):
    print_banner("FETCHING 10-K filings from EDGAR")
    print(f"Companies: {list(COMPANIES.keys())}")
    manifest = fetch_filings(output_dir="data/raw")
    return len(manifest)


def step_parse(args):
    print_banner("PARSING 10-K sections")
    manifest_path = "data/raw/manifest.json"
    if not Path(manifest_path).exists():
        print(f"[ERROR] Manifest not found at {manifest_path}")
        print("Run --step fetch first")
        sys.exit(1)
    results = parse_all_from_manifest(manifest_path, output_dir="data/parsed")
    return len(results)


def step_chunk(args):
    print_banner("CHUNKING sections for embedding")
    parsed_dir = Path("data/parsed")
    if not any(parsed_dir.glob("*_parsed.json")):
        print(f"ERROR no parsed files found in {parsed_dir}")
        print("RUN --step parse first")
        sys.exit(1)

    stats = chunk_all_from_parsed_dir(
        parsed_dir="data/parsed",
        output_dir="data/chunks",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    return stats

def step_ingest(args):
    print_banner("INGESTING chunks into pgvector")
    chunks_path = Path("data/chunks/all_chunks.jsonl")
    if not chunks_path.exists():
        print(f"ERROR Chunks file not found at {chunks_path}")
        print("Run --step chunk first")
        sys.exit(1)

    from src.ingest import run_ingest
    result = run_ingest()
    return result["chunks_ingested"]

def print_summary(fetch_count, parse_count, chunk_stats, ingest_count):
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Filings fetched:  {fetch_count}")
    print(f"Filings parsed:   {parse_count}")
    print(f"Chunks ingested:  {ingest_count}")

    if chunk_stats:
        total = sum(v["total_chunks"] for v in chunk_stats.values())
        print(f"Total chunks: {total}")
        print()
        print("Per company:")
        for company, info in chunk_stats.items():
            print(f"{company}:")
            print(f"Filed: {info['filing_date']}  |  {info['total_chunks']} chunks")
            for sec, cnt in info["by_section"].items():
                print(f"        {sec}: {cnt}")

    print()
    print("Output files:")
    print("data/raw/          → raw HTML filings + manifest.json")
    print("data/parsed/       → extracted sections per company")
    print("data/chunks/       → JSONL chunks")


def main():
    parser = argparse.ArgumentParser(description="SEC 10-K RAG Day 1 Pipeline")
    parser.add_argument(
        "--step",
        choices=["fetch", "parse", "chunk", "ingest", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip EDGAR fetch (use existing data/raw/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Target chunk size in characters (default: 800)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Overlap between chunks in characters (default: 150)",
    )
    args = parser.parse_args()

    fetch_count = parse_count = 0
    chunk_stats = None
    start = time.time()

    if args.step in ("all", "fetch") and not args.skip_fetch:
        fetch_count = step_fetch(args)
    elif args.skip_fetch:
        manifest = Path("data/raw/manifest.json")
        if manifest.exists():
            fetch_count = len(json.loads(manifest.read_text()))
            print(f"\nSKIP fetch using existing data ({fetch_count} companies in manifest)")
        else:
            print("ERROR --skip-fetch used but data/raw/manifest.json not found")
            sys.exit(1)

    if args.step in ("all", "parse"):
        parse_count = step_parse(args)

    if args.step in ("all", "chunk"):
        chunk_stats = step_chunk(args)
    
    ingest_count = 0
    if args.step in ("all", "ingest"):
        ingest_count = step_ingest(args)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    if args.step == "all":
        print_summary(fetch_count, parse_count, chunk_stats, ingest_count)


if __name__ == "__main__":
    main()