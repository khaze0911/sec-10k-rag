"""
src/eval.py — retrieval hit@k evaluation harness

For each gold question, the target is a (company, section) pair. A retrieved
chunk is a HIT if both its company and section match the target. hit@k = 1 if
any of the top-k retrieved chunks is a hit, else 0. Reports the MEAN hit@k
over questions, computed for three channels:

    bm25   — lexical channel alone
    vector — dense channel alone 
    fused  — RRF combination of both


Run from project root (needs src/__init__.py):
    python -m src.eval --probes 1 --out eval/results_lists100_probes1.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retriever import (
    HybridRetriever,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    get_connection,
    CANDIDATE_K,
    TOP_K,
    RRF_K,  # imported from retriever so the harness can't drift from production's value
)

@dataclass
class QuestionResult:
    id: str
    question: str
    target_company: str
    target_section: str
    tag: str
    hit_bm25: int
    hit_vector: int
    hit_fused: int
    # rank at which the first hit appears per channel (1-based), or None if no hit
    rank_bm25: int | None
    rank_vector: int | None
    rank_fused: int | None

def _is_hit(chunk: dict[str, Any], company: str, section: str) -> bool:
    """A chunk hits if both company and section match the gold target exactly"""
    return chunk.get("company") == company and chunk.get("section") == section

def _first_hit_rank(results: list[dict[str, Any]], company: str, section: str) -> int | None:
    """1-based rank of the first hit in a ranked list, or None if absent"""
    for i, c in enumerate(results, 1):
        if _is_hit(c, company, section):
            return i
    return None

def _set_probes(conn, probes: int) -> None:
    """RESET (not SET=1) to restore the server/session default cleanly"""
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = %s;", (probes,))

def _reset_probes(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("RESET ivfflat.probes;")

def evaluate(probes: int, k: int) -> dict[str, Any]:
    gold_path = Path(__file__).resolve().parent.parent / "eval" / "gold_qa.json"
    with open(gold_path) as f:
        gold = json.load(f)

    questions = gold["questions"]
    retriever = HybridRetriever()

    per_question: list[QuestionResult] = []

    with get_connection() as conn:
        # Set probes once for this run on this connection
        _set_probes(conn, probes)
        try:
            for q in questions:
                company = q["target_company"]
                section = q["target_section"]

                # Each channel queried directly HIT decision uses only the top-k slice
                bm25 = bm25_search(q["question"], retriever.bm25, retriever.chunk_ids, conn, k=CANDIDATE_K)
                vec = vector_search(q["question"], retriever.model, conn, k=CANDIDATE_K)
                fused = _fuse(bm25, vec, k=k)

                bm25_k = bm25[:k]
                vec_k = vec[:k]
                fused_k = fused[:k]

                per_question.append(QuestionResult(
                    id=q["id"],
                    question=q["question"],
                    target_company=company,
                    target_section=section,
                    tag=q["tag"],
                    hit_bm25=int(_first_hit_rank(bm25_k, company, section) is not None),
                    hit_vector=int(_first_hit_rank(vec_k, company, section) is not None),
                    hit_fused=int(_first_hit_rank(fused_k, company, section) is not None),
                    rank_bm25=_first_hit_rank(bm25, company, section),
                    rank_vector=_first_hit_rank(vec, company, section),
                    rank_fused=_first_hit_rank(fused, company, section),
                ))
        finally:
            _reset_probes(conn)

    retriever.close()
    return _summarize(per_question, probes=probes, k=k)


def _fuse(bm25: list[dict[str, Any]], vec: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Wrapper around production RRF"""
    fuse_depth = max(k, CANDIDATE_K)
    return reciprocal_rank_fusion(bm25, vec, k=fuse_depth)

def _mean(xs: list[int]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0

def _summarize(results: list[QuestionResult], probes: int, k: int) -> dict[str, Any]:
    normal = [r for r in results if r.tag == "normal"]
    impossible = [r for r in results if r.tag == "retrieval-impossible"]

    def channel_means(rows: list[QuestionResult]) -> dict[str, float]:
        return {
            "bm25": _mean([r.hit_bm25 for r in rows]),
            "vector": _mean([r.hit_vector for r in rows]),
            "fused": _mean([r.hit_fused for r in rows]),
        }

    return {
        "run_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "probes": probes,
            "k": k,
            "rrf_k": RRF_K,
            "candidate_k": CANDIDATE_K,
            "n_questions_total": len(results),
            "n_normal": len(normal),
            "n_retrieval_impossible": len(impossible),
        },
        # HEADLINE: number compared across runs
        "headline_hit_at_k_normal_only": channel_means(normal),
        # Full breakdown including the tagged-impossible questions, scored separately
        "retrieval_impossible_hit_at_k": channel_means(impossible),
        "all_questions_hit_at_k": channel_means(results),
        "per_question": [asdict(r) for r in results],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval hit@k eval.")
    ap.add_argument("--probes", type=int, default=1,
                    help="ivfflat.probes for this run. Baseline=1. Set per the config you're measuring.")
    ap.add_argument("--k", type=int, default=TOP_K,
                    help="k for hit@k (top-k slice checked for a hit). Default = retriever TOP_K.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output JSON path, e.g. eval/results_lists100_probes1.json. "
                         "Encode the lists value in the filename since the harness can't read it.")
    args = ap.parse_args()

    summary = evaluate(probes=args.probes, k=args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    h = summary["headline_hit_at_k_normal_only"]
    print(f"\n=== hit@{args.k}  (probes={args.probes}) ===")
    print(f"  HEADLINE (normal questions only, n={summary['run_meta']['n_normal']}):")
    print(f"    BM25   : {h['bm25']}")
    print(f"    vector : {h['vector']}")
    print(f"    fused  : {h['fused']}")
    imp = summary["retrieval_impossible_hit_at_k"]
    print(f"  retrieval-impossible (n={summary['run_meta']['n_retrieval_impossible']}, scored separately):")
    print(f"    bm25={imp['bm25']} vector={imp['vector']} fused={imp['fused']}")
    print(f"\n  wrote -> {out_path}")


if __name__ == "__main__":
    main()