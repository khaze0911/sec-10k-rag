"""
scripts/diagnostic_retrieval.py — F1 diagnostic: inspect retrieval channels for one query

Pairs with ../known_failures.md F1. Frozen single-query diagnostic that caught the ivfflat coverage problem; 
the generalized corpus-wide version lives in src/eval.py.

Run from project root (needs scripts/__init__.py):
    python -m scripts.diagnostic_retrieval

Goal: show BM25 top-N and vector top-N SEPARATELY (before RRF), plus a second
vector pass with ivfflat.probes raised. Confirms whether Visa risk_factors is
missing due to ivfflat index coverage (over-partitioned lists + low probes) vs
genuinely absent. If the target appears at probes=10 but not probes=1, the
vectors are fine and the index simply wasn't probing the right cells.
"""

from src.retriever import (
    HybridRetriever,
    bm25_search,
    vector_search,
    get_connection,
    CANDIDATE_K,
)

QUERY = "What does Visa identify as key risks?"
TARGET_COMPANY = "Visa"
TARGET_SECTION = "risk_factors"
N = CANDIDATE_K  # 20


def show(label, results):
    print(f"\n{'='*70}\n{label} — top {len(results)}\n{'='*70}")
    for i, r in enumerate(results, 1):
        hit = "  <<< TARGET" if (
            r["company"] == TARGET_COMPANY and r["section"] == TARGET_SECTION
        ) else ""
        print(f"  [{i:2}] {r['company']:<16} {r['section']:<20} "
              f"chunk {r.get('chunk_index','?'):<4} score={r['score']:.4f}{hit}")


def has_target(results):
    return [
        (i + 1, x["company"], x["section"], x.get("chunk_index"))
        for i, x in enumerate(results)
        if x["company"] == TARGET_COMPANY and x["section"] == TARGET_SECTION
    ]


def main():
    r = HybridRetriever()

    with get_connection() as conn:
        # default probes (=1)
        bm25 = bm25_search(QUERY, r.bm25, r.chunk_ids, conn, k=N)
        vec = vector_search(QUERY, r.model, conn, k=N)

        # ivfflat coverage discriminator: re-run the SAME vector query with more probes
        # if the target appears now but not above, the index wasn't probing the cells the true neighbors live in 
        with conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = 10;")
        vec_hi = vector_search(QUERY, r.model, conn, k=N)
        with conn.cursor() as cur:
            cur.execute("RESET ivfflat.probes;")  # back to session/server default

        # DB sanity: confirm Visa risk_factors chunks exist + how they look.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM chunks WHERE company=%s AND section=%s;",
                (TARGET_COMPANY, TARGET_SECTION),
            )
            count = cur.fetchone()[0]
            cur.execute(
                "SELECT chunk_index, left(text, 220) FROM chunks "
                "WHERE company=%s AND section=%s ORDER BY chunk_index LIMIT 2;",
                (TARGET_COMPANY, TARGET_SECTION),
            )
            samples = cur.fetchall()

    show("BM25 candidates", bm25)
    show("VECTOR candidates (probes=1)", vec)
    show("VECTOR candidates (probes=10)", vec_hi)

    bm25_hits = has_target(bm25)
    vec_hits = has_target(vec)
    vec_hi_hits = has_target(vec_hi)

    print(f"\n{'#'*70}")
    print(f"# Visa risk_factors in BM25 top-{N}:            "
          f"{bm25_hits if bm25_hits else 'NONE'}")
    print(f"# Visa risk_factors in VECTOR top-{N} (probes=1):  "
          f"{vec_hits if vec_hits else 'NONE'}")
    print(f"# Visa risk_factors in VECTOR top-{N} (probes=10): "
          f"{vec_hi_hits if vec_hi_hits else 'NONE'}")
    print(f"{'#'*70}")

    print(f"\nVisa risk_factors chunks in DB: {count}")
    for idx, preview in samples:
        print(f"  chunk {idx}: {preview!r}")

    r.close()


if __name__ == "__main__":
    main()