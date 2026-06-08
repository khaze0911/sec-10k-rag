# Known First Pass Retrieval Failures

Retrieval failures identified before formal evaluation, included in the gold Q&A set as stress tests.

---

## F1 — Visa "key risks": vector channel missed all Visa risk_factors
   [DIAGNOSED — ivfflat index over-partitioned for corpus size]

- Query: "What does Visa identify as key risks?"
- Original symptom: fused top-5 had NO risk_factors chunks despite Visa having 178 in DB; RRF scores all ~0.016.

- ROOT CAUSE: the ivfflat index is over-partitioned for this corpus. lists=100
  on ~2,900 chunks puts only ~29 vectors per cell. At the default probes=1,
  only one cell (~1% of the corpus) is scanned per query, so an entire company's
  risk_factors section lived in cells that were never probed and was invisible
  to vector search.
    - lists=100 came from a default config snippet tuned for large tables; the
      rows/1000 heuristic for this corpus would be ~3 lists, not 100. The index
      was sized for a scale this corpus doesn't have.
    - probes=1 is the DEFAULT and the trigger; lists=100 is the underlying
      cause. Both axes are mistuned.

- Evidence from diagnostic_retrieval.py (probes discriminator, lists still =100):
    - VECTOR probes=1  → Visa risk_factors: NONE in top-20
    - VECTOR probes=10 → Visa risk_factors: 12 of top-20, incl. ranks 1-4, with
      the BEST cosine distances in the whole set (chunk 157 = 0.3947 vs best
      probes=1 result = 0.5162).
  => The true nearest neighbors were always correct and well-separated; the
     index wasn't probing the cells they live in. Raising probes recovers
     them, confirming the vectors are fine and the problem is index coverage,
     not embedding quality.

- RULED OUT embedding / normalization / metric mismatch: if the embeddings or
  the cosine metric were wrong, scanning more cells would surface different
  wrong chunks, not the correct ones. Instead probes=10 returns chunks that are
  not just relevant but lower-distance (better) than anything probes=1 found.
  The vectors and metric are correct; the index coverage was the problem.

- BM25 side (contributing, secondary): BM25 did surface Visa risk_factors but
  only at rank 13, buried under 11 Visa/business chunks that match the token
  "Visa" strongly. So at fusion: BM25 contributes it weakly (rank 13), vector
  contributed nothing (probes=1) → never reached fused top-5. Both channels
  failing it for DIFFERENT reasons is why F1 looked total.

- FIX (hold for baseline). Two parts, measured separately:
    1. Rebuild the index at a corpus-appropriate lists. rows/1000 → ~3; ~16 is
       a reasonable floor that won't over-partition ~2,900 vectors.
    2. Tune ivfflat.probes against the new lists
  Tradeoff: higher probes = better recall = more cells scanned = slower.
  probes=1 fast+narrow; probes=lists ≈ full scan, slowest+exhaustive.
  Do NOT change lists and probes in the same measurement step — the recall delta
  would be unattributable. Measure across the grid {lists 16,100} × {probes 1,
  tuned} so each knob's effect is isolated.

- Expected gold answer source: Visa / risk_factors.

## F2 — Block Bitcoin query returns PayPal
- Block's filing uses "Square" / "Cash App" branding; parser missed Block's
  risk_factors entirely
- RETRIEVAL-IMPOSSIBLE, not a retrieval-quality failure: there is nothing to
  retrieve. Include the question in the gold set but TAG it retrieval-impossible.
  The fix is INGEST/PARSER; any "improvement" only appears after re-parsing
  Block, not after any retrieval change.
- NOTE: re-test F2 AFTER the index fix (lists rebuild + probes). Some of what
  looked like F2 may have been the same ivfflat coverage issue masking Block
  content. Separate the genuine parser gap (0 risk_factors chunks) from recall
  once the index is corrected.

## F3 — Goldman Sachs & JPMorgan market_risk: 1 chunk each
- Parser barely caught these sections. Any market_risk query for these two has
  almost nothing to retrieve. Likely parser/section-boundary issue.
  Flag per-question rather than scoring as pure retrieval quality.

## F4 — Goldman Sachs risk_factors chunk 202 = generic fallback
- Appears as weak match across unrelated queries. Generic legalese scoring flat
  on everything → noise in candidate pools. Watch in eval; may warrant chunk
  filtering or re-chunking. THIS is the genuine ranking-quality case (unlike F1).
- NOTE: also re-confirm AFTER the index fix — recall changes may alter which
  generic chunks surface.

---

## Fix routing (decide AFTER baseline)
- Vector recall / config: F1 — ivfflat over-partitioned (lists=100 on ~2,900
  rows). Fix = rebuild at lists≈16, then tune probes (≈sqrt(lists)).
- Parser/ingest side: F2 (missing Block risk_factors), F3 (thin market_risk).
- Ranking/chunk-quality side: F4 (generic chunk).
- Rule held to: only EXCLUDE a gold question if the section genuinely isn't in
  the filing. Never exclude a question the system merely answers badly — those
  are the point of the eval. (F2/F3 thin sections: INCLUDE but tag
  retrieval-impossible; don't exclude, don't score as retrieval quality.)

## Cross-cutting lesson
- The ivfflat config lists=100 (over-partitioned) plus the default probes=1
  is a global recall risk, not specific to F1. It plausibly degrades every
  vector query in the system; Visa is just the symptom that surfaced first. The
  config came from a default snippet sized for a much larger corpus and never
  re-tuned for ~2,900 rows. The Day-4 eval should measure recall@k corpus-wide
  across the grid {lists 16,100} × {probes 1, tuned}, not just for Visa, so the
  system-wide effect and the attribution to each knob are both visible. Re-examine
  F2 and F4 through this lens after the fix.