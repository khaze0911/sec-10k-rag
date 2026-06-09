# Known First Pass Retrieval Failures

Retrieval failures identified before formal evaluation, included in the gold Q&A set as stress tests.

**STATUS (2026-06-09, post- eval):** F1 FIXED & MEASURED. F2/F3 CONFIRMED
as parser gaps (plus one new F2 sibling: Coinbase). F4 OPEN, the hit@k eval is
structurally blind to it (see F4 note). F5 NEW, fusion demoting single-channel
hits, confirmed repeatable. Eval artifacts: eval/results_lists100_probes1.json
(frozen baseline), eval/results_lists16_probes1.json, eval/results_lists16_probes4.json.
Metric is hit@5 over 15 gold questions (12 normal, 3 retrieval-impossible);
BM25 was identical across all three runs (unchanged control validates harness
stability and attributes all movement to the index).

Editing rule for this file: superseded hypotheses are kept and marked
[SUPERSEDED BY MEASUREMENT], not deleted. The prediction-then-correction record
is the point of the document.

---

## F1 — Visa "key risks": vector channel missed all Visa risk_factors
   [FIXED & MEASURED: binding constraint was probes, not lists; see MEASURED OUTCOME]

- Query: "What does Visa identify as key risks?"
- Original symptom: fused top-5 had NO risk_factors chunks despite Visa having 178 in DB
  (count verified against live DB 2026-06-09); RRF scores all ~0.016.

- ORIGINAL ROOT-CAUSE HYPOTHESIS: the ivfflat index is over-partitioned for this
  corpus. lists=100 on ~2,900 chunks puts only ~29 vectors per cell. At the
  default probes=1, only one cell (~1% of the corpus) is scanned per query, so
  an entire company's risk_factors section lived in cells that were never probed
  and was invisible to vector search.
    - lists=100 came from a default config snippet tuned for large tables; the
      rows/1000 heuristic for this corpus would be ~3 lists, not 100. The index
      was sized for a scale this corpus doesn't have.
    - "probes=1 is the DEFAULT and the trigger; lists=100 is the underlying
      cause. Both axes are mistuned."
      [PARTIALLY SUPERSEDED BY MEASUREMENT: the measured attribution came out
      the other way around, probes was the binding constraint; the lists
      rebuild alone was net-negative on vector hit@5. See MEASURED OUTCOME.
      The unified framing that survives: lists and probes jointly set the
      scanned fraction of the corpus; defaults gave ~1%, recall needed far
      more. The rebuild rationalized the index geometry; the recovery came
      from probes.]

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
  buried it (rank 13 in the original diagnostic query; rank 8 for the gold-set
  phrasing Q01 different query strings, both out of top-5) under Visa/business
  chunks that match the token "Visa" strongly. So at fusion: BM25 contributes it
  weakly, vector contributed nothing (probes=1) → never reached fused top-5.
  Both channels failing it for DIFFERENT reasons is why F1 looked total.

- FIX, AS EXECUTED (one variable per measured step):
    1. Frozen baseline captured at lists=100, probes=1
       (eval/results_lists100_probes1.json) before any index change.
    2. Index rebuilt: DROP INDEX chunks_embedding_idx; CREATE INDEX ... ivfflat
       (embedding vector_cosine_ops) WITH (lists = 16). Re-measured at probes=1.
    3. probes raised 1 → 4 (≈sqrt(16)), set per-session on the same pooled
       connection as the vector query. Re-measured.
  [CORRECTED from an earlier draft of this section that proposed a
  {lists 16,100} × {probes 1, tuned} 2×2 grid: the cell lists=100 × probes=4
  is a mismatched point (tuned probes for lists=100 would be ~10), so the 2×2
  reintroduced the attribution problem. The 3-point single-variable path above
  replaced it.]
  Tradeoff (unmeasured in this eval): higher probes = more cells scanned =
  slower. No latency numbers were captured; the recall/latency tradeoff is
  acknowledged qualitatively, not quantified.

- MEASURED OUTCOME (hit@5, normal questions n=12; full per-question data in
  the three results files):

    config                     bm25    vector   fused
    lists=100, probes=1        0.75    0.6667   0.8333   (frozen baseline)
    lists=16,  probes=1        0.75    0.5833   0.8333   (rebuild alone)
    lists=16,  probes=4        0.75    0.75     0.9167   (tuned probes)

    Q01 (this failure) vector rank: none → 9 → 1.  Fused rank: 15 → 15 → 1.

  Findings:
    a. F1 is fixed: Visa risk_factors is rank-1 on both vector and fused at the
       final config.
    b. ATTRIBUTION: the lists rebuild alone was net-negative on vector hit@5
       (0.67 → 0.58). At probes=1 the retriever scans one
       cell regardless of lists; re-clustering reassigns which vectors share the
       probed cell, so coverage moved rather than expanded (Q07 recovered; Q04
       and Q08 regressed). Which specific questions regressed is k-means
       seed-dependent, the structural claim (probes=1 makes recall a lottery
       over cell assignments) is the durable one; the specific winners/losers
       are noise. The recovery came from probes (0.58 → 0.75 vector,
       0.83 → 0.92 fused).
    c. The prior diagnostic already showed probes=10 on the original lists=100
       index recovering F1, consistent with (b): raising probes alone would
       likely also have fixed recall. The honest claim for keeping lists=16 is
       narrower: it matches the corpus-size heuristic and means a small probes
       value scans a meaningful fraction (4/16 = 25% vs 4/100 = 4%). not 
       claiming the rebuild was necessary for recall; the data doesn't support it.
    d. The damage at the broken config was localized not global: 8/12 normal
       questions hit on vector even at baseline. See Cross-cutting lesson.
    e. Coverage can inflate apparent precision: Q04 (Lemonade) vector rank went
       2 → 7 → 13 as probes/coverage increased. More probes only ADDS candidates,
       so a falling rank means genuinely closer competitors entered from newly
       scanned cells — the baseline rank-2 "hit" was partly an artifact of
       scanning ~1% of the corpus and hiding competitors.

- Durable code fix: lists value corrected in ingest.py (computed from corpus
  size, not hardcoded) so the next re-ingest cannot silently recreate the
  over-partitioned index. Live index fixed via DDL, separately, so the eval
  delta is attributable to exactly one change at a time.

- RESIDUAL (new, distinct from the index issue): Q04 (Lemonade risk_factors,
  vector rank 13), Q13 (Coinbase market_risk, rank 9), Q15 (Visa mda, rank 7)
  still miss vector top-5 at probes=4. 
  PENDING VERIFICATION
  All three are fused HITS (BM25 carries them) — hybrid is doing its job here.

- Expected gold answer source: Visa / risk_factors. (Gold set: Q01.)

## F2 — Block Bitcoin query returns PayPal
  [CONFIRMED: parser gap, 0 chunks; plus one sibling found]
- Block's filing uses "Square" / "Cash App" branding; parser missed Block's
  risk_factors entirely.
- VERIFIED against live DB 2026-06-09: Block (Square) risk_factors = 0 chunks.
  Eval Q09 (tagged retrieval-impossible) missed on every channel in every run,
  as expected.
- RETRIEVAL-IMPOSSIBLE, not a retrieval-quality failure. The fix is
  ingest/parser; any "improvement" only appears after re-parsing Block, not
  after any retrieval change.
- Original note suggested some of F2 "may have been the same ivfflat coverage
  issue masking Block content."
  [SUPERSEDED: coverage cannot mask what was never
  ingested. With 0 chunks in the DB, the index is irrelevant. Confirmed by the
  index fix changing nothing on Q09.]
- Coinbase: risk_factors = 0 chunks AND mda = 0 chunks (live DB
  2026-06-09). Same class of parser/section-boundary failure as Block. Surfaced
  when a retriever smoke-test query for "Coinbase's main risk factors" returned
  business-section chunks, correct behavior against an absent section.
- Related text-quality note (parser, minor): EDGAR page-header boilerplate
  ("Table of Contents" + page numbers) is scattered through chunk text, it is
  per-page navigation debris, not a TOC-boundary failure. Uniform weak noise in BM25 tokens and embeddings; strip header
  lines at parse time in the parser batch.

## F3 — Goldman Sachs & JPMorgan Chase market_risk: 1 chunk each
  [CONFIRMED: near-absent sections; verified against live DB]
- Parser barely caught these sections: Goldman Sachs market_risk = 1 chunk,
  JPMorgan Chase market_risk = 1 chunk (live DB 2026-06-09). Any market_risk
  query for these two has almost nothing to retrieve. Parser/section-boundary
  issue.
- Eval Q10/Q11 (tagged retrieval-impossible) missed on every channel in every
  run, consistent with near-absence. Scored separately from retrieval quality.
- String discipline note: the DB stores 'JPMorgan Chase' (and 'Block (Square)').
- Contrast case from the eval: Coinbase market_risk (34 chunks, the richest in
  the corpus) HITS fused in all runs (Q13) market_risk retrieval works
  when the data exists. The thinness is a per-company parser problem, not a
  section-type problem.

## F4 — Goldman Sachs risk_factors chunk 202 = generic fallback
  [OPEN: hit@k eval is structurally blind to this failure class]
- Appears as weak match across unrelated queries. Generic legalese scoring flat
  on everything → noise in candidate pools. THIS is the genuine ranking-quality
  case (unlike F1).
- WHY THE DAY-4 EVAL CANNOT SCORE IT: hit@k asks "did the right (company,
  section) surface in top-k." Chunk 202 IS a Goldman/risk_factors chunk, so if
  it surfaces for a Goldman risk question it COUNTS AS A HIT even though it is
  the generic chunk, not substantive content. F4 is a PRECISION problem;
  hit@k is a section-recall metric. Gold Q12 carries an explicit metric_caveat
  to this effect and serves as the manual-inspection query.
- Measuring it properly requires a separate instrument: e.g. the rate at which
  chunk 202 appears in top-k across queries where it is NOT a correct answer
  (a false-positive rate). Not built in this sprint; logged as the known next
  eval extension alongside the parser batch.

## F5 — RRF fusion demotes single-channel vector hits (NEW, confirmed repeatable)
- Observed on Q08 (Block (Square) market_risk, 9 chunks): vector finds the
  target at rank 5; BM25 has it nowhere in top-20; fused buries it at rank 11.
- Repeatable: identical pattern at baseline (vector 5 / fused 11) and at
  lists=16 probes=4 (vector 5 / fused 11). Absent only in the probes=1
  lists=16 run because the vector channel itself lost the target there
  (seed shuffle), i.e. the one run where fusion couldn't demote it.
- Mechanism: a single-channel rank-5 contributes ~1/(5+60) ≈ 0.0154 RRF; chunks
  appearing in BOTH channels' top-20 accumulate two contributions and leapfrog
  it. With CANDIDATE_K=20 per channel, RRF systematically favors dual-channel
  presence over a strong single-channel signal.
- Status: documented limitation, not fixed in this sprint. Candidate
  mitigations to evaluate later (each needs its own before/after): larger
  CANDIDATE_K, weighted RRF, or channel-score normalization. Also a useful
  counterweight in the writeup: fusion provided robustness while the vector
  channel churned across configs (fused held 0.83 through the vector dip),
  and fusion cost one vector hit.

## F6 — risk_factors section END-boundary overshoot (NEW, post-eval; parser)
  [OPEN: fix in the parser batch; systemic across companies, quantification pending]

- DISCOVERY PATH: smoke test → rag_chain check, both by reading retrieved text.
  hit@k is structurally blind to it (a mislabeled chunk with the right section
  label scores as a hit).
- Symptom: the production answer for the gold Q01 phrasing names only
  cybersecurity risks. Sources: rank 1 = chunk 157, rank 2 = chunk 156. Chunk
  156 carries the "ITEM 1B" header; 157 follows it. Tail spot-check
  (ORDER BY chunk_index DESC) confirms chunks 173–177 are Item 1C
  (cybersecurity governance), Items 2/3 (properties, legal proceedings), and
  Item 5 (holders of record, dividends), none of it Item 1A risk factors.
  Estimated contamination for Visa: chunks ~156–177, ≈22 of 178 (~12%).
- MECHANISM: find_section_boundaries tracks five sections; a section's end is
  the next tracked section's start. The 10-K items between 1A and 7 (1B, 1C,
  2, 3, 4, 5, 6) are untracked, so everything from the true end of Item 1A to
  the start of MD&A is absorbed into risk_factors. By the same logic every
  company's risk_factors (Goldman 227, JPMorgan Chase 222, Lemonade 221) is
  predicted to carry a contaminated tail. 
  SELECT company, min(chunk_index) AS first_overshoot, max(chunk_index)+1 AS total
  FROM chunks WHERE section='risk_factors' AND text ILIKE '%item 1b%'
  GROUP BY company ORDER BY company;
- WHY (hypothesis, unconfirmed):
  chunks 156/157 contain section-header/meta language including the literal
  phrase "risk factors"; the query contains the same phrase. Genuine Item 1A
  body chunks describe specific risks without ever saying "risk factors."
  Check: run the gold phrasing through the per-channel diagnostic and see where genuine chunks 
  (index < first_overshoot) rank in vector top-20.
- EVAL IMPLICATION (honesty note): hit@k inherits the DB's section labels. The
  measured numbers are faithful to "did a chunk LABELED (company, section)
  surface". The interpretation "retrieved the company's
  risk factors" is weakened for any hit landing in a contaminated tail. This
  is label error flowing through retrieval and generation to a wrong answer
  that the metric certified as a perfect hit; F6 is metric blind to LABEL error.
- FIX (parser batch, post-sprint): a section ends at the next any-item header
  (regex on ITEM \d+[A-C]?), not the next tracked section. Same function also
  owns the F2 start-boundary failures. Then re-parse, re-ingest (ingest.py now sizes lists from the real
  row count and builds the index post-load), and re-run the eval as its own
  before/after. 
- Expected on re-eval: section counts change; F2/F3 tags must be
  re-verified (Block/Coinbase may gain sections, flipping Q09-class tags from
  retrieval-impossible to normal); the Q01 answer should broaden from
  cyber-only to genuine Item 1A coverage.

---

## Fix routing (updated post-eval)
- Vector recall / config: F1 FIXED via probes (binding constraint), index
  rebuilt to lists=16 (geometry rationalization; not necessary for recall on
  the evidence). Durable fix in ingest.py.
- Parser/ingest side (one batch, AFTER committing eval results; re-run eval
  after re-parse with its own before/after): F2 Block risk_factors, F2-sibling
  Coinbase risk_factors + mda, F3 thin market_risk (Goldman, JPMorgan Chase),
  page-header stripping.
- Ranking/chunk-quality side: F4 (generic chunk needs a precision
  instrument), F5 (RRF depth/weighting), residual vector misses Q04/Q13/Q15
  (pending probes=16 exact-search confirmation).

## Cross-cutting lesson (revised post-measurement)
- ORIGINAL CLAIM: the misconfiguration "plausibly degrades every vector query
  in the system; Visa is just the symptom that surfaced first."
  [SUPERSEDED BY MEASUREMENT: the damage was localized, not global.
  At the broken config, 8/12 normal questions hit on the vector channel
  (vector hit@5 = 0.67); the affected minority (Q01, Q07, Q13, Q15-class)
  suffered total or near-total vector misses. "Everything was broken" was a
  plausible guess; the eval corrected it to "a concentrated class of queries
  was invisible while most were untouched."]
- The grid originally proposed here ({lists 16,100} × {probes 1, tuned}) was
  wrong for attribution and was replaced by the 3-point single-variable path;
  see F1 FIX AS EXECUTED.
- Metric note: the eval uses hit@5 against (company, section) gold targets,
  not recall@k. With 178–227 relevant chunks per risk_factors question, an
  exhaustive relevant-set labeling is infeasible and any partial set bootstrapped
  from retriever output would be circular. hit@k is the honest metric for this
  gold set; "recall@k" earlier in this doc's history referred to the same
  intended measurement before the metric was settled.
- What the eval methodology bought, concretely: changing one variable per step
  is the ONLY reason the rebuild's net-negative effect is visible at all. A
  combined lists+probes change would have shown 0.67 → 0.75 and the rebuild
  would have been wrongly credited. The frozen baseline + unchanged BM25
  control + per-question ranks turned a config fix into an attributable,
  defensible result.