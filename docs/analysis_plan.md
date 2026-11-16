# Predeclared analyses

This file collects the decision rules that were fixed before the corresponding
results were computed. Each entry keeps the identifier used in the result files
(`"predeclaration"` field) and names the commit of the development repository
in which the rule was recorded, followed by the commit that recorded the
result. Times are in UTC.

| ID | Analysis | Rule recorded | Result recorded |
|---|---|---|---|
| D-032 | Fusion grid and normalization (exp_012, exp_013) | `adeb594`, 2026-10-06 18:05 | `996631a`, 2026-10-06 20:34 |
| D-044 | Feasibility gate for Mathematics and Psychology (thresholds in the exp_016 configuration) | `548b2e4`, 2026-10-06 21:08 | `3ebae62`, 2026-10-06 21:37 |
| D-056 | Controlled constructions of Step 1 (exp_019) | `2fed6e2`, 2026-10-06 23:15 | `de9858c`, 2026-10-06 23:24 |
| D-065 | DORIS feasibility and precision gates | `526562f`, 2026-10-06 23:40 | `4bd74d4`, 2026-10-09 16:03 (v3) |
| D-074 | Step 3, target AUC stratified by size (exp_022) | `e57fd4e`, 2026-10-11 18:45 | `f906dd1`, 2026-10-11 19:05 |
| D-076 | Step 2, null benchmark matched to the bound (exp_023) | `56d6708`, 2026-10-11 19:26 | `a0d5944`, 2026-10-11 20:00 |
| D-080 | Normalization and near tie sensitivity of the fusion (exp_024) | `ba3222c`, 2026-10-11 21:17 | `99fa3e9`, 2026-10-11 21:53 |

## D-032 Fusion of the raw gain with SASRec

- SASRec logits are rank normalized over the unseen candidates of each user.
- The signed gain is range normalized over the same candidates.
- The weight alpha takes the values 0.00 to 1.00 in steps of 0.05 and is
  selected by validation NDCG@10. Exact ties are resolved toward the larger
  alpha, which leaves the SASRec ranking less perturbed.
- A variant that sets negative gains to zero is evaluated as a diagnostic only;
  the signed gain remains the main formulation.

## D-044 Feasibility gate for the secondary MOOCCubeX domains

A domain qualifies for the recommendation test only if it has at least 2,000
users with five or more aligned course events, at least 30 mapped courses and
at least 50% concept coverage. The gate is not relaxed after the counts are
seen. Domains that fail it are used only for analyses that read no
recommendation target.

## D-056 Controlled constructions

Three synthetic families are computed exactly over GF(2):

1. clone cap, with expected gain k(d - 1), for d = 1 to 4 and k = 1 to 5;
2. hollow diamond extended by a directed pendant chain, with expected gain 1
   for chain lengths 0 to 5;
3. filled diamond, with expected gain 0 because the excess q - m = 1 is
   cancelled by t = 1.

Every case must agree exactly with its expected value.

## D-065 DORIS graph

- Sequence feasibility: at least 2,000 students with five or more enrollment
  events and at least 1,500 students spanning three or more semesters.
- Deterministic graph feasibility: at least 25% of prerequisite bearing course
  rows resolve one released course, at least 500 directed edges and at least
  300 graph courses.
- Precision gate: a deterministic random sample of 100 extracted edges is
  labeled against the source text and at least 90% must be supported.
- Alignment feasibility: at least 500 graph courses appear in enrollment
  histories and at least 2,000 students have three or more graph mapped events.
- Thresholds are not lowered after the audit. Edges are extracted by exact and
  unambiguous course title matching only.

The v3 graph used in the article was built after an earlier substring based
resolver failed the precision gate. Its 100 edge holdout excludes every edge
reviewed for the earlier version (`results/doris_prerequisite_manual_review_sample.csv`)
and yielded 94 supported edges (`results/doris_prerequisite_manual_holdout_v3.csv`).

## D-074 Step 3: target discrimination stratified by size

- Primary estimand: mean tie aware target AUC (chance 0.5) of the signed gain
  within candidates that have the same number of new concepts m as the target,
  with a 95% bootstrap interval over users.
- Paired contrast: primary minus unstratified AUC on the same users.
- Minimum meaningful AUC: 0.55.
- Outcome A (interval low >= 0.55 and contrast low > 0): size masks a
  topological signal.
- Outcome B (interval low >= 0.55, contrast not above 0): a signal exists but
  size does not explain the recommender result.
- Outcome C (interval low < 0.55): no meaningful signal within size; the
  recommender result and the structural analysis are reported separately.
- The outcome is reported whatever it is. The analysis selects nothing,
  evaluates no corrected score and reads no development test data.

## D-076 Step 2: null benchmark matched to the bound

- The run first reproduces the stored exp_018 and exp_021 values to 1e-9 or
  stops.
- Nulls: N1 draws an integer uniformly from -m to q; N2 equals -Binomial(m, p)
  with p fitted to the share of zero gains (DORIS); N3 is q - m (descriptive);
  N4 uses random bundles of the same sizes (DORIS, descriptive).
- MOOCCubeX primary: mean excess over N1 of the Spearman correlation between
  |g| and m computed for each user, with a bootstrap over users.
- DORIS primary: excess over N1 and over N2, with a bootstrap over students.
- Margin 0.05. A setting is beyond bound if the lower 95% limit reaches 0.05
  (for DORIS, against both N1 and N2); otherwise it is bound dominated.
- The requirement that DORIS exceed N2 as well as N1 was added after a synthetic
  check and before any real output existed.

## D-080 Normalization and near tie sensitivity of the fusion

- Six configurations: global fusion, reranking of the SASRec top 10 and top 20,
  each with range or rank normalization of the gain.
- Grid: the 21 original values together with 0.96, 0.97, 0.98 and 0.99;
  selection by validation NDCG@10 with ties toward the larger alpha.
- The range normalized configurations must reproduce exp_012 and exp_013 on the
  original grid to 1e-12 or the run stops.
- A configuration improves on SASRec only if it selects alpha below 1 and the
  paired bootstrap interval of the change in NDCG@10 lies above 0.
- For each configuration, the targets that improve or worsen at the largest
  alpha below 1 with any movement are compared with an exact two sided
  binomial test.
