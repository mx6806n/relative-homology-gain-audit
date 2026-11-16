# Relative homology gain audit

Code and derived data for the article

> Moumane A, Es-Sabery B, Oufqir S, Baslam M, Ouhda M. *Candidate size confounds
> raw relative homology gains for course ranking on prerequisite graphs.*

The study ranks candidate courses by the raw relative homology gain

    g_S(C) = beta_1(K, K[S]) - beta_1(K, K[S ∪ C])

where `K` is the directed flag complex of a prerequisite graph, `S` is the
transcript of a learner and `C` is the concept bundle of a candidate course.
The gain decomposes exactly as `g = q - m + h - t` (absorbed edges, absorbed
vertices, newly reached components and the drop in the rank of the relative
boundary map `d_2`). The repository contains the homology evaluator over GF(2),
the three steps of the comparability audit, the reference recommenders, the
fusion experiments and the scripts that produce every table and figure of the
article.

## Repository layout

```
relhom/        library: homology evaluator, data loading, models, audit statistics
experiments/   one folder per analysis, each with run.py and its configuration
results/       stored outputs of every analysis (JSON and CSV)
scripts/       data preparation, DORIS graph construction, verification, tables and figures
tests/         unit and regression tests
docs/          predeclared decision rules (analysis_plan.md)
data/          location of the third party datasets (not distributed)
```

Main modules of `relhom/`:

| Module | Content |
|---|---|
| `topology.py` | Directed flag complex, exact relative `beta_1` over GF(2), mapping cone reference with pyflagser |
| `topology_mechanism.py` | Terms `q`, `m`, `h`, `t` of the decomposition and structural features of candidates |
| `size_stratified_diagnostic.py` | Target AUC within strata of equal size (Step 3) |
| `null_models.py` | Null scores matched to the bound (Step 2) |
| `sasrec.py`, `bpr.py` | SASRec and BPR matrix factorization |
| `hybrid_cache.py`, `topn_hybrid.py`, `normalization_sensitivity.py` | Global fusion, shortlist reranking and near tie sign tests |
| `data_loader.py`, `experiment.py` | MOOCCubeX loading, concept alignment and temporal leave two out splits |
| `doris_text_match.py` | Exact course title matching for the DORIS prerequisite graph |

`scripts/audit_doris_v3.py` builds the DORIS graph and reuses the archive
reading and catalog helpers of `scripts/audit_doris_supplement.py`, the audit of
the earlier substring based resolver.

## Installation

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/). All
dependency versions are pinned in `uv.lock`. From the root of the repository:

```bash
uv sync
uv run pytest
```

The analyses reported in the article used Python 3.12, NumPy 1.26.4,
pandas 2.3.3, SciPy 1.17.1, pyflagser 0.4.7 and the CPU build of PyTorch 2.14.1.
No GPU is required.

## Data

Neither dataset is redistributed here.

* **MOOCCubeX** (Yu et al., CIKM 2021) is available from its providers. Extract
  the release and point `MOOCCUBE2X_ROOT` to the folder that contains
  `entities/`, `relations/` and `prerequisites/`:

  ```bash
  cp .env.example .env    # then edit MOOCCUBE2X_ROOT
  ```

* **DORIS** (Ma et al., PLOS ONE 2023, doi:10.1371/journal.pone.0284687) is
  available as the S1 and S2 supporting datasets of its article. Download both
  ZIP files; the scripts take their paths as arguments.

The DORIS course prerequisite graph built for this study and the labels of the
reviewed edges are included in `results/`.

## Reproducing the analyses

Every analysis is run the same way:

```bash
uv run python experiments/<experiment>/run.py --config experiments/<experiment>/config.yaml
```

Some configurations are `config.json`. Each run writes `metrics.json` and
`summary.md` to `results/<experiment>/` and refuses to overwrite existing
outputs, so move or delete the stored folder before rerunning an analysis.

### 1. Prepare the MOOCCubeX Computer Science cohort

```bash
uv run python scripts/prepare_mooccubex.py --domain cs --export-aligned
```

This writes the concept graph, the course to concept mapping and the aligned
histories to `data/processed/mooccube2x_cs/`. The experiments keep users with
at least five aligned course events (28,589 users).

### 2. Reference recommenders (Table 8)

```bash
uv run python experiments/exp_002_popularity_unseen/run.py --config experiments/exp_002_popularity_unseen/config.yaml
uv run python experiments/exp_003_markov_baseline/run.py --config experiments/exp_003_markov_baseline/config.yaml
uv run python experiments/exp_004_bpr_mf_pilot/run.py --config experiments/exp_004_bpr_mf_pilot/config.yaml
uv run python experiments/exp_006_sasrec_validation_sweep/run.py --config experiments/exp_006_sasrec_validation_sweep/config.yaml
uv run python experiments/exp_007_sasrec_multiseed/run.py --config experiments/exp_007_sasrec_multiseed/config.yaml
```

### 3. Score caches

```bash
uv run python experiments/exp_010_validation_topology_cache/run.py --config experiments/exp_010_validation_topology_cache/config.yaml
uv run python experiments/exp_011_sasrec_validation_score_cache/run.py --config experiments/exp_011_sasrec_validation_score_cache/config.yaml
```

These write the validation gains and the SASRec logits as `.npz` files. Later
analyses check both caches against the SHA-256 digests in their configurations.
A retrained SASRec model can differ in the last bits of its logits, so use the
deposited caches to reproduce the stored results exactly.

### 4. Fusion and reranking (Tables 9 and 10, Fig 4)

```bash
uv run python experiments/exp_012_hybrid_alpha_ablation/run.py --config experiments/exp_012_hybrid_alpha_ablation/config.yaml
uv run python experiments/exp_013_topn_topology_rerank/run.py --config experiments/exp_013_topn_topology_rerank/config.yaml
uv run python experiments/exp_024_normalization_sensitivity/run.py --config experiments/exp_024_normalization_sensitivity/config.json
```

### 5. Comparability audit

```bash
# Step 1: controlled constructions (Table 5, Fig 2)
uv run python experiments/exp_019_relative_h1_counterexamples/run.py --config experiments/exp_019_relative_h1_counterexamples/config.yaml

# Decomposition terms in the three MOOCCubeX domains (Table 2, Table 7)
uv run python experiments/exp_018_relative_chain_decomposition/run.py --config experiments/exp_018_relative_chain_decomposition/config.yaml

# Step 3: target AUC stratified by size (Table 6, Fig 3)
uv run python experiments/exp_022_size_stratified_diagnostic/run.py --config experiments/exp_022_size_stratified_diagnostic/config.yaml
```

Step 2 needs the DORIS graph as well (section 6):

```bash
uv run python experiments/exp_023_bound_null_benchmark/run.py \
  --config experiments/exp_023_bound_null_benchmark/config.json \
  --s2 /path/to/S2.zip \
  --edges results/doris_candidate_prerequisite_edges_v3.csv \
  --feasibility results/doris_feasibility_final_v3.json
```

### 6. DORIS course graph and structural analysis

```bash
uv run python scripts/audit_doris_v3.py --s1 /path/to/S1.zip --s2 /path/to/S2.zip
uv run python scripts/finalize_doris_v3_feasibility.py
uv run python experiments/exp_021_doris_v3_semester_bundle_mechanism/run.py \
  --config experiments/exp_021_doris_v3_semester_bundle_mechanism/config.json \
  --s2 /path/to/S2.zip \
  --edges results/doris_candidate_prerequisite_edges_v3.csv \
  --feasibility results/doris_feasibility_final_v3.json
```

The audit extracts edges by exact and unambiguous course title matching and
draws a holdout of 100 edges for review. The finalization step reads the
reviewed holdout and applies the predeclared gates.

### 7. Further analyses

| Experiment | Content | Used in the article |
|---|---|---|
| `exp_008_topology_signal_profile` | Profile of the gain on 250 users; check of 25 states against the pyflagser mapping cone | Verification of the evaluator |
| `exp_014_accuracy_validity_pareto` | Prerequisite violation rates of targets and recommendations | Discussion |
| `exp_015_topology_mechanism_diagnostic` | Linear fit of the gain on four structural counts, Computer Science | Results (coefficient of determination) |
| `exp_016_secondary_domain_feasibility` | Feasibility gate for Mathematics and Psychology | Datasets and cohorts |
| `exp_017_cross_domain_topology_mechanism` | Linear fit in Mathematics and Psychology | Results (coefficient of determination) |

### 8. Tables and figures

```bash
uv run python scripts/build_tables_and_figures.py
```

This reads only the JSON files in `results/` and writes SVG figures and CSV or
Markdown tables to `outputs/`.

### 9. Independent check of the homology values

```bash
uv run python scripts/verify_relative_h1.py
```

The script recomputes relative `beta_1` on random DAGs with its own chain
matrices over GF(2) and over the rational numbers and compares the values with
the evaluator of the package. It also checks the decomposition and its bounds.

## Mapping between the article and the results

| Article | Result files |
|---|---|
| Table 2 | `exp_018`, `exp_021` |
| Table 5, Fig 2 | `exp_019` |
| Table 6, Fig 3 | `exp_022` |
| Table 7 | `exp_018`, `exp_021`, `exp_023` |
| Table 8 | `exp_002`, `exp_003`, `exp_004`, `exp_007` |
| Tables 9 and 10, Fig 4 | `exp_012`, `exp_013`, `exp_024` |
| Linear fits | `exp_015`, `exp_017` |
| Violation rates (Discussion) | `exp_014` |

## Predeclared analyses

The decision rules of the audit, the fusion grid, the feasibility gates and the
controlled constructions were fixed before their results were computed. They are
listed in [`docs/analysis_plan.md`](docs/analysis_plan.md) with the commits in
which they were recorded.

## Evaluation protocol in brief

* Temporal leave two out split per user: the second to last course is the
  validation target and the last course is the development test target.
* Every model ranks all unseen courses of the 117 course catalog; there are no
  sampled negatives.
* All model and weight selection uses validation NDCG@10. The development test
  split was inspected during model development and is reported as development
  evidence only.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the article and the two
datasets when you use this code.

## License

MIT. See [`LICENSE`](LICENSE). The licences of MOOCCubeX and DORIS apply to the
original datasets.
