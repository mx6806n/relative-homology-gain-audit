# Data

This folder is not tracked except for this file.

* Set `MOOCCUBE2X_ROOT` in `.env` to the extracted MOOCCubeX release.
* `scripts/prepare_mooccubex.py --domain cs --export-aligned` writes the
  processed Computer Science cohort to `data/processed/mooccube2x_cs/`:
  `concept_vocab.json`, `course_vocab.json`, `course_to_concepts.json`,
  `prerequisite_edges.csv` and `sequences.jsonl`.
* The DORIS S1 and S2 ZIP files can be stored anywhere; their paths are passed
  on the command line.
