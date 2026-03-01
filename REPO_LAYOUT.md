# Repository Layout

This repository now follows a code-first root layout:

- `prismatic/`: Main Python package (models, VLA datasets, training, config).
- `vla-scripts/`: VLA training/evaluation entry scripts.
- `scripts/`: Generic preprocessing/pretraining/helper scripts.
- `experiments/`: Benchmark and robotics evaluation utilities.
- `docs/`: Documentation and non-runtime assets.
  - `docs/assets/`: README and project media assets.
  - `docs/paper/`: Paper source/archive artifacts.
- `pretrained_models/`: Local model/config payloads used by current scripts.

## Placement Rules

- Put new runtime Python code under `prismatic/` (or script entrypoints under `vla-scripts/` / `scripts/`).
- Put paper figures/tex/archive files under `docs/paper/`.
- Put repository documentation images under `docs/assets/`.
- Keep generated outputs out of git (`eval_logs/`, `runs/`, `outputs/`, `*.egg-info`).
