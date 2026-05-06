# Groundwork

An end-to-end ML pipeline for auto-insurance fraud detection — reactive marimo notebooks, LightGBM, Polars, and uv.

## The core idea

marimo notebooks are plain `.py` files. The same file you author interactively (with reactive sliders and live charts) runs headlessly as a batch job in production:

```bash
uv run python notebooks/train.py        # headless — exits 0, writes model
uv run marimo edit notebooks/train.py   # interactive — live sliders, charts
```

No conversion step. No separate "script" copy. One file, two modes.

## Quick start

```bash
pip install uv
uv sync --all-extras

# Train — generates synthetic data and saves models/fraud_model.txt
make train   # or: uv run python notebooks/train.py

# Score — reads data/claims.csv, writes data/predictions.csv
make score   # or: uv run python notebooks/batch_score.py

# Interactive training notebook (browser opens automatically)
make notebook

# Interactive scoring notebook
make notebook-score
```

## CLI overrides

Both notebooks accept `--key value` arguments and fall back to environment variables:

```bash
uv run python notebooks/batch_score.py -- \
    --input_path /data/live_claims.csv \
    --model_path /models/prod_model.txt \
    --output_path /results/predictions.csv

# Or via env vars
INPUT_PATH=/data/live_claims.csv make score
```

## Project layout

```
groundwork/          Pure Python package (no notebooks here)
  features.py        select_features() — column validation
  io.py              load_csv() / save_csv() — thin Polars wrappers
  model.py           FraudModel — wraps lgb.Booster

notebooks/
  train.py           Marimo training notebook (reactive sliders, AUC callout)
  batch_score.py     Marimo scoring notebook (threshold slider, histogram)

tests/               Unit tests — no disk I/O beyond tmp_path
data/                CSV files (git-ignored, .gitkeep present)
models/              LightGBM .txt model files (git-ignored)
docker/              Dockerfile.batch for headless production scoring
.github/workflows/   CI — lint, types, tests, headless train+score
```

## Model format

Models are saved as LightGBM's native text format (`models/fraud_model.txt`).
Human-readable (`cat models/fraud_model.txt` shows tree structure), no pickle,
no extra dependencies — load with `lgb.Booster(model_file=path)`.

## Development

```bash
make test       # pytest
make lint       # ruff check + format --check
make typecheck  # mypy groundwork/
make format     # ruff format + ruff --fix (writes)
```

## Docker

```bash
make docker-build   # builds groundwork-batch image
make docker-run     # scores data/ → predictions.csv via volume mounts
```
