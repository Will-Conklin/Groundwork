default:
    @just --list

install:
    pip install uv
    uv sync --all-extras

train:
    uv run python notebooks/train.py

score:
    uv run python notebooks/batch_score.py

notebook:
    uv run marimo edit notebooks/train.py

notebook-score:
    uv run marimo edit notebooks/batch_score.py

test:
    uv run pytest tests/ -v

lint:
    uv run ruff check .
    uv run ruff format --check .

format:
    uv run ruff format .
    uv run ruff check . --fix

typecheck:
    uv run mypy groundwork/

scan:
    uv run bandit -r groundwork/ -c pyproject.toml

docker-build:
    docker build -f docker/Dockerfile.batch -t groundwork-batch .

docker-run:
    docker run --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        groundwork-batch

mlflow-build:
    docker build -f docker/Dockerfile.mlflow -t groundwork-mlflow .

mlflow-run:
    mkdir -p mlflow-data mlruns
    docker run --rm \
        -p 5000:5000 \
        -v "$(pwd)/mlflow-data:/mlflow-data" \
        -v "$(pwd)/mlruns:/mlruns" \
        groundwork-mlflow
