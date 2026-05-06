.PHONY: install train score notebook notebook-score test lint format typecheck docker-build docker-run

install:
	pip install uv && uv sync --all-extras

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
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check . --fix

typecheck:
	uv run mypy groundwork/

docker-build:
	docker build -f docker/Dockerfile.batch -t groundwork-batch .

docker-run:
	docker run --rm \
		-v "$$(pwd)/data:/app/data" \
		-v "$$(pwd)/models:/app/models" \
		groundwork-batch
