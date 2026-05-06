from pathlib import Path

import polars as pl
from loguru import logger


def load_csv(path: Path | str) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    logger.info("Loading CSV from {}", path)
    return pl.read_csv(path)


def save_csv(df: pl.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving {} rows to {}", len(df), path)
    df.write_csv(path)
