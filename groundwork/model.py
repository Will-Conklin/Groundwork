from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from loguru import logger


class FraudModel:
    def __init__(self, model_path: Path | str) -> None:
        model_path = Path(model_path)
        logger.info("Loading model from {}", model_path)
        self._booster = lgb.Booster(model_file=str(model_path))

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Return fraud probability per row (1D array, values 0–1)."""
        return np.asarray(self._booster.predict(df.to_numpy()))
