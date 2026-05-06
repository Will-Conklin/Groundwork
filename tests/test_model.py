import numpy as np
import polars as pl
import pytest
import lightgbm as lgb

from groundwork.features import FEATURE_COLS
from groundwork.model import FraudModel


@pytest.fixture
def saved_model(tmp_path) -> str:
    rng = np.random.default_rng(42)
    n = 40
    X = rng.random((n, len(FEATURE_COLS)))
    y = rng.integers(0, 2, n)
    ds = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "num_leaves": 4, "n_estimators": 5, "verbose": -1}
    booster = lgb.train(params, ds, num_boost_round=5)
    model_path = str(tmp_path / "model.txt")
    booster.save_model(model_path)
    return model_path


def test_fraud_model_loads(saved_model):
    model = FraudModel(saved_model)
    assert model is not None


def test_predict_proba_returns_1d_array(saved_model):
    model = FraudModel(saved_model)
    df = pl.DataFrame({col: [0.5] * 10 for col in FEATURE_COLS})
    scores = model.predict_proba(df)
    assert scores.ndim == 1
    assert len(scores) == 10


def test_predict_proba_values_in_0_1(saved_model):
    model = FraudModel(saved_model)
    df = pl.DataFrame({col: [0.5] for col in FEATURE_COLS})
    scores = model.predict_proba(df)
    assert float(scores[0]) >= 0.0
    assert float(scores[0]) <= 1.0


def test_missing_model_file_raises():
    with pytest.raises(Exception):
        FraudModel("/nonexistent/path/model.txt")
