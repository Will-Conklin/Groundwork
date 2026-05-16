import polars as pl
import pytest

from groundwork.features import FEATURE_COLS, select_features


def _make_df(**extra) -> pl.DataFrame:
    base = {col: [1.0] for col in FEATURE_COLS}
    base.update(extra)
    return pl.DataFrame(base)


def test_select_features_returns_only_feature_cols():
    df = _make_df(irrelevant_col=[99])
    result = select_features(df)
    assert result.columns == FEATURE_COLS


def test_select_features_preserves_row_count():
    df = pl.DataFrame({col: list(range(5)) for col in FEATURE_COLS})
    assert select_features(df).height == 5


def test_select_features_raises_on_missing_col():
    df = pl.DataFrame({"claim_amount": [1.0]})
    with pytest.raises(ValueError, match="Missing"):
        select_features(df)


def test_select_features_accepts_exact_cols():
    df = pl.DataFrame({col: [0] for col in FEATURE_COLS})
    result = select_features(df)
    assert result.columns == FEATURE_COLS
