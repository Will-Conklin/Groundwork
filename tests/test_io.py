import polars as pl
import pytest

from groundwork.io import load_csv, save_csv


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "claim_amount": [100.0, 200.0, 300.0],
            "label": [0, 1, 0],
        }
    )


def test_save_and_load_roundtrip(tmp_path, sample_df):
    path = tmp_path / "test.csv"
    save_csv(sample_df, path)
    result = load_csv(path)
    assert result.shape == sample_df.shape
    assert result.columns == sample_df.columns


def test_load_csv_returns_dataframe(tmp_path, sample_df):
    path = tmp_path / "test.csv"
    save_csv(sample_df, path)
    result = load_csv(path)
    assert isinstance(result, pl.DataFrame)


def test_save_csv_creates_file(tmp_path, sample_df):
    path = tmp_path / "out.csv"
    assert not path.exists()
    save_csv(sample_df, path)
    assert path.exists()


def test_load_csv_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nonexistent.csv")
