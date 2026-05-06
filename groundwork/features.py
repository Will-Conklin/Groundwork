import polars as pl

FEATURE_COLS: list[str] = [
    "claim_amount",
    "policy_age_months",
    "customer_age_years",
    "num_previous_claims",
    "vehicle_age_years",
    "days_to_report",
    "num_witnesses",
    "at_fault",
    "policy_type",
]


def select_features(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df.select(FEATURE_COLS)
