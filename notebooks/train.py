import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def imports():
    import os
    from pathlib import Path

    import altair as alt
    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import polars as pl
    from loguru import logger
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from groundwork.features import FEATURE_COLS
    from groundwork.io import save_csv
    return (
        Path, alt, lgb, logger, mo, np, os, pl,
        roc_auc_score, train_test_split, save_csv, FEATURE_COLS,
    )


@app.cell
def header(mo):
    mo.md(
        """
        # Fraud Detection — Model Training

        This notebook generates synthetic auto-insurance claim data, trains a
        LightGBM binary classifier, evaluates it, and saves the model in LightGBM's
        native text format. Every cell is reactive — change a slider and downstream
        cells update automatically.
        """
    )
    return


@app.cell
def config(mo, Path, os):
    args = mo.cli_args()
    data_path = Path(args.get("data_path") or os.environ.get("DATA_PATH") or "data")
    model_path = Path(
        args.get("model_path") or os.environ.get("MODEL_PATH") or "models/fraud_model.txt"
    )
    mo.md(
        f"""
        **Config**
        - Data dir: `{data_path}`
        - Model output: `{model_path}`
        """
    )
    return data_path, model_path


@app.cell
def hyperparams(mo):
    n_estimators = mo.ui.slider(50, 500, step=50, value=100, label="n_estimators")
    num_leaves = mo.ui.slider(15, 63, step=4, value=31, label="num_leaves")
    mo.md(
        f"""
        ### Hyperparameters

        Drag the sliders to tune the model — downstream cells update automatically.

        {mo.hstack([n_estimators, num_leaves])}
        """
    )
    return n_estimators, num_leaves


@app.cell
def generate_data(np, pl, save_csv, data_path, mo):  # noqa: F811
    rng = np.random.default_rng(0)
    n = 5_000
    at_fault = rng.integers(0, 2, n)
    policy_type = rng.integers(0, 3, n)
    claim_amount = rng.exponential(scale=3_000, size=n).round(2)
    num_prev = rng.poisson(lam=1.0, size=n)

    days_to_report = rng.integers(0, 30, n)

    # Coefficients chosen so test AUC ≥ 0.80 at default hyperparams
    fraud_log_odds = (
        -5.5
        + 2.5 * at_fault
        + 0.00015 * claim_amount
        + 1.2 * num_prev
        - 0.2 * policy_type
        + 0.03 * days_to_report
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_log_odds))
    is_fraud = rng.binomial(1, fraud_prob).astype(int)

    claims = pl.DataFrame(
        {
            "claim_amount": claim_amount,
            "policy_age_months": rng.integers(1, 120, n).tolist(),
            "customer_age_years": rng.integers(18, 80, n).tolist(),
            "num_previous_claims": num_prev.tolist(),
            "vehicle_age_years": rng.integers(0, 20, n).tolist(),
            "days_to_report": days_to_report.tolist(),
            "num_witnesses": rng.integers(0, 5, n).tolist(),
            "at_fault": at_fault.tolist(),
            "policy_type": policy_type.tolist(),
            "is_fraud": is_fraud.tolist(),
        }
    )

    data_path.mkdir(parents=True, exist_ok=True)
    save_csv(claims, data_path / "claims.csv")
    mo.md(
        f"Generating **{n:,}** synthetic auto-insurance claims. "
        f"Fraud rate: **{is_fraud.mean():.1%}**"
    )
    return claims,


@app.cell
def eda(claims, mo):
    fraud_rate = claims["is_fraud"].mean()
    mo.md(
        f"""
        ### Data Explorer

        {mo.ui.table(claims.head(20))}

        **Class balance**: {fraud_rate:.1%} fraud ({int(claims["is_fraud"].sum())} / {len(claims)} claims).
        A stratified split preserves this ratio in train and test sets.
        """
    )
    return fraud_rate,


@app.cell
def split(claims, np, train_test_split, FEATURE_COLS):
    X = claims.select(FEATURE_COLS).to_numpy()
    y = claims["is_fraud"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


@app.cell
def train(lgb, n_estimators, num_leaves, X_train, y_train, logger, mo):
    mo.md("Training LightGBM with your chosen settings...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=n_estimators.value,
        num_leaves=num_leaves.value,
        min_child_samples=20,
        reg_alpha=0.05,
        objective="binary",
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    logger.info(
        "Trained LGBMClassifier — n_estimators={}, num_leaves={}",
        n_estimators.value,
        num_leaves.value,
    )
    return lgb_model,


@app.cell
def evaluate(lgb_model, X_test, y_test, roc_auc_score, mo):
    y_pred = lgb_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    mo.md(
        f"""
        ### Evaluation

        AUC-ROC measures how well the model separates fraud from legitimate claims.
        1.0 is perfect; 0.5 is random. ≥ 0.80 is production-ready for this domain.

        {mo.callout(mo.md(f"**AUC-ROC: {auc:.4f}**"), kind="success")}
        """
    )
    return auc, y_pred


@app.cell
def feature_importance(lgb_model, alt, mo, FEATURE_COLS):
    importances = lgb_model.feature_importances_
    chart_df = {"feature": FEATURE_COLS, "importance": importances.tolist()}
    chart = (
        alt.Chart(alt.Data(values=[
            {"feature": f, "importance": i}
            for f, i in zip(FEATURE_COLS, importances.tolist())
        ]))
        .mark_bar()
        .encode(
            x=alt.X("importance:Q"),
            y=alt.Y("feature:N", sort="-x"),
            tooltip=["feature:N", "importance:Q"],
        )
        .properties(title="Feature Importance (split count)")
    )
    mo.md(
        f"""
        ### Feature Importance

        Which claim characteristics are most predictive of fraud?

        {mo.ui.altair_chart(chart)}
        """
    )
    return


@app.cell
def save_model(lgb_model, model_path, mo, logger):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    lgb_model.booster_.save_model(str(model_path))
    logger.info("Model saved to {}", model_path)
    mo.md(
        f"""
        ### Model Saved

        Model written to `{model_path}` in LightGBM's native text format — human-readable,
        no pickle, no extra dependencies. Load it at inference time with
        `lgb.Booster(model_file=path)` using only the `lightgbm` package.
        """
    )
    return


if __name__ == "__main__":
    app.run()
