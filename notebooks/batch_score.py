import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def imports():
    import os
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    from loguru import logger

    from groundwork.features import select_features
    from groundwork.io import load_csv, save_csv
    from groundwork.model import FraudModel

    return (
        Path,
        alt,
        logger,
        mo,
        os,
        pl,
        select_features,
        load_csv,
        save_csv,
        FraudModel,
    )


@app.cell
def header(mo):
    mo.md(
        """
        # Fraud Detection — Batch Scoring

        Point this notebook at a claims CSV and a trained model to score every claim
        with a fraud probability. Drag the threshold slider to explore the
        precision/recall trade-off — all summary stats and charts update automatically.
        """
    )
    return


@app.cell
def config(mo, Path, os):
    args = mo.cli_args()
    input_path = Path(
        args.get("input_path") or os.environ.get("INPUT_PATH") or "data/claims.csv"
    )
    model_path = Path(
        args.get("model_path")
        or os.environ.get("MODEL_PATH")
        or "models/fraud_model.txt"
    )
    output_path = Path(
        args.get("output_path")
        or os.environ.get("OUTPUT_PATH")
        or "data/predictions.csv"
    )
    mo.md(
        f"""
        **Config**
        - Input: `{input_path}`
        - Model: `{model_path}`
        - Output: `{output_path}`
        """
    )
    return input_path, model_path, output_path


@app.cell
def threshold_slider(mo):
    threshold = mo.ui.slider(0.1, 0.9, step=0.05, value=0.5, label="Fraud threshold")
    mo.md(
        f"""
        ### Threshold

        A higher threshold flags fewer claims but misses more fraud.
        A lower threshold flags more claims but increases false positives.

        {threshold}
        """
    )
    return (threshold,)


@app.cell
def load_data(input_path, load_csv, logger, mo):
    raw_df = load_csv(input_path)
    mo.md(f"Loaded **{len(raw_df):,}** claims from `{input_path}`.")
    return (raw_df,)


@app.cell
def preprocess(raw_df, select_features):
    features_df = select_features(raw_df)
    return (features_df,)


@app.cell
def load_model(model_path, FraudModel, logger):
    model = FraudModel(model_path)
    return (model,)


@app.cell
def score(model, features_df, mo):
    fraud_scores = model.predict_proba(features_df)
    mo.md(f"Scored **{len(fraud_scores):,}** claims.")
    return (fraud_scores,)


@app.cell
def build_output(raw_df, fraud_scores, threshold, pl):

    output_df = raw_df.with_columns(
        pl.Series("fraud_score", fraud_scores.round(4)),
        pl.Series("predicted_fraud", (fraud_scores >= threshold.value).astype(int)),
    )
    return (output_df,)


@app.cell
def summary(output_df, threshold, mo):
    total = len(output_df)
    flagged = int(output_df["predicted_fraud"].sum())
    flag_rate = flagged / total if total > 0 else 0.0
    mo.md(
        f"""
        ### Summary — threshold = {threshold.value:.2f}

        Drag the threshold slider above to see these numbers update.

        {
            mo.hstack(
                [
                    mo.stat(label="Total claims", value=f"{total:,}"),
                    mo.stat(label="Flagged for review", value=f"{flagged:,}"),
                    mo.stat(label="Flag rate", value=f"{flag_rate:.1%}"),
                ]
            )
        }
        """
    )
    return


@app.cell
def score_chart(output_df, threshold, alt, mo):
    chart = (
        alt.Chart(output_df.select("fraud_score"))
        .mark_bar(opacity=0.7)
        .encode(
            alt.X("fraud_score:Q", bin=alt.Bin(maxbins=40), title="Fraud score"),
            alt.Y("count()", title="Claims"),
        )
        .properties(title="Distribution of fraud scores")
    ) + alt.Chart(alt.Data(values=[{"threshold": threshold.value}])).mark_rule(
        color="red", strokeWidth=2
    ).encode(x="threshold:Q")
    mo.md(f"### Score Distribution\n\n{mo.ui.altair_chart(chart)}")
    return


@app.cell
def predictions_table(output_df, mo):
    mo.md(
        f"""
        ### Predictions (first 100 rows)

        {mo.ui.table(output_df.head(100))}
        """
    )
    return


@app.cell
def save(output_df, output_path, save_csv, logger, mo):
    save_csv(output_df, output_path)
    logger.info("Predictions written to {}", output_path)
    mo.md(
        f"Results written to `{output_path}` — each row has a `fraud_score` "
        f"and `predicted_fraud` flag "
        f"({output_df['predicted_fraud'].mean():.1%} flagged at this threshold)."
    )
    return


@app.cell
def mlflow_log(output_df, threshold, input_path, mo):
    import mlflow

    n_total = len(output_df)
    n_flagged = int(output_df["predicted_fraud"].sum())
    pct_flagged = n_flagged / n_total if n_total > 0 else 0.0

    mlflow.set_experiment("fraud-detection-scoring")
    with mlflow.start_run() as run:
        mlflow.log_param("threshold", threshold.value)
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_metrics(
            {
                "total_claims": n_total,
                "flagged_claims": n_flagged,
                "flag_rate": pct_flagged,
            }
        )
    mo.md(
        f"""
        ### MLflow Run Logged

        Experiment: **fraud-detection-scoring**
        Run ID: `{run.info.run_id}`

        {
            mo.callout(
                mo.md(
                    f"Logged `flag_rate={pct_flagged:.1%}` "
                    f"at threshold `{threshold.value}`."
                ),
                kind="info",
            )
        }
        """
    )
    return


if __name__ == "__main__":
    app.run()
