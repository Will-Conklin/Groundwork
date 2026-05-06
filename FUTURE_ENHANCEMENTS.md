# Future Enhancements

Tracked separately from the README to keep the main docs focused on what works today.

## ONNX export

Export the trained LightGBM model to ONNX for framework-agnostic inference (e.g., C# scoring service, ONNX Runtime in a sidecar).

```python
# Requires: pip install onnxmltools skl2onnx
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
onnx_model = convert_lightgbm(lgb_model.booster_, initial_types=initial_type)
with open("models/fraud_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

Add a `groundwork/model_onnx.py` that wraps `onnxruntime.InferenceSession` with the same `predict_proba(df)` interface as `FraudModel`.

## Real data integration

Replace the synthetic DGP in `notebooks/train.py` with a `groundwork/io.py` loader that reads from a real claims database. Consider adding `connectorx` for fast SQL→Polars without pandas.

## Feature engineering

Add a `groundwork/transforms.py` module for:
- Claim amount log-transform (reduce right-skew)
- Policy age bucketing (new vs. established)
- Days-to-report categorical (same-day / within-week / delayed)

## Hyperparameter search

Wrap the `hyperparams` cell with Optuna or Ax for automated tuning rather than manual slider exploration.

## SHAP explanations

Add a `shap_explain` cell after `feature_importance` using `shap.TreeExplainer` for per-claim explanation output appended to `predictions.csv`.

## Monitoring

Add a `notebooks/monitor.py` marimo notebook that reads prediction logs and tracks score distribution drift over time (PSI, KS test).
