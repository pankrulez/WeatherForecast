# src/predict.py
import joblib
import pandas as pd
from typing import Any, List

from .config import MODEL_PATH

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _get_expected_columns_from_model(model) -> List[str]:
    """
    Extract the full list of feature column names that the
    ColumnTransformer expects at transform time.
    """
    preprocess = model.named_steps["preprocess"]
    expected: List[str] = []

    for name, transformer, cols in preprocess.transformers_:
        if cols is None or cols == "drop":
            continue
        # cols can be list-like or array-like
        if isinstance(cols, list):
            expected.extend(cols)
        else:
            try:
                expected.extend(list(cols))
            except TypeError:
                pass

    # Remove duplicates while preserving order
    seen = set()
    ordered = []
    for c in expected:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _align_input_to_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure df has all columns the model expects.
    Missing columns are added as NaN (to be handled by imputers).
    """
    expected_cols = _get_expected_columns_from_model(model)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only expected columns and in correct order
    return df[expected_cols]


def predict_rain(input_data: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    df = pd.DataFrame([input_data])

    df_aligned = _align_input_to_model(df, model)

    proba = model.predict_proba(df_aligned)[0, 1]
    pred = model.predict(df_aligned)[0]

    return {
        "prediction": int(pred),
        "probability": float(proba),
    }