"""
CatBoost inference for food delivery ETA (minutes).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from src.preprocessing import prepare_model_features, preprocess_input

# Model path relative to project root (src/ -> project/)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "catboost_model.cbm"

_model: CatBoostRegressor | None = None


def _load_model() -> CatBoostRegressor:
    """Load the CatBoost model once and reuse it for later predictions."""
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = CatBoostRegressor()
        _model.load_model(str(MODEL_PATH))
    return _model


def predict_eta(data: pd.DataFrame) -> float | list[float]:
    """
    Predict delivery time in minutes for one or more orders.

    Parameters
    ----------
    data : pd.DataFrame
        Raw order rows (see preprocessing.py for required columns).

    Returns
    -------
    float
        Single prediction when one row is passed.
    list[float]
        One prediction per row for batch input.
    """
    processed = preprocess_input(data)
    features = prepare_model_features(processed)

    model = _load_model()
    predictions = model.predict(features)

    if len(predictions) == 1:
        return float(predictions[0])
    return predictions.tolist()
