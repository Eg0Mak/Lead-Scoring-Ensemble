import os
import sys
from pathlib import Path

import joblib
import pandas as pd

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import EnsembleModel   # model/model.py

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "ensemble_model.joblib"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

_bundle = joblib.load(MODEL_PATH)
model = _bundle["model"]
feature_cols = _bundle["feature_cols"]
cat_features = _bundle["cat_features"]


def predict(df: pd.DataFrame):
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = "missing" if col in cat_features else 0

    # категориальные
    for col in cat_features:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("missing")
                .astype(str)
                .astype("category")
            )

    # числовые
    numeric_cols = [c for c in feature_cols if c not in cat_features]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return model.predict_proba(df[feature_cols])