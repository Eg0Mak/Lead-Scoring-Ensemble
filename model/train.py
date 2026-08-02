from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import EnsembleModel
from src.preprocessing import add_business_features, build_event_features_advanced

MODEL_PATH = Path(__file__).resolve().parent / "ensemble_model.joblib"


if __name__ == "__main__":
    train_df = pd.read_csv(PROJECT_ROOT / "data/train.csv")
    events_df = pd.read_csv(PROJECT_ROOT / "data/events.csv")

    train_df = add_business_features(train_df)
    event_features = build_event_features_advanced(train_df, events_df)
    train_df = train_df.merge(event_features, on="lead_id", how="left")

    id_cols = ["lead_id", "user_id"]
    time_cols = ["assignment_ts", "assignment_date"]
    target_col = "target"

    feature_cols = [
        c for c in train_df.columns
        if c not in id_cols + time_cols + [target_col]
    ]

    cat_features = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(train_df[c])
    ]

    for col in cat_features:
        train_df[col] = (
            train_df[col]
            .fillna("missing")
            .astype(str)
            .astype("category")
        )

    cb_params_final = dict(
        iterations=713,
        learning_rate=0.11197039245696323,
        depth=4,
        l2_leaf_reg=12.947138543838287,
        max_ctr_complexity=4,
        auto_class_weights="Balanced",
    )

    lgb_params_final = dict(
        n_estimators=713,
        learning_rate=0.11197039245696323,
        max_depth=4,
        num_leaves=52,
        min_child_samples=19,
        class_weight="balanced",
    )

    xgb_params_final = dict(
        n_estimators=713,
        learning_rate=0.11197039245696323,
        max_depth=4,
        reg_lambda=0.8808020704774955,
        min_child_weight=6,
        scale_pos_weight=2.0003405327022197,
        tree_method="hist",
        enable_categorical=True,
    )

    ENSEMBLE_SEEDS = [42, 43, 44]

    final_model = EnsembleModel(
        cb_params_final,
        lgb_params_final,
        xgb_params_final,
        cat_features,
        ENSEMBLE_SEEDS,
    )
    final_model.fit(train_df[feature_cols], train_df[target_col])

    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "cat_features": cat_features,
        },
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")