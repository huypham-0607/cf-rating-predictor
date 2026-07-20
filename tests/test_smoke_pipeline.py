"""
End-to-end smoke test on a tiny synthetic dataset: feature encoding -> train
-> predict -> save artifact. Uses Ridge (cheap) rather than LightGBM/XGBoost —
this checks the pipeline wiring, not model quality.
"""

import numpy as np
import pandas as pd
import joblib

from src.features.encoder import FeatureEncoder
from src.models.baseline import build_ridge


def _tiny_dataset(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    indices = ["A", "B", "C", "D", "E"]
    return pd.DataFrame({
        "problem_index": [indices[i % len(indices)] for i in range(n)],
        "tags": [["dp"] if i % 2 == 0 else ["greedy"] for i in range(n)],
        "contest_name": ["Codeforces Round (Div. 2)"] * n,
        "contest_type": ["CF"] * n,
        "contest_start_time": [1700000000 + i * 86400 for i in range(n)],
        "contest_duration_secs": [7200] * n,
        "solved_count": rng.integers(10, 5000, size=n),
        "rating": rng.integers(800, 3500, size=n).astype(float),
    })


def test_training_pipeline_smoke(tmp_path):
    df = _tiny_dataset()
    train_df, test_df = df.iloc[:15], df.iloc[15:]

    encoder = FeatureEncoder()
    X_train = encoder.fit_transform(train_df, variant="A")
    X_test = encoder.transform(test_df, variant="A")

    model = build_ridge(alpha=10.0)
    model.fit(X_train, train_df["rating"].values)

    preds = model.predict(X_test)
    assert np.isfinite(preds).all()

    artifact_path = tmp_path / "ridge_smoke.joblib"
    joblib.dump(model, artifact_path)
    assert artifact_path.exists()
