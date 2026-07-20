"""
Feature-encoder tests: self-contained on tiny synthetic frames (no pipeline
run required). Focus is that train-time and inference-time transforms stay
consistent — that's the property `RatingPredictor` relies on.
"""

import numpy as np
import pandas as pd

from src.features.encoder import ALL_TAGS, FeatureEncoder


def _sample_df(n: int = 5) -> pd.DataFrame:
    tags = [["dp"], ["greedy", "math"], [], ["graphs"], ["totally-fake-tag"]]
    solved_counts = [0, 100, 5000, None, 200]
    indices = ["A", "B", "C1", "D", "E"]
    return pd.DataFrame({
        "problem_index": indices[:n],
        "tags": tags[:n],
        "contest_name": ["Codeforces Round (Div. 2)"] * n,
        "contest_type": ["CF"] * n,
        "contest_start_time": [1700000000 + i * 86400 for i in range(n)],
        "contest_duration_secs": [7200] * n,
        "solved_count": solved_counts[:n],
    })


def test_encoder_produces_consistent_columns():
    train_df = _sample_df(n=5)
    val_df = _sample_df(n=3)

    encoder = FeatureEncoder()
    X_train = encoder.fit_transform(train_df, variant="B")
    X_val = encoder.transform(val_df, variant="B")

    assert list(X_train.columns) == list(X_val.columns)


def test_unseen_tags_do_not_crash_transform():
    df = _sample_df()
    assert "totally-fake-tag" not in ALL_TAGS

    encoder = FeatureEncoder().fit(df)
    X = encoder.transform(df, variant="B")

    # row with only an unknown tag should get zero rarity, not raise
    assert X["tag_rarity_mean"].iloc[-1] == 0.0


def test_log_solved_count_is_finite():
    df = _sample_df()
    encoder = FeatureEncoder().fit(df)
    X = encoder.transform(df, variant="C")

    assert np.isfinite(X["solved_count_log"]).all()
    assert np.isfinite(X["solved_count_raw"]).all()
