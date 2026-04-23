"""
Feature pipeline: applies time-based train/val/test and run FeatureEncoder for each
variant. Saves processed feature matrices

Split strat: partition contests by startTimeSeconds (ascending), then assign
problems according to their contest's bucket. This simulate real deployment where
we use data from older contests to predict newer ones
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.encoder import FeatureEncoder
from src.utils import get_logger

logger = get_logger(__name__)

VARIANTS = ["A","B","C"]

_CFG_PATH = Path("configs/model.yaml")

def _load_cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)

def time_based_split(df: pd.DataFrame, cfg:dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labeled problems into train/val/test by contest start time.
    Problems from the same contest belongs to the same category to prevent
    contest leaking.
    """

    split_cfg = cfg["split"]
    train_frac = split_cfg["train_frac"]
    val_frac = split_cfg["val_frac"]

    # Sort contest by start time
    contest_times = (
        df[["contest_id", "contest_start_time"]]
        .drop_duplicates("contest_id")
        .sort_values("contest_start_time")
    )

    n_contests = len(contest_times)
    n_train = int(np.floor(n_contests * train_frac))
    n_val = int(np.floor(n_contests * val_frac))

    train_ids = set(contest_times.iloc[:n_train]["contest_id"])
    val_ids = set(contest_times.iloc[n_train: n_train + n_val]["contest_id"])
    test_ids = set(contest_times.iloc[n_train + n_val:]["contest_id"])

    train_df = df[df["contest_id"].isin(train_ids)].copy()
    val_df = df[df["contest_id"].isin(val_ids)].copy()
    test_df = df[df["contest_id"].isin(test_ids)].copy()

    logger.info(
        "Split — Train: %d problems (%d contests) | Val: %d (%d) | Test: %d (%d)",
        len(train_df), len(train_ids),
        len(val_df), len(val_ids),
        len(test_df), len(test_ids),
    )
    return train_df, val_df, test_df

def build_feature_pipeline(
    labeled_path: str | Path = "data/intermediate/labeled.parquet",
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    cfg: dict | None = None,
) -> None:
    """
    Runs the full feature engineering pipeline for all three variants.
    Saves:
      data/processed/{split}_{variant}.parquet   — X (features) + y (rating)
      models/feature_encoder_{variant}.joblib    — fitted FeatureEncoder
      data/processed/feature_names_{variant}.json
    """
    if cfg is None:
        cfg = _load_cfg()

    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(labeled_path)
    logger.info("Loaded labeled data: %d rows", len(df))

    train_df, val_df, test_df = time_based_split(df, cfg)

    # Save split indices for reference
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split[["problem_key", "contest_id", "rating"]].to_parquet(
            processed_dir / f"split_index_{name}.parquet", index=False
        )

    for variant in VARIANTS:
        logger.info("Building features for variant %s …", variant)

        encoder = FeatureEncoder()
        X_train = encoder.fit_transform(train_df, variant=variant)
        X_val = encoder.transform(val_df, variant=variant)
        X_test = encoder.transform(test_df, variant=variant)

        y_train = train_df["rating"].astype(float).values
        y_val = val_df["rating"].astype(float).values
        y_test = test_df["rating"].astype(float).values

        for split_name, X, y, raw_df in [
            ("train", X_train, y_train, train_df),
            ("val", X_val, y_val, val_df),
            ("test", X_test, y_test, test_df),
        ]:
            out = X.copy()
            out["rating"] = y
            out["problem_key"] = raw_df["problem_key"].values
            out.to_parquet(processed_dir / f"{split_name}_{variant}.parquet", index=False)

        # Save fitted encoder + feature names
        encoder_path = models_dir / f"feature_encoder_{variant}.joblib"
        joblib.dump(encoder, encoder_path)

        feature_names = list(X_train.columns)
        with open(processed_dir / f"feature_names_{variant}.json", "w") as f:
            json.dump(feature_names, f, indent=2)

        logger.info(
            "Variant %s: %d features | train=%d val=%d test=%d",
            variant, len(feature_names), len(X_train), len(X_val), len(X_test),
        )

    logger.info("Feature pipeline complete.")