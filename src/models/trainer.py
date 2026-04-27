"""
Tree ensemble model builders and the full training loop.

Trains 9 model artifacts = 3 variants (A/B/C) x 3 model types (ridge/lgbm/xgb).
Each artifact is saved as models/{model_name}_{variant}.joblib with a JSON sidecar
containing training metadata and validation MAE.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.models.baseline import MeanPredictor, MedianPredictor, build_ridge
from src.utils import get_logger

logger = get_logger(__name__)

_CFG_PATH = Path("configs/model.yaml")
VARIANTS = ["A", "B", "C"]
MODEL_NAMES = ["mean", "median", "ridge", "lgbm", "xgb"]

def _load_cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)

def build_lgbm(cfg: dict) -> LGBMRegressor:
    p = cfg["models"]["lgbm"]
    return LGBMRegressor(
        n_estimators = p["n_estimators"],
        learning_rate = p["learning_rate"],
        max_depth = p["max_depth"],
        num_leaves = p["num_leaves"],
        min_child_samples = p["min_child_samples"],
        subsample = p["subsample"],
        colsample_bytree = p["colsample_bytree"],
        random_state = cfg["random_seed"],
        verbose = -1,
    )

def build_xgb(cfg: dict) -> XGBRegressor:
    p = cfg["models"]["xgb"]
    return XGBRegressor(
        n_estimators = p["n_estimators"],
        learning_rate = p["learning_rate"],
        max_depth = p["max_depth"],
        subsample = p["subsample"],
        colsample_bytree = p["colsample_bytree"],
        tree_method = p["tree_method"],
        random_state = cfg["random_seed"],
        verbosity = 0,
    )


def _load_split(variant: str, split: str, processed_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Take directory to processed data, variant and split, and return feature matrix and ratings"""
    df = pd.read_parquet(processed_dir / f"{split}_{variant}.parquet")
    y = df["rating"].values.astype(float)
    X = df.drop(columns=["rating", "problem_key"], errors="ignore").astype(float)
    return X, y

def train_all_models(
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Trains all 9 model artifacts. Returns a DataFrame with val MAE for each.
    Also writes models/best_model.json pointing to the lowest val-MAE model.
    """
    if cfg is None:
        cfg = _load_cfg()

    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for variant in VARIANTS:
        logger.info("=== Variant %s ===", variant)
        X_train, y_train = _load_split(variant, "train", processed_dir)
        X_val, y_val = _load_split(variant, "val", processed_dir)

        model_builders = {
            "mean": MeanPredictor(),
            "median": MedianPredictor(),
            "ridge": build_ridge(alpha=cfg["models"]["ridge"]["alpha"]),
            "lgbm": build_lgbm(cfg),
            "xgb": build_xgb(cfg),
        }

        for model_name, model in model_builders.items():
            logger.info("  Training %s_%s …", model_name, variant)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_preds)

            # Save artifact
            artifact_path = models_dir / f"{model_name}_{variant}.joblib"
            joblib.dump(model, artifact_path)

            # Save sidecar metadata
            meta = {
                "model": model_name,
                "variant": variant,
                "train_date": datetime.now(timezone.utc).isoformat(),
                "num_train_samples": len(X_train),
                "num_val_samples": len(X_val),
                "num_features": X_train.shape[1],
                "val_mae": round(float(val_mae), 2),
            }
            with open(models_dir / f"{model_name}_{variant}.json", "w") as f:
                json.dump(meta, f, indent=2)

            results.append(meta)
            logger.info("    val MAE = %.1f", val_mae)

    results_df = pd.DataFrame(results).sort_values("val_mae")

    # Write best model pointer
    best = results_df.iloc[0]
    with open(models_dir / "best_model.json", "w") as f:
        json.dump({"model": best["model"], "variant": best["variant"], "val_mae": best["val_mae"]}, f, indent=2)
    logger.info("Best model: %s_%s (val MAE=%.1f)", best["model"], best["variant"], best["val_mae"])

    return results_df
