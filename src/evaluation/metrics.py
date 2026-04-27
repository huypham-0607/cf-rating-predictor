"""
Evaluation metrics and model comparison report generation.

Primary metric: MAE (intuitive — in the same units as rating).
Secondary: RMSE, R², within-100, within-200 accuracy.
Breakdowns by rating band, division, problem index.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.encoder import parse_division
from src.utils import get_logger

logger = get_logger(__name__)

RATING_BANDS = [
    (0, 1200, "<1200"),
    (1200, 1400, "1200-1399"),
    (1400, 1599, "1400-1599"),
    (1600, 1899, "1600-1899"),
    (1900, 2099, "1900-2099"),
    (2100, 2299, "2100-2299"),
    (2300, 2399, "2300-2399"),
    (2400, 2599, "2400-2599"),
    (2600, 2999, "2600-2999"),
    (3000, 9999, "3000+"),
]

MODEL_NAMES = ["mean", "median", "ridge", "lgbm", "xgb"]
VARIANTS = ["A", "B", "C"]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns:
        - Mean Absolute Error
        - Root Mean Squared Error
        - R2
        - Median Absolute Error
        - Fraction of predictions within 100/200 rating of true value
    """
    abs_err = np.abs(y_true - y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MedAE": round(float(np.median(abs_err)), 2),
        "within_100": round(float(np.mean(abs_err <= 100)), 4),
        "within_200": round(float(np.mean(abs_err <= 200)), 4),
    }

def evaluate_all_models(
    processed_dir: str | Path = "data/processed",
    intermediate_dir: str | Path = "data/intermediate",
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    intermediate_dir = Path(intermediate_dir)
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load the raw labeled data (for breakdown context columns)
    labeled = pd.read_parquet(intermediate_dir / "labeled.parquet")

    all_results = []
    error_rows = []

    for variant in VARIANTS:
        # Load test split features
        test_df = pd.read_parquet(processed_dir / f"test_{variant}.parquet")
        y_test = test_df["rating"].values.astype(float)
        problem_keys = test_df["problem_key"].values
        X_test = test_df.drop(columns=["rating", "problem_key"], errors="ignore").astype(float)

        # Attach context from labeled df for breakdown analysis
        context = labeled.set_index("problem_key").loc[
            problem_keys,
            ["contest_name", "contest_type", "problem_index", "tags"]
        ].copy()
        context["division"] = context.apply(
            lambda r: parse_division(r["contest_name"], r["contest_type"]), axis=1
        )

        for model_name in MODEL_NAMES:
            artifact = models_dir / f"{model_name}_{variant}.joblib"
            if not artifact.exists():
                logger.warning("Missing: %s — skipping", artifact)
                continue

            model = joblib.load(artifact)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            metrics["model"] = model_name
            metrics["variant"] = variant
            all_results.append(metrics)

            # Collect worst errors for the best-performing tree models
            if model_name in ("lgbm", "xgb") and variant == "C":
                abs_err = np.abs(y_test - y_pred)
                worst_idx = np.argsort(abs_err)[-20:]
                for i in worst_idx:
                    error_rows.append({
                        "model": model_name,
                        "variant": variant,
                        "problem_key": problem_keys[i],
                        "true_rating": int(y_test[i]),
                        "pred_rating": int(round(y_pred[i])),
                        "abs_error": round(float(abs_err[i]), 1),
                    })

    results_df = pd.DataFrame(all_results)

    # Save comparison CSV
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    # Write Markdown report
    _write_comparison_report(results_df, reports_dir)

    # Save error analysis
    if error_rows:
        pd.DataFrame(error_rows).sort_values("abs_error", ascending=False).to_csv(
            reports_dir / "error_analysis.csv", index=False
        )

    # Feature importance for tree models
    _write_feature_importance(models_dir, processed_dir, reports_dir)

    # Update best_model.json with test metrics
    _update_best_model(results_df, models_dir)

    logger.info("Evaluation complete. Reports saved to %s", reports_dir)
    return results_df

def _write_comparison_report(df: pd.DataFrame, reports_dir: Path) -> None:
    lines = [
        "# Model Comparison Report",
        "",
        "Evaluated on held-out **test set** (most recent 15% of contests by start time).",
        "",
        "## All Models",
        "",
        "| Model | Variant | MAE | RMSE | R² | Within±100 | Within±200 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.sort_values("MAE").iterrows():
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['MAE']} | {r['RMSE']} | {r['R2']} "
            f"| {r['within_100']:.2%} | {r['within_200']:.2%} |"
        )

    # Best model callout
    best = df.sort_values("MAE").iloc[0]
    lines += [
        "",
        f"**Best model:** `{best['model']}` variant **{best['variant']}** — MAE={best['MAE']}, "
        f"Within±100={best['within_100']:.1%}, Within±200={best['within_200']:.1%}",
        "",
    ]

    with open(reports_dir / "model_comparison.md", "w") as f:
        f.write("\n".join(lines))
    logger.info("Saved model_comparison.md")

def _write_feature_importance(models_dir: Path, processed_dir: Path, reports_dir: Path) -> None:
    """
    Write a report for feature importance for tree-based models.
    """
    for variant in VARIANTS:
        fi_path = processed_dir / f"feature_names_{variant}.json"
        if not fi_path.exists():
            continue
        with open(fi_path) as f:
            feature_names = json.load(f)

        for model_name in ("lgbm", "xgb"):
            artifact = models_dir / f"{model_name}_{variant}.joblib"
            if not artifact.exists():
                continue
            model = joblib.load(artifact)

            # Get importances (works for both LightGBM and XGBoost)
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                continue

            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False)

            fi_df.to_csv(reports_dir / f"feature_importance_{model_name}_{variant}.csv", index=False)


def _update_best_model(results_df: pd.DataFrame, models_dir: Path) -> None:
    best = results_df.sort_values("MAE").iloc[0]
    best_path = models_dir / "best_model.json"
    with open(best_path) as f:
        meta = json.load(f)
    meta["test_MAE"] = best["MAE"]
    meta["test_within_100"] = best["within_100"]
    meta["test_within_200"] = best["within_200"]
    with open(best_path, "w") as f:
        json.dump(meta, f, indent=2)