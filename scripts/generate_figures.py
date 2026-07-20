"""
Generates the two report figures from already-computed results:
  reports/figures/model_comparison.png  — test MAE by model x feature-variant
  reports/figures/error_by_rating.png   — best model's mean abs error by rating band

Reads reports/model_comparison.csv (produced by evaluate_all_models) and the
saved best-model artifact + its test split. Run after scripts/run_pipeline.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import RATING_BANDS
from src.utils import get_logger

logger = get_logger("generate_figures")

# Palette (light mode) — see dataviz skill's reference/palette.md
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
VARIANT_COLORS = {"A": "#2a78d6", "B": "#008300", "C": "#e87ba4"}  # categorical slots 1-3
SEQUENTIAL_BLUE = "#2a78d6"

MODEL_LABELS = {"mean": "Mean", "median": "Median", "ridge": "Ridge", "lgbm": "LightGBM", "xgb": "XGBoost"}
MODEL_ORDER = ["mean", "median", "ridge", "lgbm", "xgb"]
VARIANT_ORDER = ["A", "B", "C"]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def plot_model_comparison(reports_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(reports_dir / "model_comparison.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    _style_axes(ax)

    n_models = len(MODEL_ORDER)
    n_variants = len(VARIANT_ORDER)
    group_width = 0.8
    bar_width = group_width / n_variants
    x = np.arange(n_models)

    for i, variant in enumerate(VARIANT_ORDER):
        sub = df[df["variant"] == variant].set_index("model").reindex(MODEL_ORDER)
        offset = (i - (n_variants - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, sub["MAE"], width=bar_width * 0.92,
            color=VARIANT_COLORS[variant], label=f"Variant {variant}", zorder=3,
        )
        for bar, mae in zip(bars, sub["MAE"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{mae:.0f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], color=INK_PRIMARY)
    ax.set_ylabel("Test MAE (rating points)", color=INK_SECONDARY)
    ax.set_title("Test MAE by model and feature regime", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_error_by_rating(
    models_dir: Path, processed_dir: Path, reports_dir: Path, out_path: Path,
    model_name: str = "lgbm", variant: str = "C",
) -> None:
    test_df = pd.read_parquet(processed_dir / f"test_{variant}.parquet")
    y_true = test_df["rating"].values.astype(float)
    X_test = test_df.drop(columns=["rating", "problem_key"], errors="ignore").astype(float)

    model = joblib.load(models_dir / f"{model_name}_{variant}.joblib")
    y_pred = model.predict(X_test)
    abs_err = np.abs(y_true - y_pred)

    labels, means, counts = [], [], []
    for lo, hi, label in RATING_BANDS:
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() == 0:
            continue
        labels.append(label)
        means.append(abs_err[mask].mean())
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    _style_axes(ax)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, width=0.6, color=SEQUENTIAL_BLUE, zorder=3)
    for bar, mean, n in zip(bars, means, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{mean:.0f}\n(n={n})", ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK_PRIMARY, rotation=30, ha="right")
    ax.set_ylabel("Mean absolute error (rating points)", color=INK_SECONDARY)
    ax.set_title(
        f"Error by rating band — {MODEL_LABELS[model_name]} variant {variant} (test set)",
        color=INK_PRIMARY, fontsize=13, loc="left",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(reports_dir, figures_dir / "model_comparison.png")
    plot_error_by_rating(Path("models"), Path("data/processed"), reports_dir, figures_dir / "error_by_rating.png")


if __name__ == "__main__":
    main()
