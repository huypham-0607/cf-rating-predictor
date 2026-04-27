"""
Inference pipeline: single-problem and batch prediction.

Auto-selects feature variant:
  - Variant C if solved_count is provided (post-contest mode)
  - Variant B otherwise (cold-start mode)

Loading is lazy: models and encoders are loaded once on first predict() call.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time as _time

import joblib
import numpy as np
import pandas as pd
import math

from src.utils import get_logger

logger = get_logger(__name__)

RATING_BANDS = [
    (0, 1200, "<1200 (Newbie)"),
    (1200, 1400, "1200-1399 (Pupil)"),
    (1400, 1599, "1400-1599 (Specialist)"),
    (1600, 1899, "1600-1899 (Expert)"),
    (1900, 2099, "1900-2099 (Candidate Master)"),
    (2100, 2299, "2100-2299 (Master)"),
    (2300, 2399, "2300-2399 (International Master)"),
    (2400, 2599, "2400-2599 (Grandmaster)"),
    (2600, 2999, "2600-2999 (International Grandmaster)"),
    (3000, 9999, "3000+ (Legendary Grandmaster)"),
]

@dataclass
class ProblemInput:
    problem_index: str                                  # "A", "B", "C", etc.
    tags: list[str] = field(default_factory=list)
    contest_division: str = "div2"                      # div1/div2/div3/div4/educational/global/icpc/other
    contest_type: str = "CF"                            # CF / ICPC / IOI
    contest_year: int = 2023
    contest_duration_hours: float = 2.0
    solved_count: Optional[int] = None                  # None = cold-start (Variant B)

@dataclass
class PredictionOutput:
    predicted_rating: int
    predicted_rating_raw: float
    rating_band: str
    variant_used: str
    top_features: list[tuple[str, float]]    # (feature_name, importance)
    is_cold_start: bool

class RatingPredictor:
    def __init__(
        self,
        models_dir: str | Path = "models",
        processed_dir: str | Path = "data/processed",
    ):
        self.models_dir = Path(models_dir)
        self.processed_dir = Path(processed_dir)
        self._loaded: dict = {}   # cache: variant → (encoder, model, feature_names)
        self._best_model_name = self._read_best_model_name()

    def predict(self, problem: ProblemInput) -> PredictionOutput:
        variant = "C" if problem.solved_count is not None else "B"
        encoder, model, feature_names = self._load(variant)

        row = self._input_to_row(problem)
        df = pd.DataFrame([row])
        X = encoder.transform(df, variant=variant)

        # Align columns to match training features
        X = X.reindex(columns=feature_names, fill_value=0)

        raw_pred = float(model.predict(X.values)[0])
        rounded = int(math.floor(raw_pred / 100) * 100)
        rounded = max(800, min(3500, rounded))

        band = next((label for lo, hi, label in RATING_BANDS if lo <= raw_pred < hi), "Unknown")

        top_features = self._top_features(model, feature_names)

        return PredictionOutput(
            predicted_rating=rounded,
            predicted_rating_raw=raw_pred,
            rating_band=band,
            variant_used=variant,
            top_features=top_features,
            is_cold_start=problem.solved_count is None,
        )

    def predict_batch(self, problems: list[ProblemInput]) -> list[PredictionOutput]:
        return [self.predict(p) for p in problems]
    
    # == Internals ===============================================================================

    def _load(self, variant: str) -> tuple:
        if variant in self._loaded:
            return self._loaded[variant]

        model_name = self._best_model_name
        encoder_path = self.models_dir / f"feature_encoder_{variant}.joblib"
        model_path = self.models_dir / f"{model_name}_{variant}.joblib"
        names_path = self.processed_dir / f"feature_names_{variant}.json"

        for p in (encoder_path, model_path, names_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    "Run the full pipeline first: python scripts/run_pipeline.py"
                )

        encoder = joblib.load(encoder_path)
        model = joblib.load(model_path)
        with open(names_path) as f:
            feature_names = json.load(f)

        self._loaded[variant] = (encoder, model, feature_names)
        logger.info("Loaded %s_%s for inference", model_name, variant)
        return self._loaded[variant]

    def _read_best_model_name(self) -> str:
        best_path = self.models_dir / "best_model.json"
        if best_path.exists():
            with open(best_path) as f:
                return json.load(f)["model"]
        return "lgbm"  # sensible default

    @staticmethod
    def _input_to_row(p: ProblemInput) -> dict:
        """Convert ProblemInput to a dict that mimics the canonical DataFrame row."""
        division_map = {
            "div1": "Codeforces Round (Div. 1)",
            "div2": "Codeforces Round (Div. 2)",
            "div3": "Codeforces Round (Div. 3)",
            "div4": "Codeforces Round (Div. 4)",
            "div1+2": "Codeforces Round (Div. 1 + Div. 2)",
            "educational": "Educational Codeforces Round",
            "global": "Codeforces Global Round",
            "icpc": "ICPC",
        }
        contest_name = division_map.get(p.contest_division, "Other")
        # Synthesise a fake start timestamp in the given year
        start_ts = int(_time.mktime((p.contest_year, 6, 1, 12, 0, 0, 0, 0, 0)))
        return {
            "problem_index": p.problem_index,
            "tags": [t.strip().lower() for t in p.tags],
            "contest_name": contest_name,
            "contest_type": p.contest_type,
            "contest_start_time": start_ts,
            "contest_duration_secs": int(p.contest_duration_hours * 3600),
            "solved_count": p.solved_count if p.solved_count is not None else 0,
        }

    @staticmethod
    def _top_features(model, feature_names: list[str], n: int = 5) -> list[tuple[str, float]]:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return []
        idx = np.argsort(importances)[::-1][:n]
        return [(feature_names[i], round(float(importances[i]), 4)) for i in idx]