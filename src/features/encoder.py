"""
Feature encoder: derives structured features columns from original DataFrame.

Produces three feature variants:
    A - metadata only (index, division, year)
    B - A + tags (multi-hot, tag counts, tag rarity)
    C - B + solved_count features (raw + log-transformed)

Output is a DataFrame with one col per feature.
"""

import re
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd

# == Known Codeforces tags (as of 2025) =============================================
ALL_TAGS: list[str] = [
    "2-sat", "binary search", "bitmasks", "brute force", "chinese remainder theorem",
    "combinatorics", "constructive algorithms", "data structures", "dfs and similar",
    "divide and conquer", "dp", "dsu", "expression parsing", "fft", "flows",
    "games", "geometry", "graph matchings", "graphs", "greedy", "hashing",
    "implementation", "interactive", "math", "matrices", "meet-in-the-middle",
    "number theory", "probabilities", "schedules", "shortest paths", "sortings",
    "string suffix structures", "strings", "ternary search", "trees",
    "two pointers",
]

# == Advanced tags (picked by yours truly) ==========================================
ADVANCED_TAGS: frozenset[str] = frozenset([
    "2-sat", "chinese remainder theorem", "expression parsing", "fft", "flows",
    "games", "graph matchings", "matrices", "meet-in-the-middle",
    "string suffix structures",
])

DIVISIONS = ["div1", "div2", "div3", "div4", "div1+2", "educational", "global", "icpc", "other"]



# == Division parsing ==============================================================

def parse_division(contest_name: str, contest_type: str = "CF") -> str:
    if not isinstance(contest_name, str):
        return "other"
    n = contest_name.lower()
    if re.search(r"div\.?\s*1\s*\+\s*div\.?\s*2|div\.?\s*1\+2", n):
        return "div1+2"
    if re.search(r"div\.?\s*1", n):
        return "div1"
    if re.search(r"div\.?\s*2", n):
        return "div2"
    if re.search(r"div\.?\s*3", n):
        return "div3"
    if re.search(r"div\.?\s*4", n):
        return "div4"
    if "educational" in n:
        return "educational"
    if "global" in n:
        return "global"
    if contest_type == "ICPC":
        return "icpc"
    return "other"

def parse_index_numeric(index: str) -> int:
    """Convert problem index to 1-based integer. A1/A2 → 1; unknown → 0."""
    if not isinstance(index, str):
        return 0
    base = re.match(r"^([A-Za-z]+)", index)
    if base:
        base = base.group(1).upper()
        cur = 0
        for c in base:
            cur = cur*26 + (ord(c)-65)
        return cur + 1
    return 0

# == Encoder class =================================================================

class FeatureEncoder:
    """
    
    """

    def __init__(self):
        self._tag_freq: dict[str, float] = {} # populated by fit() call
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """
        Compute tag frequency
        """
        all_tags_flat = [t for tags in df["tags"] for t in tags]
        n = len(df)
        counts = Counter(all_tags_flat)
        self._tag_freq = {tag: (counts[tag] / n) for tag in ALL_TAGS}
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, variant: str = "B") -> pd.DataFrame:
        """
        variant: "A", "B", "C"
        Returns a feature DataFrame aligned with df's index
        """

        assert self._fitted, "Call fit() before transform()"
        assert variant in ("A", "B", "C"), f"Unknown variant: {variant}"

        feats = self._metadata_features(df)

        if variant in ("B", "C"):
            feats = pd.concat([feats, self._tag_features(df)], axis = 1)
        
        if variant == "C":
            feats = pd.concat([feats, self._stats_features(df)], axis = 1)

        return feats

    def fit_transform(self, df: pd.DataFrame, variant: str = "B") -> pd.DataFrame:
        return self.fit(df).transform(df, variant)

    # == Feature families =================================================================

    def _metadata_features(self, df:pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # Problem index
        out["index_numeric"] = df["problem_index"].apply(parse_index_numeric)
        out["is_opener"] = (out["index_numeric"] == 1).astype(int)
        out["is_late_problem"] = (out["index_numeric"] >= 5).astype(int)

        # Contest divison (one-hot)
        divisions = df.apply(lambda r: parse_division(r.get("contest_name",""), r.get("contest_type","")), axis=1)
        for div in DIVISIONS:
            out[f"div_{div}"] = (divisions == div).astype(int)

        # Contest type
        out["is_cf_type"] = (df["contest_type"] == "CF").astype(int)
        out["is_icpc_type"] = (df["contest_type"] == "ICPC").astype(int)

        # Contest era
        out["contest_year"] = df["contest_start_time"].apply(_ts_to_year)
        out["contest_duration_hours"] = (
            pd.to_numeric(df["contest_duration_secs"], errors="coerce").fillna(0) / 3600
        )

        return out

    def _tag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # Multi-hot
        for tag in ALL_TAGS:
            col = "tag_" + tag.replace(" ", "_").replace("-", "_")
            out[col] = df["tags"].apply(lambda tags, t=tag: int(t in tags))

        # Tag counts
        out["num_tags"] = df["tags"].apply(len)
        out["advanced_tag_count"] = df["tags"].apply(lambda tags: sum(1 for t in tags if t in ADVANCED_TAGS))

        # Tag rarity: mean inverse-frequency across a problem's tags
        out["tag_rarity_mean"] = df["tags"].apply(self._tag_rarity_mean)

        return out

    def _stats_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        solved = pd.to_numeric(df["solved_count"], errors="coerce").fillna(0)
        out["solved_count_log"] = np.log1p(solved)
        out["solved_count_raw"] = solved
        return out

    def _tag_rarity_mean(self, tags) -> float:
        if not isinstance(tags, (list, np.ndarray)) or len(tags) == 0:
            return 0.0
        scores = [1.0 / (self._tag_freq.get(t, 1e-6) + 1e-6) for t in tags if t in self._tag_freq]
        return float(np.mean(scores)) if scores else 0.0

def _ts_to_year(ts) -> int:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).year
    except Exception:
        return 0 