"""
Data-layer tests against the cleaned/split pipeline output.

Run `python scripts/run_pipeline.py` first to generate the fixtures these
tests read (data/intermediate/*.parquet, data/processed/split_index_*.parquet).
"""

import pandas as pd
import pytest

from src.data.cleaner import RATING_MIN, RATING_MAX

LABELED_PATH = "data/intermediate/labeled.parquet"
PROCESSED_DIR = "data/processed"


def _load_labeled() -> pd.DataFrame:
    try:
        return pd.read_parquet(LABELED_PATH)
    except FileNotFoundError:
        pytest.skip(f"{LABELED_PATH} not found — run scripts/run_pipeline.py first")


def _load_split_contests(name: str) -> set:
    try:
        df = pd.read_parquet(f"{PROCESSED_DIR}/split_index_{name}.parquet")
    except FileNotFoundError:
        pytest.skip(f"split_index_{name}.parquet not found — run scripts/run_pipeline.py first")
    return set(df["contest_id"])


def test_merged_rows_have_problem_identity():
    df = _load_labeled()
    assert df["problem_key"].is_unique
    expected_key = df["contest_id"].astype(str) + "_" + df["problem_index"].astype(str)
    assert (df["problem_key"] == expected_key).all()


def test_ratings_are_in_expected_range():
    df = _load_labeled()
    assert df["rating"].notna().all()
    unflagged = df[~df["rating_oob_flag"]]
    assert (unflagged["rating"] >= RATING_MIN).all()
    assert (unflagged["rating"] <= RATING_MAX).all()


def test_duplicate_problem_keys_are_removed():
    df = _load_labeled()
    assert not df["problem_key"].duplicated().any()


def test_required_columns_exist():
    df = _load_labeled()
    required = {
        "problem_key", "contest_id", "problem_index", "rating", "tags",
        "solved_count", "contest_name", "contest_type", "contest_start_time",
    }
    assert required.issubset(df.columns)


def test_contests_do_not_overlap_across_splits():
    train_contests = _load_split_contests("train")
    val_contests = _load_split_contests("val")
    test_contests = _load_split_contests("test")

    assert train_contests.isdisjoint(val_contests)
    assert train_contests.isdisjoint(test_contests)
    assert val_contests.isdisjoint(test_contests)
