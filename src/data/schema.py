"""
Builds a merged DataFrame from raw API JSON dumps.

Joins problems + problemStatistics from problems_api.json with contest
metadata from contests_api.json on contestId.

Codeforces APIs:
    https://codeforces.com/apiHelp/methods#problemset.problems (Problem & ProblemStatistics)
        - https://codeforces.com/apiHelp/objects#Problem
        - https://codeforces.com/apiHelp/objects#ProblemStatistics
    https://codeforces.com/apiHelp/methods#contest.list (Contests)
        - https://codeforces.com/apiHelp/objects#Contest

Output columns:
    problem_key             str                 PK: "{contest_id}_{index}"
    contest_id              Int64
    problem_index           str                 eg: A,B,C,D,...
    name                    str
    problem_type            str                 PROGRAMMING | QUESTION
    points                  float               nullable
    rating                  Int64               nullable - TARGET variable
    tags                    object list[str]
    solved_count            Int64
    contest_name            str
    contest_type            str                 CF | ICPC | IOI
    contest_duration_secs   Int64
    contest_start_time      Int64               nullable
"""

import json
from pathlib import Path

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

def build_merged_dataframe(
    problems_path: str | Path = "data/raw/problems_api.json",
    contests_path: str | Path = "data/raw/contests_api.json",
    scrape_dir: str | Path = "data/raw/problem_pages",
) -> pd.DataFrame:
    problems_path = Path(problems_path)
    contests_path = Path(contests_path)

    # Load problems + statistics

    with open(problems_path) as f:
        raw_p = json.load(f)

    problems_raw = raw_p["result"]["problems"]
    stats_raw = raw_p["result"]["problemStatistics"]

    df_problems = pd.DataFrame(problems_raw)
    df_stats = pd.DataFrame(stats_raw)

    # Normalize join keys
    df_problems["contestId"] = pd.to_numeric(df_problems.get("contestId"), errors="coerce")

    # Handle empty stats
    if df_stats.empty or "contestId" not in df_stats.columns:
        df_stats = pd.DataFrame(columns=["contestId", "index", "solvedCount"])
    else:
        df_stats["contestId"] = pd.to_numeric(df_stats.get("contestId"), errors="coerce")
    
    for col in ("contestId", "index", "solvedCount"):
        if col not in df_stats.columns:
            df_stats[col] = pd.NA
    
    # join df_stats to df_problems
    df = df_problems.merge(
        df_stats[["contestId", "index", "solvedCount"]],
        on=["contestId", "index"],
        how="left"
    )

    # Load contests
    with open(contests_path) as f:
        raw_c = json.load(f)
    
    df_contests = pd.DataFrame(raw_c["result"])[
        ["id", "name", "type", "durationSeconds", "startTimeSeconds"]
    ].rename(
        columns={
            "id": "contestId",
            "name": "contest_name",
            "type": "contest_type",
            "durationSeconds": "contest_duration_secs",
            "startTimeSeconds": "contest_start_time",
        }
    )
    df_contests["contestId"] = pd.to_numeric(df_contests["contestId"], errors="coerce")

    df = df.merge(df_contests, on="contestId", how="left")

    # Rename again
    df = df.rename(
        columns={
            "contestId": "contest_id",
            "index": "problem_index",
            "name": "name",
            "type": "problem_type",
            "solvedCount": "solved_count",
        }
    )

    # Problem Key
    
    df["problem_key"] = df["contest_id"].astype(str) + "_" + df["problem_index"].astype(str)

    cols = [
        "problem_key", "contest_id", "problem_index", "name", "problem_type",
        "points", "rating", "tags", "solved_count", "contest_name", "contest_type",
        "contest_duration_secs", "contest_start_time",
    ]

    # Set column to NA if not present
    
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    
    # Type conversion

    df = df[cols].copy()
    df["solved_count"] = pd.to_numeric(df["solved_count"], errors="coerce").fillna(0).astype("Int64")
    df["contest_id"] = pd.to_numeric(df["contest_id"], errors="coerce").astype("Int64")
    df["contest_duration_secs"] = pd.to_numeric(df["contest_duration_secs"], errors="coerce").astype("Int64")
    df["contest_start_time"] = pd.to_numeric(df["contest_start_time"], errors="coerce").astype("Int64")

    # Ensure tags is always a list (API sometimes gives strings or None)
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])

    logger.info("Merged DataFrame: %d rows, %d columns", df.shape[0], df.shape[1])
    return df