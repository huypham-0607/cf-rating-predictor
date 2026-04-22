"""
Data cleaning + validation
"""


from pathlib import Path
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

RATING_MIN = 500
RATING_MAX = 3500

def clean_and_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (labeled_df, unlabeled_df).

    labeled_df - problems with an official rating, used for supervised training
    unlabeled_df - problems without a rating, maybe perserved for future use
    """

    original_count = len(df)
    logger.info("Start cleaning. No of rows: %d", original_count)

    # == Inclusion rules ===============================================================
    # 1. problem_type == "PROGRAMMING"
    # 2. contest_name != NULL
    # 3. contest_start_time != NULL
    # 4. solved_count != NULL
    
    df = df[df["problem_type"] == "PROGRAMMING"].copy()
    df = df[df["contest_name"].notna()].copy()
    df = df[df["contest_start_time"].notna()].copy()
    df = df[df["solved_count"].notna()].copy()

    logger.info("After filter: %d rows", len(df))

    # == Deduplication ===============================================================

    df = df.drop_duplicates(subset=["problem_key"]).copy()
    logger.info("Dropped duplicates (by problem_key): %d rows", len(df))

    # == Normalise problem_index =====================================================
    # Preserve raw index, also create base_index (strip trailing digits A1→A)
    df["base_index"] = df["problem_index"].str.extract(r"^([A-Za-z]+)", expand=False).str.upper()

    # == Tag normalisation ===========================================================
    df["tags"] = df["tags"].apply(_normalize_tags)

    # == Rating validation ===========================================================
    # Flag out-of-range ratings but don't drop - just mark them
    mask_oob = df["rating"].notna() & ((df["rating"] < RATING_MIN) | (df["rating"] > RATING_MAX))
    if mask_oob.sum() > 0:
        logger.warning("Found %d problems with rating outside [%d, %d] — flagged but kept", mask_oob.sum(), RATING_MIN, RATING_MAX)
    df["rating_oob_flag"] = mask_oob

    labeled = df[df["rating"].notna()].copy()
    unlabeled = df[df["rating"].isna()].copy()

    logger.info("Labeled (rating known): %d rows", len(labeled))
    logger.info("Unlabeled (rating unknown): %d rows", len(unlabeled))

    return labeled, unlabeled

def generate_data_quality_report(labeled: pd.DataFrame, unlabeled: pd.DataFrame, output_path: str | Path = "reports/data_quality.md") -> None:
    total = len(labeled) + len(unlabeled)
    lines = [
        "# Data Quality Report",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total PROGRAMMING problems | {total:,} |",
        f"| Labeled (rating known) | {len(labeled):,} ({100*len(labeled)/total:.1f}%) |",
        f"| Unlabeled (no rating) | {len(unlabeled):,} ({100*len(unlabeled)/total:.1f}%) |",
        f"| Out-of-range rating flag | {int(labeled['rating_oob_flag'].sum()):,} |",
        "",
        "## Rating Distribution (labeled set)",
        "",
    ]

    # Rating distribution by band
    bands = [(0, 1200, "<1200"), (1200, 1600, "1200–1599"), (1600, 2000, "1600–1999"),
             (2000, 2400, "2000–2399"), (2400, 9999, "2400+")]
    lines.append("| Band | Count | % |")
    lines.append("|---|---|---|")
    for lo, hi, label in bands:
        mask = (labeled["rating"] >= lo) & (labeled["rating"] < hi)
        n = int(mask.sum())
        pct = 100 * n / len(labeled) if len(labeled) > 0 else 0
        lines.append(f"| {label} | {n:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Missing Values (labeled set)",
        "",
        "| Column | Missing | % |",
        "|---|---|---|",
    ]
    for col in labeled.columns:
        missing = int(labeled[col].isna().sum())
        if missing > 0:
            pct = 100 * missing / len(labeled)
            lines.append(f"| {col} | {missing:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Tag Coverage",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Problems with ≥1 tag | {int((labeled['tags'].apply(len) > 0).sum()):,} |",
        f"| Problems with 0 tags | {int((labeled['tags'].apply(len) == 0).sum()):,} |",
        f"| Unique tags | {len({t for tags in labeled['tags'] for t in tags}):,} |",
        f"| Mean tags per problem | {labeled['tags'].apply(len).mean():.2f} |",
        "",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Data quality report saved to %s", output_path)

def _normalize_tags(tags) -> list[str]:
    if not isinstance(tags, list):
        return []
    return sorted({t.strip().lower() for t in tags if isinstance(t, str) and t.strip()})