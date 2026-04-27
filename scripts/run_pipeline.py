import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import CodeforcesAPICollector
from src.data import build_merged_dataframe, clean_and_validation
from src.data.cleaner import generate_data_quality_report
from src.evaluation import evaluate_all_models
from src.features.pipeline import build_feature_pipeline
from src.models.trainer import train_all_models
from src.utils import get_logger

logger = get_logger("pipeline")

def main(force_collect: bool = False) -> None:
    # Step 1: Collect
    logger.info("=" * 60)
    logger.info("STEP 1/5: Data collection")
    logger.info("=" * 60)

    collector = CodeforcesAPICollector()
    collector.fetch_problems(force=force_collect)
    collector.fetch_contests(force=force_collect)

    # Step 2: Merge + clean
    logger.info("=" * 60)
    logger.info("STEP 2/5: Cleaning and merging")
    logger.info("=" * 60)

    df = build_merged_dataframe()
    labeled, unlabeled = clean_and_validation(df)
    Path("data/intermediate").mkdir(parents=True,exist_ok=True)
    labeled.to_parquet("data/intermediate/labeled.parquet")
    unlabeled.to_parquet("data/intermediate/unlabeled.parquet")
    generate_data_quality_report(labeled, unlabeled, "reports/data_quality.md")

    # Step 3: Features
    logger.info("=" * 60)
    logger.info("STEP 3/5: Feature engineering")
    logger.info("=" * 60)

    build_feature_pipeline()

    # Step 4: Train
    logger.info("=" * 60)
    logger.info("STEP 4/5: Model training")
    logger.info("=" * 60)

    train_all_models()

    # Step 5: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 5/5: Evaluation")
    logger.info("=" * 60)

    results = evaluate_all_models()

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Best model results:")
    logger.info("\n%s", results.sort_values("MAE").head(5)[["model", "variant", "MAE", "within_100", "within_200"]].to_string(index=False))
    logger.info("Run the app: streamlit run src/app/streamlit_app.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-collect", action="store_true", help="Force re-fetch API")
    args = parser.parse_args()
    main(force_collect=args.force_collect)