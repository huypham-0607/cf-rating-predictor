# Codeforces Rating Predictor

A MVP using tree-based models to predicts the difficulty rating of a Codeforces problem (800 - 3500) from metadata using. No statement required/considered.

## What It Does

Given a set of features (contest position, tags, division, year, solve counts, etc), the system outputs an estimated numeric difficulty rating.

## Feature Variants

tbd

## Models

Three model types are trained on each variant:

| Model | Role |
|---|---|
| `mean` / `median` | Naive baselines |
| `lgbm` | LightGBM — primary model |
| `xgb` | XGBoost — comparison |

## Project Structure

```
codeforces-rating-predictor/
├── configs/                  # collection.yaml, model.yaml
├── data/
│   ├── raw/                  # API JSON dumps
│   ├── intermediate/         # merged + cleaned parquet files
│   └── processed/            # feature matrices per variant/split
├── models/                   #
├── reports/                  #
├── scripts/                  #
├── src/
│   ├── api/                  # CodeforcesAPICollector
│   ├── data/                 # schema.py (merge), cleaner.py (clean + validate)
│   ├── features/             # encoder.py (feature logic), pipeline.py (split + save)
│   ├── models/               #
│   ├── evaluation/           #
│   └── utils/                # logger
```

---

## Limitations

tbd