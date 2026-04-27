# Codeforces Rating Predictor

A MVP using tree-based models to predicts the difficulty rating of a Codeforces problem (800 - 3500) from metadata using tree-based models (XGBoost, LGBM). No statement required/considered.

## What It Does

Given a set of features (contest position, tags, division, year, solve counts, etc), the system outputs an estimated numeric difficulty rating.

## Feature Variants

The system trains three model variants, each building on the previous:

| Variant | Features | Use case |
|---|---|---|
| **A** | Problem index, division, contest type, year, duration | Absolute cold-start — no tags, no stats |
| **B** | Variant A + tag multi-hot, tag counts, tag rarity | Standard cold-start prediction missing solvedCount |
| **C** | Variant B + solvedCount (log + raw) | Post-contest prediction (stronger but requires solve data) |

These model variants also allows us to compare the performance between models, determining which features has the most impact on predictive performance.

## Models

Three model types are trained on each variant:

| Model | Role |
|---|---|
| `mean` / `median` | Naive baselines |
| `ridge` | Linear baseline (StandardScaler + Ridge) |
| `lgbm` | LightGBM — primary recommended model |
| `xgb` | XGBoost — comparison |

Best model is auto-selected by validation MAE and saved to `models/best_model.json`.
## Project Structure

```
codeforces-rating-predictor/
├── configs/                  # collection.yaml, model.yaml
├── data/
│   ├── raw/                  # API JSON dumps
│   ├── intermediate/         # merged + cleaned parquet files
│   └── processed/            # feature matrices per variant/split
├── models/                   # trained .joblib artifacts + metadata JSON
├── reports/                  # model_comparison.md
├── scripts/                  # 01_collect.py … 05_evaluate.py, run_pipeline.py
├── src/
│   ├── api/                  # CodeforcesAPICollector
│   ├── data/                 # schema.py (merge), cleaner.py (clean + validate)
│   ├── features/             # encoder.py (feature logic), pipeline.py (split + save)
│   ├── models/               # baseline.py, trainer.py
│   ├── evaluation/           # metrics.py
│   ├── inference/            # 
│   └── utils/                # logger
```


## Limitations

tbd