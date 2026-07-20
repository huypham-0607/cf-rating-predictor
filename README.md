# Codeforces Problem Difficulty Predictor

`cf-rating-predictor` is a supervised machine-learning pipeline for estimating the
Codeforces-assigned difficulty rating of programming problems from structured
contest metadata. It compares metadata-only, tag-aware, and popularity-aware
feature sets using heuristic, linear, and gradient-boosted regression baselines.

The project does not analyze problem statements and does not claim to measure
intrinsic algorithmic difficulty. Its objective is to quantify how much of the
published rating can be inferred from structured metadata.

## Results

Evaluated on a held-out **test set** of 2,059 rated problems from 291 contests
(the most recent 15% of contests by start time), using a chronological,
contest-grouped split. Full numbers: [`reports/metrics.json`](reports/metrics.json).

| Feature regime | Model | Test MAE | Test RMSE | Within ±100 | Within ±200 |
|---|---:|---:|---:|---:|---:|
| — | Median (naive) | 687.4 | 815.0 | 12.8% | 19.7% |
| Metadata-only | LightGBM | 238.3 | 340.9 | 34.1% | 56.0% |
| Metadata + tags | XGBoost | 228.2 | 322.7 | 33.8% | 56.1% |
| Popularity-aware | **LightGBM** | **138.5** | **187.0** | **47.0%** | **76.3%** |

The best model (LightGBM, popularity-aware) achieved a test MAE of **138.5
rating points**, a **42% reduction** over metadata-only LightGBM and **5x**
better than the naive median baseline. Adding tags on top of metadata alone did
not meaningfully improve MAE for LightGBM (238.3 → 241.0) though it did help
XGBoost; the large jump comes from adding `solvedCount` in the popularity-aware
regime. See [`reports/experiment_report.md`](reports/experiment_report.md) for
the full comparison across all 5 model types x 3 feature regimes, and error
analysis by rating band, division, and problem index.

![Test MAE by model and feature regime](reports/figures/model_comparison.png)

## Problem Definition

The target is the Codeforces-assigned numerical difficulty rating (800-3500) of
a problem — **not** an objective or intrinsic measurement of problem difficulty.
Ratings are set by problem authors and adjusted by community performance; they
reflect Codeforces' rating conventions as much as the problem's inherent
complexity.

## Dataset

Data was collected from the Codeforces API (`problemset.problems` +
`contest.list`) on **2026-04-22**.

| Metric | Value |
|---|---|
| Total PROGRAMMING problems | 11,155 |
| Labeled (rating known) — used for training/eval | 10,864 (97.4%) |
| Unlabeled (no rating) — excluded | 291 (2.6%) |
| Contests represented | 1,973 |
| Rating range | 800-3500 |
| Out-of-range ratings flagged | 0 |
| Raw data format | JSON (`data/raw/`) |
| Intermediate/processed format | Parquet (`data/intermediate/`, `data/processed/`) |

Gym contests are excluded at collection time (`contest.list?gym=false`). No
duplicate `(contestId, index)` pairs were found after merging. Full breakdown
(rating distribution, missingness, tag coverage) in
[`reports/data_quality.md`](reports/data_quality.md).

## Feature Regimes

Three cumulative feature regimes let us isolate how much each source of
information contributes to predictive power:

| Regime | Adds | Meaning |
|---|---|---|
| **Metadata-only** | Problem index, division, contest type, year, duration | Contest structure known before any tags or solve behavior are available |
| **Metadata + tags** | Multi-hot tags, tag count, tag rarity, advanced-tag count | Adds algorithmic tags, still no solve data |
| **Popularity-aware** | `solvedCount` (raw + log) | Adds cumulative solve statistics — **retrospective**, not original in-contest solve counts |

**Note on popularity-aware features:** the Codeforces API's `solvedCount` is
defined only as "number of users who solved the problem," accumulated over all
time (practice, virtual, and upsolve submissions), not just the original
contest. This project does not reconstruct original-contest solve counts, so
Variant C should be read as *retrospective/popularity-aware* rather than a
same-day post-contest signal — older problems have had more time to accumulate
solves. Full feature docs: [`docs/features.md`](docs/features.md).

## Models and Baselines

| Model | Role |
|---|---|
| `mean` / `median` | Naive baselines — ignore all features |
| `ridge` | Linear baseline (StandardScaler + Ridge, α=10) |
| `lgbm` | LightGBM — primary model |
| `xgb` | XGBoost — comparison |

5 model types x 3 feature regimes = 15 trained artifacts (`models/`), each with
a JSON metadata sidecar. The lowest validation-MAE model is auto-selected and
recorded in `models/best_model.json`.

## Evaluation Methodology

- **Split:** chronological and contest-grouped. Contests are sorted by start
  time and bucketed 70/15/15 (train/val/test) by *contest count*; every problem
  from a given contest stays in the same split, so no contest leaks across
  splits. This simulates real deployment — predicting new contests from only
  older ones.
- **Model selection:** all 15 models are trained on train, compared by MAE on
  validation; the best is reported on the untouched test set exactly once.
- **Metrics:** MAE (headline — directly interpretable in rating points), RMSE,
  R², median absolute error, and % of predictions within ±100 / ±200 rating
  points.

## Quick Start

```bash
git clone https://github.com/huypham-0607/cf-rating-predictor.git
cd cf-rating-predictor

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python scripts/run_pipeline.py
pytest -q
```

`run_pipeline.py` re-fetches raw data only if `data/raw/*.json` is missing, so
re-running it after the first pass just re-cleans, re-trains, and re-evaluates
against the cached API dump (~12s end-to-end on the reference machine, not
counting the initial API collection).

Generated outputs:

```text
data/intermediate/, data/processed/  Cleaned + feature-engineered data
models/                              Trained model artifacts + best_model.json
reports/metrics.json                 Machine-readable evaluation results
reports/model_comparison.md          Human-readable results table
reports/experiment_report.md         Full write-up: methodology, results, error analysis
reports/figures/                     model_comparison.png, error_by_rating.png
```

Try predictions interactively:

```bash
streamlit run src/app/streamlit_app.py
```

## Repository Structure

```text
cf-rating-predictor/
├── configs/            collection.yaml, model.yaml
├── data/
│   ├── raw/            API JSON dumps (not committed — see data/README.md)
│   ├── intermediate/   merged + cleaned parquet files
│   └── processed/      feature matrices per variant/split
├── models/             trained .joblib artifacts + metadata JSON
├── reports/            metrics.json, experiment_report.md, figures/
├── scripts/            run_pipeline.py, generate_figures.py
├── src/
│   ├── api/            CodeforcesAPICollector
│   ├── data/           schema.py (merge), cleaner.py (clean + validate)
│   ├── features/       encoder.py (feature logic), pipeline.py (split + save)
│   ├── models/         baseline.py, trainer.py
│   ├── evaluation/     metrics.py
│   ├── inference/      predictor.py — single/batch prediction
│   ├── app/            streamlit_app.py
│   └── utils/          logger
└── tests/              test_data.py, test_features.py, test_smoke_pipeline.py
```

## Reproducing the Results

```bash
python scripts/run_pipeline.py       # collect (cached) -> clean -> features -> train -> evaluate
python scripts/generate_figures.py   # regenerate reports/figures/*.png
```

Both scripts are deterministic given the cached raw data (`random_seed: 67` in
`configs/model.yaml`). Expect `reports/metrics.json`, `reports/model_comparison.md`,
and `models/*.joblib` to match the numbers in this README.

## Limitations

- The target is the Codeforces-assigned rating, an imperfect proxy for
  intrinsic problem difficulty, not a ground-truth measurement.
- Predictions use structured metadata only and do not inspect problem
  statements.
- Cumulative `solvedCount` may include practice and virtual submissions made
  long after the original contest, and older problems accumulate more solves —
  Variant C is retrospective, not a same-day post-contest signal.
- Contest division and problem index encode organizer expectations and may
  dominate more semantically meaningful features (problem index alone explains
  much of the metadata-only signal — see `reports/experiment_report.md`).
- The model shows a systematic positive bias (~+90 rating points on average)
  across most rating bands, flipping to underprediction above 3000 — likely
  reflecting rating-convention drift over time rather than pure noise, since
  the split is chronological (see Interpretation in the experiment report).
- Results may not generalize to other competitive-programming platforms.
- Historical Codeforces data may contain distribution shifts across contest
  formats (CF/ICPC/IOI) and time periods.

## Future Work

- Add a cheap `(division, problem_index)` median heuristic baseline to
  quantify how much of the metadata-only signal is just problem position.
- Investigate the rating-drift bias noted above with an explicit time-based
  calibration term.
