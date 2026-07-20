# Data

Raw Codeforces API responses and generated feature matrices are not committed
because they are reproducible from the API and may be large. Only the empty
directory structure (`.gitkeep`) is versioned.

Run:

```bash
python scripts/run_pipeline.py
```

This fetches raw data only if `data/raw/*.json` is missing (pass
`--force-collect` to re-fetch), then regenerates everything downstream.

## Expected outputs

```text
data/raw/
  problems_api.json      Raw problemset.problems + problemStatistics response
  contests_api.json      Raw contest.list response (non-gym)

data/intermediate/
  labeled.parquet         Cleaned, merged problems with a known rating
  unlabeled.parquet       Cleaned, merged problems with no rating (excluded from training)

data/processed/
  split_index_{train,val,test}.parquet   problem_key/contest_id/rating per split
  {train,val,test}_{A,B,C}.parquet       feature matrix + rating per split/variant
  feature_names_{A,B,C}.json             ordered feature column names per variant
```

See the main [`README.md`](../README.md) for dataset size, collection date, and
split methodology.
