# Feature Documentation

## Variant A — Metadata Only

| Feature | Description | Notes |
|---|---|---|
| `index_numeric` | Problem position: A=1, B=2, ..., AA = 27 ... | A1/A2 both -> 1 |
| `is_opener` | 1 if index_numeric == 1 | Usually easy |
| `is_late_problem` | 1 if index_numeric >= 5 | Usually hard |
| `div_{division}` | One-hot: div1/div2/div3/div4/div1+2/educational/global/icpc/other | Parsed from contest name |
| `is_cf_type` | 1 if contest type == "CF" | |
| `is_icpc_type` | 1 if contest type == "ICPC" | |
| `contest_year` | Year from contest start Unix timestamp | |
| `contest_duration_hours` | Contest duration in hours | |

## Variant B adds — Tags

| Feature | Description |
|---|---|
| `tag_{name}` | Multi-hot binary for each of ~38 known CF tags |
| `num_tags` | Total number of tags |
| `advanced_tag_count` | Count of tags in the advanced set (fft, flows, suffix structures, etc.) |
| `tag_rarity_mean` | Mean inverse-frequency of problem's tags (fit on train only) |

### Advanced tags (indicate harder problems, hand-picked by me)
`2-sat`, `chinese remainder theorem`, `expression parsing`, `fft`, `flows`, `games`, `graph matchings`, `matrices`, `meet-in-the-middle`, `string suffix structures`

## Variant C adds — Public Statistics

| Feature | Description | Leakage note |
|---|---|---|
| `solved_count_log` | log1p(solved_count) | Post-contest only |
| `solved_count_raw` | Raw solved count | Post-contest only |

**Warning:** `solved_count` is highly predictive but only available after a contest runs.

## Leakage Audit

| Feature | Risk | Status |
|---|---|---|
| `solved_count` | High — directly correlates with difficulty | Isolated to Variant C |
| `index_numeric` | Medium — encodes contest ordering | Intentional; well-understood signal |
| `tag_rarity_mean` | Low — fit on train split only | Safe |
| `contest_year` | Very low — time drift of difficulty norms | Acceptable |