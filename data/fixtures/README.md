# Demo fixture

`demo_sample.parquet` — **20,000 rows** sampled deterministically from the source dataset
(October 2026 partition), used to make a fresh clone demo-able without any download.

## Why this exists

A fresh clone has no data and no trained model, so `uvicorn` alone renders an empty dashboard and
every prediction returns `503`. Downloading the corpus and training takes roughly fifteen minutes.
This fixture closes that gap:

```bash
python scripts/seed_demo.py     # copies the fixture, trains on it, ~30s
uvicorn api.main:app --port 8000
```

## What it is, and is not

- **Real data.** These are genuine rows from the source dataset, not synthetic records. The
  dashboard therefore shows real measurements — just of a small slice.
- **Not representative.** 20,000 rows spanning three days of October 2026. Cohort thresholds fall
  back to the global p99 far more often than in a full run, and the benchmark numbers produced from
  it are **not** the numbers quoted in the main README. Those come from a 300,000-row training
  window across three months.

## Provenance

| | |
|---|---|
| Source | [`datasocial/tiktok-5.6B-videos`](https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos) |
| Partition | `2026-10` |
| Rows | 20,000 (of 1,734,031) |
| Selection | `pl.DataFrame.sample(n=20_000, seed=20261030)` — deterministic and reproducible |
| Columns | 21, matching `SAMPLE_COLUMNS` in `src/ingestion/sample_builder.py` |
| Licence | **CC BY-NC 4.0** — non-commercial use only, attribution required |

The source dataset's licence applies to this file. It is included for non-commercial
demonstration and evaluation only. See the repository `LICENSE` for the code's terms.

## Regenerating

```python
import polars as pl
df = pl.read_parquet("data/samples/r__2026-10.parquet")
df.sample(n=20_000, seed=20261030).sort("posted_at").write_parquet(
    "data/fixtures/demo_sample.parquet"
)
```
