# Architecture

## Design principle

Every reported number must be traceable to a stored artefact: a Parquet file, a model
binary, a threshold, or a SQL aggregate. Where a capability cannot be built honestly, the
system reports the gap instead of approximating it.

## Layers

```
            ┌──────────────────────────────────────────────┐
            │  dashboard/  (single-page, no build step)    │
            └───────────────────┬──────────────────────────┘
                                │ fetch /api/v1/*
            ┌───────────────────▼──────────────────────────┐
            │  api/  FastAPI                                │
            │   routes/   predictions · data · models ·     │
            │             analytics                         │
            │   services/ store (DuckDB) · analytics (SQL)  │
            └───────────────────┬──────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────────┐
        │                       │                            │
┌───────▼────────┐   ┌──────────▼─────────┐   ┌──────────────▼────────┐
│ src/inference  │   │ src/monitoring     │   │ src/evaluation        │
│ Predictor      │   │ freshness          │   │ metrics · report      │
└───────┬────────┘   └──────────┬─────────┘   └──────────────┬────────┘
        │                       │                            │
┌───────▼───────────────────────▼────────────────────────────▼────────┐
│ src/explainability  attribution (SHAP, shared by training + serving) │
│ src/models        encoder · reference · train                       │
│ src/feature_engineering   pre-publication · early-traction          │
│ src/labeling      cohort thresholds · labels · leakage guard        │
│ src/preprocessing quality · schema validation                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ src/ingestion          │
                    │ hf_source · duckdb     │
                    │ sample_builder         │
                    │ importer (parquet/csv) │
                    └───────────┬────────────┘
                                │ HTTPS, column-pruned
                    ┌───────────▼────────────┐
                    │ Hugging Face: 148      │
                    │ monthly Parquet files  │
                    │ 460 GB total           │
                    └────────────────────────┘
```

## Data flow

1. **Audit** (`scripts/audit.py`) — DuckDB reads Parquet footers and a narrow column set over
   HTTPS. Measures row counts, coverage, view distribution and the observation-clock
   integrity. Writes `data/metadata/dataset_audit.json`.
2. **Sample** (`scripts/build_sample.py`) — deterministic hash sampling on `video_id`,
   projected to 21 columns. Writes `data/samples/*.parquet` plus a `.meta.json` provenance
   record.
3. **Label** (`src/labeling`) — fits per-cohort p99 thresholds on the training month only,
   attaches `observation_valid`, and emits fixed-age labels as NULL.
4. **Feature** (`src/feature_engineering`) — builds the pre-publication matrix. Sound
   statistics are frozen from the training window.
5. **Train** (`src/models/train.py`) — chronological split, four model families plus two
   baselines, probability calibration on the validation slice, evaluation on test and a
   forward-time holdout, artefacts written to `models/model_registry/<run_id>/`.
6. **Serve** (`api/`) — loads the active run from `index.json`, scores requests, persists
   every prediction with its model version.
7. **Report** (`src/evaluation/report.py`) — renders benchmark, executive summary and model
   card from `run.json`.

## Why DuckDB and Polars

The dataset is 460 GB across 148 files. DuckDB queries remote Parquet with column pruning and
predicate pushdown, so an audit costs a few hundred megabytes of transfer rather than 460 GB.
Polars handles in-memory transformation of the sampled frames. Pandas is used only at the
model boundary, where scikit-learn and CatBoost expect it.

## Why a DuckDB application database

Predictions, observations and ingestion runs are persisted in a single embedded DuckDB file
(`database/viraliq.duckdb`). This keeps the MVP deployable with no external service while
remaining SQL-queryable. The schema is intentionally Postgres-compatible; swapping in
PostgreSQL means changing the connection string and the JSON column type.

## Leakage controls

| Risk | Control |
|---|---|
| Engagement used as a pre-publication feature | Request schema omits it; `assert_no_performance_features` raises on any feature frame containing performance columns |
| Threshold fitted on test data | Thresholds fitted on the training month, frozen in `thresholds.json`, applied unchanged |
| Sound popularity reflecting future usage | `fit_sound_stats` runs on the training window only and is persisted with the model |
| Category vocabulary fitted on test | `FeatureEncoder.fit` sees training rows only; unseen categories map to `__unseen__` |
| Random split inflating scores | Splits are strictly chronological |
| Correlated rows shrinking confidence intervals | Bootstrap resamples whole sounds |

## Failure modes and how they surface

- **No model** → `/predictions/prepublish` returns `503` with a training instruction.
- **No sample data** → analytics returns `available: false` with a reason, never a fake chart.
- **Partial Parquet write** → file discovery ignores files without the trailing `PAR1` magic.
- **Stale source** → `/data/status` reports `historical` mode with the measured delay.
- **Early-traction request** → `409` explaining that velocity needs repeated observations.
