# ViralIQ — TikTok Virality Intelligence

Predict whether a TikTok video will land in the top 1% of its peer group, understand *why*,
and measure whether the predictions hold up over time.

ViralIQ is built around a single uncomfortable fact about TikTok data: **most public datasets
record one engagement snapshot per video, with an unreliable observation timestamp.** A model
trained on those snapshots can recognise historical viral videos, but that is not the same as
forecasting a new video's future. This project keeps that distinction visible everywhere —
in the schema, in the API, in the dashboard, and in the reported numbers.

---

## What actually works today

| Capability | Status |
|---|---|
| Remote dataset audit (460 GB, no full download) | Implemented and verified against the live source |
| Reproducible stratified sampling | Implemented and verified |
| Cohort-relative virality labelling with leakage guards | Implemented, unit-tested |
| Temporal train/validation/test split | Implemented, unit-tested |
| CatBoost / LightGBM / Random Forest / Logistic Regression benchmark | Implemented, executed |
| Calibrated probabilities + percentile + expected view range | Implemented |
| SHAP per-prediction attribution | Implemented for tree **and** linear models |
| Global feature importance | Implemented for every model family |
| External Parquet/CSV import with normalisation | Implemented, tested |
| FastAPI inference + persistence | Implemented, tested |
| Dashboard wired to real endpoints | Implemented, browser-verified |
| Data freshness / operating-mode reporting | Implemented |
| Early-traction prediction (velocity) | **Blocked** — needs repeated observations |
| Multimodal (video/audio) features | **Not implemented** — no media in the source |
| Live TikTok ingestion | **Not implemented** — requires approved API access |

The API refuses early-traction requests with `409` and an explanation rather than approximating
velocity from a single snapshot.

---

## Quick start

```bash
cd /Users/derr/Documents/Random/ViralIQ
python3 -m pip install -r requirements.txt

# 1. Audit the source (writes data/metadata/dataset_audit.json + reports/research/dataset_audit.md)
python3 scripts/audit.py --tail 2

# 2. Build a local sample (streams from Hugging Face; nothing 460 GB is downloaded)
python3 scripts/build_sample.py --months 2026-08 2026-09 --fraction 0.01
python3 scripts/build_sample.py --months 2026-10 --fraction 0.5 --name recent_2026_10

# 3. Train and evaluate
python3 scripts/train.py

# 4. Serve
uvicorn api.main:app --reload --port 8000
#    API docs:  http://localhost:8000/docs
#    Dashboard: http://localhost:8000/app/
```

No credentials are required for any of the above.

---

## The two engines

**Engine 01 — pre-publication.** Uses only metadata knowable before publishing: caption
statistics, hashtags, duration, format, publication time, language, country, commercial flags,
and sound popularity frozen at the training window. Engagement counters are structurally
excluded — the request schema has no field for them, and a runtime guard rejects any feature
frame containing `views`, `likes`, `shares`, `saves`, `comments`, `downloads` or `engagement_rate`.

**Engine 02 — early traction.** Requires two observations of the same video at different ages.
The source dataset does not contain them, so this engine is declared unavailable rather than
faked. Its feature builder and endpoint exist and will activate once the observation store is
populated.

---

## Why the label is not "250,000 views"

Measured on the audited partitions, p99 view counts are **152,250** (2026-10) and **246,407**
(2026-09), with a median near **350–400**. A single global threshold would mislabel most of the
corpus and would be meaningless across countries.

The peer group is `country × type × observation_age_bucket`. The age bucket matters: a video whose
counters were read 3 hours after publication cannot be fairly compared with one read 20 days after
publication. Rather than discard ~70% of rows or compare them naively, age becomes part of the
definition — a video is only ever ranked against videos observed at a similar age. Thresholds are
fitted on the training window only and frozen before validation or test is touched. Cohorts smaller
than 200 videos fall back to the global p99.

---

## Evaluation philosophy

Viral videos are rare, so accuracy is not reported. The primary metrics are ranking metrics:
**Precision@K, Recall@K, Lift@K, PR-AUC**, with Brier score and expected calibration error for
probability quality. PR-AUC carries a clustered bootstrap interval, because videos sharing a
sound are not independent observations.

Splits are chronological, never random:

```
train  = month M        (thresholds, encoder, sound stats all fitted here)
val    = first half M+1 (calibration only)
test   = second half M+1
recent = month M+2      (untouched forward-time holdout)
```

---

## Measured results

Run `20261030T101104_primary_ee73a8`. Train = 2026-09 (300k rows), test = 2026-10 (200k rows,
never seen during fitting). Full detail in `reports/evaluation/model_benchmark.md`.

| Model | PR-AUC | 95% CI | ROC-AUC | Precision@1000 | Lift |
|---|---|---|---|---|---|
| **E1 logistic regression** | **0.0318** | 0.030–0.035 | **0.742** | **2.50%** | **1.79×** |
| E3 CatBoost | 0.0250 | — | 0.686 | — | — |
| E4 LightGBM | 0.0241 | — | 0.667 | — | — |
| E2 random forest | 0.0232 | — | 0.655 | — | — |
| Random ranking (baseline) | 0.0146 | — | 0.516 | 1.20% | 1.00× |

Base rate 1.40%. **The linear model won.** CatBoost was expected to be the primary candidate, but
on this feature set the extra capacity did not pay for itself — which is exactly why the champion
is chosen by evaluation rather than by assumption.

Read honestly: PR-AUC of 0.0318 against a 0.0146 baseline is **2.18× random**, and ROC-AUC 0.742
is a real but modest signal. This model is a useful ranking prior for triage, not an oracle. The
probabilities also over-predict in absolute terms, so they should be read as a ranking signal.

---

## Repository map

```
src/ingestion/       Hugging Face connector, DuckDB engine, sample builder
src/preprocessing/   schema validation, observation-clock integrity checks
src/labeling/        cohort thresholds, label engineering, leakage guard
src/feature_engineering/  pre-publication and early-traction feature sets
src/models/          encoder, reference distributions, training orchestration
src/evaluation/      ranking metrics, calibration, clustered bootstrap
src/explainability/  SHAP attribution shared by training and serving
src/inference/       predictor + professional prediction object
src/monitoring/      freshness classification and model recency
api/                 FastAPI app, routes, schemas, persistence
dashboard/           single-page analytics UI (no build step)
docs/                architecture, methodology, data dictionary, model card, limitations
reports/             generated audit, evaluation and executive reports
tests/               50 tests: labels, metrics, encoder, splits, importer, attribution, API
```

---

## Verification

```bash
python3 -m pytest tests/ -q          # 50 tests
python3 -m ruff check src api scripts tests
```

The dashboard is also verified in a real browser: every tab, the prediction form, deep
links, and API-failure states are exercised by a Playwright run.

---

## Honest limitations

- **Retrospective, not prospective.** The shipped model estimates cohort-relative snapshot
  virality. It does not predict a seven-day view count, because the source has no seven-day
  observation.
- **No creator features beyond publishing history.** Follower counts are absent from the dataset
  and are not invented.
- **No visual or audio signal.** The dataset carries metadata, not media.
- **Sound popularity is frozen** at the training window, so it decays as trends move. It is also
  the single strongest feature in the current model, which is a concentration risk.
- **Historical-data mode.** Measured publication coverage lags the audit clock by **26.9 days**;
  the app reports this rather than presenting stale data as live.
- **Invalid observation clock.** Rows whose engagement counters are timestamped before publication
  measure **5.7%** (2026-09) to **34.7%** (2026-10) of partitions and are excluded from labelling.

See `docs/limitations.md` for the full list, and `reports/research/dataset_audit.md` for the
measured evidence behind each one.

---

## Licence

Source dataset: **CC BY-NC 4.0** (`datasocial/tiktok-5.6B-videos`) — non-commercial use only.
This repository contains code only; no dataset rows are committed.
