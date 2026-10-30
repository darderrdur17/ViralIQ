# Limitations

This document is deliberately blunt. Every claim the platform makes is bounded by the items
below, and the API surfaces the most important ones in each prediction response.

## 1. It is a retrospective classifier, not a forecaster

The shipped model estimates whether a video lands in its cohort's top 1% of **observed** views.
It does not predict a seven-day view count.

Why: the source dataset stores a single engagement snapshot per video. There is no second
observation, so neither growth rate nor a fixed-age outcome can be computed. `viral_24h` and
`viral_7d` are emitted as NULL rather than estimated.

Any claim that this model "predicts virality in seven days" would be false.

## 2. The observation clock is unreliable

A large fraction of rows record engagement counters at a timestamp **earlier than the video's
publication** — impossible, and measured per partition in `reports/research/dataset_audit.md`.
Those rows are excluded from label construction.

Implication: even the snapshot label is measured at an unknown, inconsistent age. A video seen at
2 hours and a video seen at 30 days are compared as if their view counts were commensurable.
They are not, which compresses the signal.

## 3. No creator signal

The dataset has no creator identifier and no follower counts. Consequences:

- Creator historical performance cannot be computed.
- Generalisation to unseen creators cannot be evaluated.
- The model cannot distinguish "a first-time poster got lucky" from "a large account did what it
  always does".

This is likely the single largest missing predictor.

## 4. No visual or audio content

The dataset contains metadata, not media. There are no frame embeddings, no audio
representations, no scene features. Content *quality* — the thing a human would judge — is
entirely unobserved. The model sees the wrapper, not the video.

## 5. Sound popularity is frozen

`sound_train_usage_log` and `sound_train_median_views_log` are computed on the training window
and stored with the model. As trends move, these decay. A sound that is exploding today is
scored using last month's popularity.

This is a genuine limitation, not a bug: using present-day popularity to score a video published
last month would be leakage.

## 6. Historical-data mode

Publication coverage in the audited partitions lags the audit clock by weeks. The platform
therefore reports `historical` mode and never presents its figures as live TikTok data.

## 7. Geographic and linguistic skew

Country and language are modelled as categorical features with a 60-value vocabulary. Long-tail
markets collapse into `__unseen__` and receive the pooled behaviour of everything else. Predictions
for small markets should be treated as low-confidence even though the API does not currently
express per-market confidence.

## 8. Correlation is not causation

SHAP attributions describe how the model weighs features. They do not establish that changing a
feature would change a video's fate. "Hashtag count increases the score" is a statement about the
model, not about TikTok.

## 9. Calibration is population-level

Probabilities are calibrated on the validation slice. Calibration can drift for subgroups
(specific countries, formats, or durations) even when overall calibration looks good. Per-segment
calibration is not currently measured.

## 10. Single forward holdout

Drift is assessed against one later month. That is a weak test: it cannot distinguish a stable
model from one that happened to survive one shift. Rolling evaluation across several months is
the recommended next step.

## 11. Not integrated with any live API

No TikTok Research API, Display API, Creative Center or commercial provider integration is
implemented or claimed. The supported path for new data is manual Parquet/CSV import, which
updates the freshness record.

## 11a. Persistence assumes a single application process

The application database is an embedded DuckDB file. DuckDB holds an **OS-level lock per database
file**, so:

- Two processes cannot open it at once. `uvicorn --workers 4` will fail on the second worker.
- A running server blocks the test suite unless the suite points elsewhere. Tests do exactly that
  via `VIRALIQ_DB_PATH` (`tests/conftest.py`).
- Concurrent writes within one process must share a connection. The store now holds a single
  locked connection; opening one per request previously failed under load with
  `Unique file handle conflict`.

For horizontal scaling, replace the store with PostgreSQL — the schema is already
Postgres-compatible. This is a deliberate MVP trade-off, not an oversight.

## 11b. Capability differences between the two deployments

The full service and the serverless function expose the same routes and the same response
*shapes*, but not the same capabilities. Where a capability is missing, the response says so with
`available: false` and a reason — the same pattern the analytics routes use — rather than returning
a 5xx that reads as a fault.

| Route | Full service | Serverless |
|---|---|---|
| `GET /data/schema` | `available: true` + measured schema | `available: false`, `reason: remote_schema_check_unavailable` |
| `GET /analytics/*` | Live DuckDB aggregates | Pre-computed snapshot, stamped with its measurement time |
| `POST /data/import` | Writes to the sample corpus | `503 read_only_deployment` |
| `GET /predictions/history` | Stored predictions | Empty, with a `deployment_note` |

`HEAD` is served for every `GET` route by `src/common/middleware.py`, on both deployments.

## 12. What would make this materially stronger

In rough order of impact:

1. **Repeated observations of the same videos** at 30m / 1h / 6h / 24h / 7d. This alone converts
   the project from retrospective classification into genuine forecasting.
2. **Creator-level data** (identifier plus follower count at observation time).
3. **Media features** from an authorised source.
4. **Rolling monthly evaluation** to detect drift rather than test it once.
5. **Per-segment calibration** so that low-confidence markets are visibly low-confidence.
