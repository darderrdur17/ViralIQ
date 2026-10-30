# Can ViralIQ predict the *next* trend?

**Short answer: no — not with this dataset, and not by a small margin.**

The blocker is not modelling skill. It is arithmetic: trends rise faster than this data arrives.
By the time a hashtag's acceleration is visible in the corpus, the trend is already roughly three
weeks past its peak.

This document records the measurement behind that claim, what the system does instead, and what it
would actually take to build genuine trend forecasting.

---

## 1. What the product currently does

The Trend Intelligence panel ranks hashtags and sounds by **acceleration inside the observed
window**:

```
acceleration = usage in the final 7 days of the window / usage in all earlier days
```

That is **detection of something that already happened**, not a forecast. It answers "what was
accelerating, as of the last day of data?" — with a lag described below. The API labels it that way
(`caveat`, `pattern: emerging | accelerating`), but until now the *size* of the lag was not stated.

---

## 2. The measurement

Computed with DuckDB over the ingested sample (2026-09 at 1%, 2026-10 at 100%), inverse-probability
weighted so the two sampling fractions are comparable. Window: **32 days**.

For every hashtag with at least 2,000 weighted uses, I measured:

- **rise time** — days from the first day crossing 20% of peak usage to the peak day
- **days since peak** — how long before the last day of data the hashtag peaked

| Volume threshold | Hashtags | Rise time (median) | Rise time (p75) | Days since peak at window end |
|---|---:|---:|---:|---:|
| ≥ 2,000 uses | 11,085 | **7 d** | 14 d | 24 d |
| ≥ 20,000 uses | 971 | **11 d** | 15 d | 21 d |
| ≥ 100,000 uses | 120 | **9 d** | 14 d | 23 d |

The result is stable across thresholds, so it is not an artefact of the 1% sample's noise.

**Compare with the measured data delay: 26.9 days** (from `reports/research/dataset_audit.md`).

```
trend rises and peaks        7–11 days
data becomes available      26.9 days later
```

The delay is **~3–4× the rise time**. A hashtag that peaks on day *t* first appears in this corpus
around day *t + 27*. At that moment it is already ~20 days past peak, and by the time a human reads
the dashboard it is further still.

### Why this is fatal rather than inconvenient

To predict "what will trend next week", you need observations from *before* the trend. This dataset
cannot provide them, because:

1. **One snapshot per video.** There is no time series of a hashtag's usage — only a set of videos
   each carrying a single `posted_at`. Daily counts are reconstructed after the fact, which is
   fine for history and useless for now.
2. **The corpus lags reality.** At the audit date, the newest video in the dataset was 26.9 days
   old. Anything you compute is a historical statement.
3. **No counterfactual.** You observe hashtags that were used; you never observe the adoption
   decisions that were *not* made. Trends are rare, bursty events, and the negatives are the whole
   problem.
4. **The driver is exogenous.** What actually makes a sound or format explode is largely the
   recommendation algorithm and off-platform events. Neither appears in metadata, so no amount of
   feature engineering recovers it.

A model trained here could only learn "hashtags that were already growing continued to grow",
which is not a forecast and would score well in backtests while being useless in production.

---

## 3. What would make it possible

Trend forecasting needs a different data collection design, not a better model.

**Required: repeated observations of the same entity over time.**

| Requirement | Why | Current state |
|---|---|---|
| Daily (ideally 6-hourly) usage counts per hashtag and sound | Growth rate, curvature, and acceleration are only measurable with a time series | Absent — single snapshots |
| Freshness within ~24 h | Prediction is worthless if the input is 27 days old | 26.9-day lag |
| Creator diversity per trend | A hashtag used by 5 accounts is not a trend; by 5,000 it might be | Not available (no creator id) |
| Cross-country / cross-language spread | Real trends propagate geographically; a single-market spike usually decays | Partially available via `country` |
| Novelty signal | First-seen date, prior usage of the same sound/format | Derivable once observations exist |

With that, the task becomes a **hazard model**: estimate

```
P(hashtag accelerates in the next 7 days | current growth, creator diversity,
                                            geographic spread, novelty, sound adoption)
```

and evaluate it the only way that means anything — **forward in time**, on weeks the model has not
seen, with Precision@K against the trends that actually materialised. That is exactly the protocol
already used for video virality in `docs/evaluation_protocol.md`; only the data source changes.

**Realistic expectation, even then.** Trend emergence is driven substantially by platform-level
changes and off-platform events. A well-built hazard model on good observation data might
plausibly reach modest lift over a naive "yesterday's growth continues" baseline. Anyone promising
reliable next-week trend prediction from metadata is overclaiming.

---

## 4. What we changed as a result

Rather than ship a panel that invites the wrong reading, the lag is now stated in the product:

- `GET /api/v1/trends` returns `data_lag_days` and a plain-language `forecast_note`.
- The dashboard renders that note above the trend tables.

The capability is unchanged — it is still detection. The difference is that a user can no longer
mistake it for prediction.

---

## 5. Bottom line

| Question | Answer |
|---|---|
| Can ViralIQ predict the next trend? | **No.** The data arrives ~4× slower than trends rise. |
| Can it detect trends that already happened? | **Yes**, with a ~27-day lag, clearly labelled. |
| Is a trend predictor buildable at all? | **Yes**, but it needs repeated observations of hashtag/sound usage, not this snapshot corpus. |
| Is that worth doing? | Yes — it is the single highest-value extension, and the observation table already exists (`video_observations`) waiting to be populated. |

*Measurements reproducible with the query in §2 against `data/samples/`; see also
`reports/research/dataset_audit.md` for the observation-clock and freshness evidence.*
