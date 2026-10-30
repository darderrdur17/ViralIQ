# Model Card

**Version** `E2_random_forest@20261030T132450_candidate_297080`

## Intended use

Ranking candidate videos by their probability of landing in the top 1% of observed views within their country-and-format peer group. Intended for content triage and analyst prioritisation, with human review.

## Out-of-scope use

- Predicting exact view counts or seven-day outcomes.
- Any causal claim about why a video performed well.
- Automated publishing or removal decisions.
- Comparing creators, since follower counts are absent from the source.

## Training data

- Source: `datasocial/tiktok-5.6B-videos` (CC BY-NC 4.0)
- Samples: r__2026-10.parquet, s__2026-09.parquet
- Training rows: 20,000
- Rows excluded for an invalid observation clock: 5.685%

## Label

- Definition: top 1% of observed views within `country × type × observation_age_bucket` cohorts
- Global threshold: 274,023 views
- Cohort thresholds fitted: 8
- Prevalence on valid rows: 1.240%
- Fixed-age labels (`viral_24h`, `viral_7d`): **unavailable by design**

## Performance

| Metric | Value |
|---|---|
| PR-AUC | 0.0062 |
| ROC-AUC | 0.5814 |
| Brier | 0.0041 |
| ECE (10 bins) | 0.0046 |
| Test rows | 11,684 |

## Ethical and legal considerations

- The source dataset is **non-commercial only**. Commercial deployment requires a separate licence.
- No creator identifiers or follower data are used.
- Predictions are statistical associations, not judgements about creators.

## Environment

- Python 3.13.12
- Platform macOS-27.0.1-arm64-arm-64bit-Mach-O
