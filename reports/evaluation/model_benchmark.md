# Model Benchmark

Run: `20261030T103039_primary_ef7c86`  
Trained: 2026-10-30T10:34:12.868590+00:00  
Wall time: 219.2 s  
Samples: r__2026-10.parquet, s__2026-09.parquet

## Splits

| Split | Rows | From | To |
|---|---|---|---|
| train | 300,000 | 2026-09-01 00:00:00 | 2026-09-30 23:59:57 |
| val | 200,002 | 2026-10-01 00:00:00 | 2026-10-01 14:38:07 |
| test | 199,998 | 2026-10-01 14:38:08 | 2026-10-03 13:00:00 |
| recent | 0 | None | None |

> No separate forward holdout was available for this run (only two months were ingested). The `test` split is still strictly later in time than `train`, so the reported numbers are out-of-time rather than in-sample, but they rest on a single month boundary. Treat drift conclusions as provisional.

## Headline comparison (test split)

Ranking metrics at K = 1000. Prevalence shown so Precision@K can be read against the base rate.

| Model | PR-AUC | PR-AUC 95% CI | ROC-AUC | Precision@1000 | Lift@1000 | Brier | ECE |
|---|---|---|---|---|---|---|---|
| E1_logistic_regression **(champion)** | 0.0318 | 0.030–0.035 | 0.7419 | 2.50% | 1.79× | 0.0137 | 0.0104 |
| E3_catboost | 0.0250 | 0.023–0.028 | 0.6858 | 3.70% | 2.65× | 0.0138 | 0.0082 |
| E4_lightgbm | 0.0241 | 0.022–0.027 | 0.6667 | 3.40% | 2.43× | 0.0138 | 0.0076 |
| E2_random_forest | 0.0232 | 0.022–0.026 | 0.6550 | 2.70% | 1.93× | 0.0138 | 0.0056 |
| E0_random_ranking | 0.0146 | — | 0.5162 | 1.20% | 0.86× | 0.3335 | 0.4867 |
| E0_majority_prevalence | — | — | — | — | — | — | — |

Test-set prevalence: 1.399%

### Reading these numbers

- PR-AUC is **2.27×** the base rate. A random ranker scores 0.0140; the champion scores 0.0318. This is real but modest discrimination — not a strong predictor.
- Precision@1000 lift is **1.79×** over random selection at the same cutoff.
- The model over-predicts on average: mean predicted probability 0.01656 against an observed rate of 0.01399 (ratio 1.18×). Probabilities should be read as a ranking signal, not as a literal risk percentage.

## Calibration (champion, test split)

| Predicted band | Count | Mean predicted | Observed rate |
|---|---|---|---|
| 0.00000–0.01262 | 11,704 | 0.01159 | 0.00384 |
| 0.01262–0.01363 | 11,704 | 0.01316 | 0.00521 |
| 0.01363–0.01427 | 11,704 | 0.01397 | 0.00479 |
| 0.01427–0.01473 | 11,704 | 0.01451 | 0.00333 |
| 0.01473–0.01517 | 11,703 | 0.01494 | 0.00427 |
| 0.01517–0.01583 | 11,704 | 0.01545 | 0.00470 |
| 0.01583–0.01699 | 11,704 | 0.01638 | 0.00906 |
| 0.01699–0.01871 | 11,704 | 0.01780 | 0.02973 |
| 0.01871–0.02086 | 11,693 | 0.01978 | 0.03823 |
| 0.02086–0.26301 | 11,715 | 0.02797 | 0.03671 |

## Feature importance (champion)

| Feature | Importance |
|---|---|
| country | 0.49880 |
| language | 0.25499 |
| type | 0.07488 |
| sound_train_median_views_log | 0.06186 |
| sound_train_usage_log | 0.02573 |
| caption_len | 0.01306 |
| caption_words | 0.01273 |
| is_photo_carousel | 0.01009 |
| log_duration | 0.00852 |
| is_shop_video | 0.00704 |
| is_ad | 0.00577 |
| caption_has_mention | 0.00417 |
| caption_emoji_count | 0.00347 |
| has_branded_content | 0.00322 |
| posted_hour | 0.00293 |

## How to read this

- **PR-AUC** is the primary metric. A random ranker scores approximately the prevalence, so compare PR-AUC against the prevalence line above, not against 0.5.
- **Lift@K** states how many times better than random the top-K selection is.
- **Brier** and **ECE** measure probability quality. A model can rank well and still be badly calibrated.
- Confidence intervals resample whole sounds, because videos sharing a sound are correlated.

## Limitations

See `docs/limitations.md`. In short: this is a retrospective cohort-relative snapshot model. It is not a seven-day view forecaster, and no fixed-age label was available.
