# Model Benchmark

Run: `20261030T132450_candidate_297080`  
Trained: 2026-10-30T13:25:00.412515+00:00  
Wall time: 11.2 s  
Samples: r__2026-10.parquet, s__2026-09.parquet

## Splits

| Split | Rows | From | To |
|---|---|---|---|
| train | 20,000 | 2026-09-01 00:00:00 | 2026-09-30 23:58:48 |
| val | 20,000 | 2026-10-01 00:00:00 | 2026-10-01 14:38:06 |
| test | 20,000 | 2026-10-01 14:38:11 | 2026-10-03 12:47:20 |
| recent | 0 | None | None |

> No separate forward holdout was available for this run (only two months were ingested). The `test` split is still strictly later in time than `train`, so the reported numbers are out-of-time rather than in-sample, but they rest on a single month boundary. Treat drift conclusions as provisional.

## Headline comparison (test split)

Ranking metrics at K = 1000. Prevalence shown so Precision@K can be read against the base rate.

| Model | PR-AUC | PR-AUC 95% CI | ROC-AUC | Precision@1000 | Lift@1000 | Brier | ECE |
|---|---|---|---|---|---|---|---|
| E2_random_forest **(champion)** | 0.0062 | 0.004–0.011 | 0.5814 | 0.70% | 1.70× | 0.0041 | 0.0046 |
| E4_lightgbm | 0.0053 | 0.004–0.008 | 0.5610 | 0.50% | 1.22× | 0.0041 | 0.0043 |
| E0b_sound_popularity | 0.0041 | — | 0.4813 | 0.00% | 0.00× | 0.0413 | 0.2334 |
| E0_random_ranking | 0.0039 | — | 0.4985 | 0.30% | 0.73× | 0.3308 | 0.4935 |
| E1_logistic_regression | 0.0035 | 0.003–0.005 | 0.4350 | 0.20% | 0.49× | 0.0041 | 0.0039 |
| E0c_duration_only | 0.0033 | — | 0.3853 | 0.20% | 0.49× | 0.0041 | 3.1269 |
| E3_catboost | 0.0028 | 0.002–0.004 | 0.3146 | 0.00% | 0.00× | 0.0041 | 0.0044 |
| E0_majority_prevalence | — | — | — | — | — | — | — |

Test-set prevalence: 0.411%

### Reading these numbers

- PR-AUC is **1.52×** the base rate. A random ranker scores 0.0041; the champion scores 0.0062. This is real but modest discrimination — not a strong predictor.
- Precision@1000 lift is **1.70×** over random selection at the same cutoff.
- The model over-predicts on average: mean predicted probability 0.00863 against an observed rate of 0.00411 (ratio 2.10×). Probabilities should be read as a ranking signal, not as a literal risk percentage.

## Calibration (champion, test split)

| Predicted band | Count | Mean predicted | Observed rate |
|---|---|---|---|
| 0.00639–0.00718 | 1,169 | 0.00695 | 0.00171 |
| 0.00718–0.00746 | 1,168 | 0.00732 | 0.00428 |
| 0.00746–0.00769 | 1,157 | 0.00757 | 0.00432 |
| 0.00769–0.00793 | 1,180 | 0.00781 | 0.00254 |
| 0.00793–0.00820 | 1,168 | 0.00806 | 0.00685 |
| 0.00820–0.00849 | 1,168 | 0.00834 | 0.00086 |
| 0.00849–0.00886 | 1,169 | 0.00866 | 0.00171 |
| 0.00886–0.00942 | 1,168 | 0.00912 | 0.00257 |
| 0.00942–0.01048 | 1,168 | 0.00990 | 0.01027 |
| 0.01048–0.11374 | 1,169 | 0.01251 | 0.00599 |

## Feature importance (champion)

| Feature | Importance |
|---|---|
| sound_train_median_views_log | 0.69890 |
| duration_s | 0.05495 |
| log_duration | 0.03886 |
| caption_len | 0.02420 |
| language | 0.02322 |
| caption_words | 0.02289 |
| posted_hour | 0.02008 |
| caption_upper_ratio | 0.01953 |
| country | 0.01844 |
| posted_dow | 0.01223 |
| caption_emoji_count | 0.01068 |
| log_hashtag_count | 0.00963 |
| sound_train_usage_log | 0.00919 |
| hashtag_count | 0.00767 |
| type | 0.00505 |

## How to read this

- **PR-AUC** is the primary metric. A random ranker scores approximately the prevalence, so compare PR-AUC against the prevalence line above, not against 0.5.
- **Lift@K** states how many times better than random the top-K selection is.
- **Brier** and **ECE** measure probability quality. A model can rank well and still be badly calibrated.
- Confidence intervals resample whole sounds, because videos sharing a sound are correlated.

## Limitations

See `docs/limitations.md`. In short: this is a retrospective cohort-relative snapshot model. It is not a seven-day view forecaster, and no fixed-age label was available.
