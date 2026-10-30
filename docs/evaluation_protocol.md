# Evaluation Protocol

The purpose of this document is to make the reported numbers auditable: given the same sample
files and the same protocol, an independent reviewer should be able to reproduce or refute them.

## 1. Evaluation populations

| Population | Definition | What it can support |
|---|---|---|
| `test` | Second half of month M+1 | Headline performance claims |
| `recent` | Month M+2, never used in fitting | Claims about forward-time robustness |
| `train` | Month M | Never used for reporting performance |

Reporting a training-set score as evidence of predictive skill is prohibited.

## 2. Mandatory preconditions

Before any metric is computed:

1. `observation_valid` must be true for the row. Invalid-clock rows are dropped from the
   evaluation population and their count is reported.
2. Thresholds must come from `thresholds.json`, produced by the training window. Recomputing
   thresholds on the evaluation population invalidates the result.
3. The encoder vocabulary must come from `encoder.json`, fitted on training rows.
4. Sound statistics must come from `sound_stats.parquet`, fitted on the training window.

## 2a. Split construction

Splits are derived from the calendar, never from a random seed:

| Months ingested | train | val | test | recent |
|---|---|---|---|---|
| 1 | first 70% of the month | next 15% | final 15% | — |
| 2 | month 0 | first half month 1 | second half month 1 | — |
| 3+ | month 0 | first half month 1 | second half month 1 | month 2 |

Months after `month 2` are deliberately unused. `tests/test_splits.py` pins this
behaviour, including that no row appears in two splits.

## 3. Metrics and their exact definitions

Let `K` be the ranking cutoff and `π` the positive prevalence of the evaluation population.

| Metric | Definition |
|---|---|
| Precision@K | `|{top K} ∩ {positive}| / K` |
| Recall@K | `|{top K} ∩ {positive}| / |{positive}|` |
| Lift@K | `Precision@K / π` |
| PR-AUC | Average precision: `Σ (Rₙ − Rₙ₋₁) · Pₙ` over the score ranking |
| ROC-AUC | Probability a random positive outranks a random negative |
| Brier | `mean((p − y)²)` |
| Log loss | `−mean(y·log p + (1−y)·log(1−p))` |
| ECE@10 | `Σ (n_b / N) · |observed_b − predicted_b|` over 10 equal-width probability bins |

Ties in score are broken with a stable sort, so Precision@K is deterministic.

## 4. Uncertainty

PR-AUC is accompanied by a 95% percentile bootstrap interval:

- resampling unit: **`sound_id` cluster**, not the row;
- iterations: 200 (`configs/default.yaml`, key `model.bootstrap_iterations`);
- resamples where the bootstrap sample contains zero or all positives are discarded;
- interval: 2.5th and 97.5th percentiles of the retained statistics.

Clustering by sound is required because videos using the same sound share latent popularity and
are not independent draws.

## 5. Comparison rules

- All models in a run are scored on **identical rows** of the same split.
- Baselines are mandatory: random ranking, a train-prevalence constant, and two single-feature
  heuristics (`E0b_sound_popularity`, `E0c_duration_only`). Beating random proves little; beating
  the sound-popularity heuristic is the claim that matters, because sound features dominate tree
  importance.
- **`E0*` entries are never eligible to be champion.** They exist to be beaten. `select_champion`
  and the report renderer both skip them, pinned by `tests/test_champion_selection.py`.
- The champion is selected by test PR-AUC. Ties break on the best Precision@K.
- A model is never promoted on training or validation performance.

## 6. What would falsify the result

The reported performance should be treated as invalid if any of the following hold:

1. Any performance column appears in the feature matrix (guarded at runtime by
   `assert_no_performance_features`).
2. Thresholds or encoder vocabulary were fitted on rows outside the training window.
3. Test rows appear in any split used for fitting or calibration.
4. The `recent` split was used to select hyperparameters or models.
5. Precision@K is reported without the corresponding prevalence.

## 7. Known weaknesses of this protocol

- **Snapshot label, not fixed-age.** The label reflects views at an unknown, unreliable age.
  A high PR-AUC here means "this model ranks videos by eventual snapshot reach", which is a
  weaker claim than "this model forecasts seven-day virality".
- **Single forward holdout.** One `recent` month is a weak drift test. A rolling evaluation
  across several months would be stronger and is the recommended next step.
- **No creator-grouped evaluation yet.** Generalisation to unseen creators cannot be measured
  because the source has no creator identifier.
- **Residual correlation.** Clustering by sound handles one dependence structure; crawl batches
  are another and are only partially addressed.
