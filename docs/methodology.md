# Methodology

## 1. Problem statement

Given only information available before a TikTok video is published, estimate the probability
that it will land in the top 1% of observed views within its peer group. Then measure whether
that estimate is better than chance on data the model has never seen, in a strictly later
period.

## 2. What virality means here

Virality is defined **relative to peers**, not against a global view count:

```
viral(v) = observation_valid(v)
           AND views(v) >= p99{ views(u) : cohort(u) = cohort(v), u ∈ training window }
```

`cohort` is `country × type × observation_age_bucket`.

### Why the age bucket is part of the peer group

A video whose counters were read 3 hours after publication and one read 20 days after
publication are not comparable — the second has had far longer to accumulate views. Two bad
options present themselves: discard most of the corpus, or compare them naively. The first
throws away roughly 70% of rows; the second builds a label that mostly measures observation age.

ViralIQ instead makes age part of the definition. A video is only ever ranked against videos
observed at a similar age, using buckets `0-6h`, `6-24h`, `1-3d`, `3-7d`, `7-30d`. This keeps
the corpus and makes the comparison meaningful, at the cost of smaller cohorts (hence the
200-video minimum).

A cohort needs at least 200 training videos to receive its own threshold; otherwise the global
training-window p99 applies.

Why not a fixed threshold: observed p99 values in the 2026 partitions sit far below the
250,000–400,000 figures usually quoted. A fixed threshold would label almost nothing viral in
small markets and almost everything viral in large ones.

Why fitted on the training window only: fitting on the full dataset would let the label
definition absorb the test distribution, which inflates measured performance.

## 3. The observation clock problem

The dataset records one snapshot of engagement counters per video, stamped with
`stats_updated_at`. If that timestamp is trustworthy, `stats_updated_at − posted_at` is the
video's age when measured.

Measured on real partitions, a large share of rows have a **negative** age — counters stamped
before the video was published. The audit quantifies this per partition. Rows with an invalid
clock are:

- excluded from threshold fitting,
- excluded from positive labels,
- counted and reported (`pct_excluded_invalid_clock`), never silently repaired.

Consequence: `viral_24h` and `viral_7d` are emitted as NULL. The label actually used is
`viral_at_snapshot`, and every API response names it.

## 4. Feature construction

Only pre-publication signals enter the model:

| Group | Features |
|---|---|
| Content | caption length, word count, emoji count, exclamation and question counts, uppercase ratio, mention presence, hashtag count |
| Format | duration (seconds and log), photo-carousel flag |
| Timing | publication hour, day of week, weekend flag |
| Context | country, language, content type |
| Commercial | ad flag, shop-video flag, AI-generated flag, not-recommended flag, branded-content presence |
| Sound | original-sound flag, training-window usage (log), training-window median views (log) |

Deliberately excluded: all engagement counters, follower counts (absent from the source),
on-screen OCR text (noisy and unvalidated), and any sound or hashtag statistic computed after
the training window.

## 5. Validation protocol

Chronological, never random:

| Split | Composition | Used for |
|---|---|---|
| train | month M | thresholds, encoder vocabulary, sound statistics, model fitting |
| val | first half of month M+1 | probability calibration |
| test | second half of month M+1 | reported performance |
| recent | month M+2 (untouched) | forward-time robustness check |

The `recent` split is the only evidence that bears on whether the model survives a genuine time
shift. If it is present, its numbers are reported alongside the test numbers rather than
replacing them.

## 6. Models

| ID | Model | Role |
|---|---|---|
| E0_random_ranking | Random scores | Floor; any model must beat this |
| E0_majority_prevalence | Constant at train prevalence | PR-AUC degenerates to prevalence |
| E0b_sound_popularity | Score = sound's training-window median views | The heuristic that matters |
| E0c_duration_only | Score = −log(duration) | Weak-signal sanity check |
| E1 | Logistic regression (one-hot, balanced) | Linear baseline |
| E2 | Random forest (one-hot, balanced) | Nonlinear baseline |
| E3 | CatBoost (native categorical, balanced) | Primary candidate |
| E4 | LightGBM (native categorical, balanced) | Comparative gradient booster |

### Why a heuristic baseline, not just random

Beating random is not evidence of much. Since sound features dominate the tree importances, the
question a reader actually asks is *"does this beat just using the sound's historical popularity?"*
`E0b_sound_popularity` answers it directly, on the same rows and with the same metrics. It is the
baseline the model has to clear to justify its complexity.

`E0*` entries are **never eligible to be champion** — they exist to be beaten. Both
`select_champion` and the report renderer skip them, and `tests/test_champion_selection.py` pins
that behaviour, because a heuristic that outscored the models would otherwise have been shipped.

Class imbalance is handled with balanced class weights rather than resampling, which keeps the
calibration step meaningful. Probabilities are calibrated with **Platt scaling** (sigmoid) fitted on
the validation slice. Isotonic regression was tried first and rejected: on a ~1.4% base rate it fits
a step function whose plateaus map wide score ranges to one probability, so distinct videos received
identical scores.

CatBoost is treated as a **candidate**, not a presumed winner. The champion is whichever eligible
model achieves the best test PR-AUC.

## 7. Metrics

Primary (ranking):

- **Precision@K** — of the K highest-scored videos, what share are viral.
- **Recall@K** — of all viral videos, what share appear in the top K.
- **Lift@K** — Precision@K ÷ prevalence.
- **PR-AUC** — area under the precision-recall curve; the right summary for rare positives.

Secondary:

- **ROC-AUC** — discrimination, insensitive to base rate, so secondary here.
- **Brier score**, **log loss** — probability accuracy.
- **ECE (10 bins)** — calibration gap.

Not reported: accuracy. With a ~1% positive rate, a model that predicts "never viral" scores
99% accuracy while being useless.

Uncertainty: PR-AUC carries a percentile bootstrap interval that resamples whole sounds, because
videos sharing a sound are correlated and row-level resampling would understate variance.

## 8. Reference distributions for interpretation

At training time, validation scores and their realised views are stored as:

- a score-quantile grid, used to report a **percentile** for each prediction;
- equal-frequency score bands with p10/p50/p90 observed views, used to report an **expected
  view range**.

Both are computed on validation outcomes, not training outcomes, so the model is not graded on
data it memorised.

## 9. Explainability

Tree models use SHAP (`TreeExplainer`) to attribute each prediction to features. Attributions
are reported as "increases" or "decreases" the score. They describe model behaviour and are not
causal claims about why a video performed well.

## 10. Reproducibility

- Sampling is a deterministic hash of `video_id`, so a given fraction always yields the same rows.
- Encoder vocabularies, thresholds, sound statistics, reference distributions and the model
  binary are all persisted under `models/model_registry/<run_id>/`.
- `run.json` records splits, sample files, label diagnostics, metrics, feature importance and
  environment versions.
- Re-running `scripts/train.py` on the same sample reproduces the same run modulo library
  nondeterminism in the gradient boosters.
