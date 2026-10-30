# Threshold Sensitivity

Champion model: **E1_logistic_regression**, held fixed. Only the label definition moves.

Thresholds are refitted on the training window at each percentile, then applied to the
test month unchanged — the same protocol as the main evaluation.

| Label | Prevalence | Positives | PR-AUC | ROC-AUC | Precision@1000 | Recall@1000 | Lift@1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| top 5.0% | 7.53% | 8,812 | 0.0999 | 0.601 | 8.40% | 1.0% | 1.12× |
| top 1.0% | 1.40% | 1,637 | 0.0318 | 0.742 | 2.50% | 1.5% | 1.79× |
| top 0.5% | 0.66% | 768 | 0.0157 | 0.762 | 1.30% | 1.7% | 1.98× |

## How to read this

- A looser threshold (top 5%) produces more positives, so **recall rises and precision
  falls**. Neither number is better in isolation; they answer different questions.
- PR-AUC shifts because the positive class changed, not because the model changed.
  Comparisons across rows are comparisons of *label definitions*, not of models.
- The shipped default is the strictest of these (top 1%), chosen because a content team
  acting on a shortlist cares about precision at the top of the ranking.
- Realised prevalence does not equal the nominal percentile: thresholds are fitted on the
  training month and applied to a later month whose view distribution differs. At p99 the
  realised rate lands close to 1%; at p95 it overshoots. That gap is a measure of
  month-to-month drift, not an error in the fitting.
- Lift and ROC-AUC **improve** as the label tightens, because the signal concentrates at the
  very top of the ranking — which is exactly where a shortlist operates.

## Reproduce

```bash
python scripts/threshold_sweep.py
```
