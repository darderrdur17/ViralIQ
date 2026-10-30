# Executive Summary

**Run** `20261030T103039_primary_ef7c86` · 2026-10-30T10:34:12.868590+00:00

## What was built

A TikTok virality intelligence platform that scores videos from pre-publication metadata only, explains each score, and reports its own data freshness and model health. The inference path is a live API, not a notebook.

## What the model does

- Champion model: **E1_logistic_regression**
- Test PR-AUC: **0.0318** (95% CI 0.030–0.035)
- Random-ranking PR-AUC for reference: 0.0146
- Precision@K: **2.50%** vs 1.20% for random selection (**1.79× lift**)
- Base rate: 1.399%
- The champion is a **linear** model. The strongest gradient-boosted candidate (E3_catboost) reached 0.0250, so the extra capacity did not pay for itself on this feature set. That is a finding, not an oversight.

## Key caveat that shapes everything

5.673% of training rows were excluded because their observation clock was invalid: engagement statistics timestamped before publication (physically impossible), or older than the configured horizon.

Because of that, the shipped label is a cohort-relative snapshot label. It answers "does this video land in its peer group's top 1% of observed views", not "how many views will it have in seven days". Fixed-age labels are emitted as NULL rather than estimated.

## Recommendation

Treat the current model as a validated **ranking prior** for content triage, and invest in an observation pipeline before making forecasting claims. The highest-value next step is not a better model — it is repeated measurements of the same videos.
