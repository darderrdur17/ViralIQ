# Executive Summary

**Run** `20261030T132450_candidate_297080` · 2026-10-30T13:25:00.412515+00:00

## What was built

A TikTok virality intelligence platform that scores videos from pre-publication metadata only, explains each score, and reports its own data freshness and model health. The inference path is a live API, not a notebook.

## What the model does

- Champion model: **E2_random_forest**
- Test PR-AUC: **0.0062** (95% CI 0.004–0.011)
- Random-ranking PR-AUC for reference: 0.0039
- Precision@K: **0.70%** vs 0.30% for random selection (**1.70× lift**)
- Base rate: 0.411%
- The champion is a **linear** model. The strongest gradient-boosted candidate (E4_lightgbm) reached 0.0053, so the extra capacity did not pay for itself on this feature set. That is a finding, not an oversight.

## Key caveat that shapes everything

5.685% of training rows were excluded because their observation clock was invalid: engagement statistics timestamped before publication (physically impossible), or older than the configured horizon.

Because of that, the shipped label is a cohort-relative snapshot label. It answers "does this video land in its peer group's top 1% of observed views", not "how many views will it have in seven days". Fixed-age labels are emitted as NULL rather than estimated.

## Recommendation

Treat the current model as a validated **ranking prior** for content triage, and invest in an observation pipeline before making forecasting claims. The highest-value next step is not a better model — it is repeated measurements of the same videos.
