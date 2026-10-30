# Three-Month Experiment: Why the Forward Holdout Cannot Be Populated

**Question.** The evaluation protocol defines a `recent` split — a month after the test month,
never touched during fitting — as the only real evidence that a model survives a time shift. It was
never exercised, because only two months had been ingested. Does adding a third, older month fix
that?

**Answer: no.** The `recent` path works, but the data cannot populate it soundly. Adding 2026-08 as
the training month degrades the label rather than strengthening the evaluation, and it does so
silently unless you measure it.

---

## What was run

Three monthly partitions ingested:

| Partition | Sampling | Rows |
|---|---|---:|
| 2026-08 | 1% | 1,118,592 |
| 2026-09 | 1% | 987,260 |
| 2026-10 | 100% | 1,734,031 |

`temporal_slices` then produced the intended layout — so the code path is correct:

```
train  = 2026-08            300,000 rows
val    = 2026-09 (1st half) 200,001 rows
test   = 2026-09 (2nd half) 199,999 rows
recent = 2026-10            300,000 rows   <- first time this split has been populated
```

## What went wrong

The result is not merely weaker — it is degenerate.

| Measure | Two months (train 09 → test 10) | Three months (train 08 → test 09) |
|---|---:|---:|
| Training rows dropped on the observation clock | 5.7% | **52.8%** |
| Fitted global p99 threshold | 258,801 views | **58,526 views** |
| Cohort thresholds fitted | 130 | 78 |
| Test-set prevalence | 1.40% | **4.29%** |
| Champion PR-AUC | 0.0318 | 0.0571 |
| Champion ROC-AUC | 0.742 | 0.597 |
| CatBoost PR-AUC | 0.0250 | **0.0364** |
| Random-ranking PR-AUC | 0.0146 | 0.0432 |

CatBoost scoring *below* random ranking is the tell. The model has not got worse; the label has
become noise.

## Why

Observation age — the gap between publication and when the engagement counters were read — grows
with a partition's distance from the crawl date. Measured per month:

| Month | Median observation age | Rows beyond the 720 h window | Rows with a negative age |
|---|---:|---:|---:|
| 2026-08 | **756 h** | **53.0%** | 0.00% |
| 2026-09 | 288 h | 0.1% | 5.70% |
| 2026-10 | 14 h | 0.0% | 34.70% |

August's median age sits just past the 30-day validity window, so **half the month is discarded**.
What remains is not a random half: it is the subset whose counters happened to be refreshed late,
which is not representative of August's view distribution. The p99 fitted on that biased subset
collapses to 58,526 views. Applied to September — a month whose own p99 is 240,927 — four times as
many videos as intended clear the bar, and the positive class stops meaning "top 1%".

This is the same structural defect documented in `dataset_audit.md`, seen from a different angle:
one snapshot per video, with a crawl timestamp that makes older partitions progressively less
usable. The observation clock does not just prevent *forecasting*; it also bounds how far back
training can reach.

## Consequences adopted

1. **The primary evaluation stays on two months** (train 2026-09 → test 2026-10), the only
   configuration where the label is sound. Reproduce with:

   ```bash
   python scripts/train.py --months 2026-09 2026-10
   ```

2. **`--months` exists** so the split can be scoped deliberately rather than inheriting whichever
   partitions happen to be on disk.

3. **Training now refuses** when more than 25% of the training split falls past the observation-age
   window. `run_training` raises `DegradedLabelError`, `scripts/train.py` exits with code 2, and the
   message names both remedies (`--months` to scope, `--allow-degraded` to override). The check is
   keyed on *rows past the window* specifically, not on all invalid rows: a negative observation age
   is a stale crawl batch spread across the month, whereas rows past the window are the biased
   subsample that drags the threshold. Without that distinction the guard would also block the demo
   fixture, whose October rows are 35% negative-age but 0% past the window.

4. **The `recent` split stays in the protocol** and remains unpopulated in the shipped run. That is
   an honest gap, not an oversight: the code supports it, the data does not.

## What would actually fix it

Partitions crawled close together in time, or — better — repeated observations of the same videos.
A dataset that re-reads counters daily would make both the `recent` holdout and the early-traction
engine viable. Nothing about the modelling approach needs to change; the data collection does.

*Run referenced: `20261030T121526_primary_d3d6bd` (three-month), superseded by the two-month
primary. Raw metrics retained in this document because the run artefacts were replaced.*
