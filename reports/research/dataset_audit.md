# Dataset Audit — TikTok 5.6B Videos

Generated: `2026-10-30T09:55:39.183967+00:00`  
Method: remote Parquet queries via DuckDB. No partition was downloaded in full.

## 1. Repository

| Property | Measured value |
|---|---|
| Repository | `datasocial/tiktok-5.6B-videos` |
| Licence | CC BY-NC 4.0 |
| Monthly partitions | 148 |
| Total size | 460.48 GB |
| Earliest partition | `2014-07` |
| Latest partition | `2026-10` (0.19 GB) |

## 2. Coverage and freshness

| Measure | Value |
|---|---|
| Latest publication timestamp | `2026-10-03T13:00:00+00:00` |
| Latest statistics timestamp | `2026-10-03T13:11:38+00:00` |
| Publication delay at audit time | **26.9 days** |
| Operating mode implied by that delay | `historical` |

The delay above is the gap between the newest video in the dataset and the audit clock.
That delay places the platform in `historical` mode (live ≤ 2 days, recently refreshed ≤ 14 days). The application derives this from the same rule, so the report and the running service cannot disagree.

## 3. Partition profiles

| Month | Rows | Median views | p99 views | % zero views | % negative obs. age | Distinct stat timestamps |
|---|---|---|---|---|---|---|
| 2026-09 | 98,594,752 | 405 | 247,636 | 0.7857% | **5.7125%** | 528,588 |
| 2026-10 | 1,734,031 | 348 | 151,936 | 0.951% | **34.718%** | 73,582 |

## 4. Critical integrity finding: the observation clock is unreliable

`stats_updated_at` is documented as the time engagement counts were read. For a large
share of rows it is *earlier than the video's own publication timestamp*, which is
physically impossible: a video cannot accumulate views before it exists.

- **2026-09**: 5.7125% of 98,594,752 rows have a negative observation age (median age 283.31 h, p99 696.04 h).
- **2026-10**: 34.718% of 1,734,031 rows have a negative observation age (median age 14.04 h, p99 49.5 h).

Rows with a negative age also under-report reach, which is consistent with them being a
stale crawl batch whose counters were never refreshed after publication.

### Consequences

1. `observation_age_hours` cannot be trusted as a normalising variable.
2. Fixed-age targets (`viral_24h`, `viral_7d`) **cannot be constructed** from this source.
3. The only defensible retrospective label is a cohort-relative snapshot label, which is
   what ViralIQ trains on — and it is named accordingly everywhere it appears.
4. Rows with an invalid observation clock are excluded from label construction, not
   silently repaired.

## 5. What this dataset cannot provide

| Missing capability | Consequence |
|---|---|
| Repeated observations per video | No velocity features; early-traction engine blocked |
| Creator follower counts | Creator features limited to prior publishing history |
| Video and audio files | No visual or audio embeddings |
| Guaranteed live feed | Platform operates in historical-data mode |

## 6. Threshold implication

Observed p99 view counts sit far below the 250,000 / 400,000 figures often quoted for
TikTok virality. A hard-coded global threshold would mislabel the majority of the corpus.
ViralIQ therefore derives its threshold per country-and-format cohort at the 99th
percentile, fitted on the training window only.
