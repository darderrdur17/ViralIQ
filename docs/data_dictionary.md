# Data Dictionary

Column types below were read from a live `DESCRIBE` against the `2026-10` partition.

## 1. Source columns (31)

| Column | Type | Meaning | Used as |
|---|---|---|---|
| `video_id` | UBIGINT | TikTok video identifier | Key, sampling hash |
| `posted_at` | TIMESTAMPTZ | Publication time (UTC) | Split, timing features, observation clock |
| `country` | VARCHAR | Region TikTok reports | Categorical feature, cohort key |
| `language` | VARCHAR | Detected language (`un` = undetermined) | Categorical feature |
| `type` | VARCHAR | Content type | Categorical feature, cohort key |
| `duration_ms` | UINTEGER | Duration in milliseconds | `duration_s`, `log_duration` |
| `image_count` | UTINYINT | Number of images (photo carousels) | `is_photo_carousel` |
| `width` / `height` | USMALLINT | Pixel dimensions | Not used |
| `caption` | VARCHAR | Caption text | Caption statistics |
| `hashtags` | VARCHAR[] | Hashtag strings without `#` | `hashtag_count`, trend analysis |
| `hashtag_ids` | UBIGINT[] | Hashtag identifiers | Not used |
| `mentions` | UBIGINT[] | Mentioned user identifiers | Not used |
| `on_screen_text` | VARCHAR[] | OCR text from the video | Not used (noisy, unvalidated) |
| `sound_id` | UBIGINT | Sound identifier | Sound features, clustering for CIs |
| `is_ad` | UTINYINT | Advertising flag | Boolean feature |
| `branded_content` | VARCHAR | Branded-content disclosure | `has_branded_content` |
| `product_id` | UBIGINT | Shop product identifier | Not used |
| `seller_id` | UBIGINT | Shop seller identifier | Not used |
| `is_shop_video` | UTINYINT | Commerce video flag | Boolean feature |
| `is_ai_generated` | UTINYINT | AI-generated content flag | Boolean feature |
| `not_recommended_for_fyp` | UTINYINT | Excluded from For You feed | Boolean feature |
| `sound_muted` | UTINYINT | Sound removed | Not used |
| `views` | UBIGINT | Observed view count | **Label only — never a feature** |
| `likes` | UBIGINT | Observed likes | **Label diagnostics only** |
| `comments` | UBIGINT | Observed comments | **Label diagnostics only** |
| `shares` | UBIGINT | Observed shares | **Label diagnostics only** |
| `saves` | UBIGINT | Observed saves | **Label diagnostics only** |
| `downloads` | UBIGINT | Observed downloads | Not used |
| `engagement_rate` | FLOAT | Platform-reported engagement rate | **Never used** |
| `stats_updated_at` | TIMESTAMPTZ | When counters were read | Observation clock |

> `views`, `likes`, `comments`, `shares`, `saves`, `downloads` and `engagement_rate` are
> **performance columns**. `src.labeling.labels.assert_no_performance_features` raises if any
> of them reach a feature frame.

## 2. Derived columns

| Column | Type | Definition |
|---|---|---|
| `observation_age_hours` | float | `(stats_updated_at − posted_at) / 3600`. Negative = counters timestamped before publication (impossible). |
| `observation_age_bucket` | string | `invalid` / `0-6h` / `6-24h` / `1-3d` / `3-7d` / `7-30d` / `over30d`. Part of the peer group, so videos are only ranked against others observed at a similar age. |
| `observation_valid` | bool | `0 ≤ observation_age_hours ≤ 720` and `stats_updated_at` is not null. |
| `viral_view_threshold` | float | Cohort p99 fitted on the training window, or the global p99 fallback. || `viral_at_snapshot` | bool | `observation_valid AND views ≥ viral_view_threshold`. |
| `performance_percentile` | float | Within-cohort percentile rank of observed views. |
| `viral_24h` | bool \| null | **Always NULL.** No repeated observations exist. |
| `viral_7d` | bool \| null | **Always NULL.** No repeated observations exist. |
| `future_view_count` | int \| null | **Always NULL.** Snapshot views are not a future count. |
| `label_basis` | string | `snapshot_top1pct_cohort_relative` or `excluded_invalid_observation_clock`. |

## 3. Model features (26)

### Numeric (23)

`duration_s`, `log_duration`, `is_photo_carousel`, `caption_len`, `caption_words`,
`caption_emoji_count`, `caption_exclaim_count`, `caption_question_count`,
`caption_upper_ratio`, `caption_has_mention`, `hashtag_count`, `log_hashtag_count`,
`posted_hour`, `posted_dow`, `is_weekend`, `is_ad`, `is_shop_video`, `is_ai_generated`,
`not_recommended_for_fyp`, `has_branded_content`, `sound_is_original`,
`sound_train_usage_log`, `sound_train_median_views_log`

### Categorical (3)

`country`, `language`, `type` — vocabulary capped at the 60 most frequent values per column,
fitted on training rows only; unseen values map to `__unseen__`.

### Frozen-at-training-window columns

`sound_train_usage_log` and `sound_train_median_views_log` come from
`fit_sound_stats(training_rows)` and are persisted in
`models/model_registry/<run_id>/sound_stats.parquet`. At inference, an unknown sound receives 0.

## 4. Application tables

### `predictions`

| Column | Type |
|---|---|
| `prediction_id` | VARCHAR (UUID, primary key) |
| `created_at` | TIMESTAMPTZ |
| `task` | VARCHAR (`prepublication` \| `early_traction`) |
| `model_version` | VARCHAR (`<model>@<run_id>`) |
| `viral_probability` | DOUBLE |
| `virality_tier` | VARCHAR |
| `predicted_percentile` | DOUBLE |
| `payload` | JSON (the exact request) |
| `response` | JSON (the exact response) |

### `video_observations`

| Column | Type |
|---|---|
| `observation_id` | VARCHAR (UUID) |
| `video_id` | UBIGINT |
| `observed_at` | TIMESTAMPTZ |
| `observation_age_hours` | DOUBLE |
| `views`, `likes`, `comments`, `shares`, `saves` | UBIGINT |
| `source` | VARCHAR |
| `raw` | JSON |

Populating this table is the prerequisite for the early-traction engine and for any genuine
seven-day forecasting claim.

### `ingestion_runs`

| Column | Type |
|---|---|
| `run_id` | VARCHAR |
| `started_at` / `finished_at` | TIMESTAMPTZ |
| `source` | VARCHAR |
| `rows_ingested` | BIGINT |
| `status` | VARCHAR |
| `detail` | JSON |
