# Data Sources

## 1. Primary: Hugging Face `datasocial/tiktok-5.6B-videos`

| Property | Value |
|---|---|
| Repository | `datasocial/tiktok-5.6B-videos` |
| Format | Parquet, one file per month |
| Partitions | 148 (`2014-07` → `2026-10`) |
| Total size | 460.48 GB (measured from the repository tree API) |
| Licence | **CC BY-NC 4.0** — non-commercial |
| Access | Public, not gated |
| Content | Metadata and engagement counters. **No video or audio files.** |

### What it supports

- Pre-publication feature engineering (caption, hashtags, duration, format, timing, language,
  country, commercial flags, sound id).
- Retrospective, cohort-relative snapshot classification.
- Trend analysis within the observed window.

### What it cannot support

- **Fixed-age outcomes.** A single snapshot per video with an unreliable timestamp means
  `viral_24h` and `viral_7d` cannot be constructed. Measured evidence is in
  `reports/research/dataset_audit.md`.
- **Velocity features.** No second observation exists.
- **Creator features beyond publishing history.** No follower counts.
- **Visual or audio modelling.** No media.

### Licensing constraint

CC BY-NC 4.0 prohibits commercial use. Any commercial deployment needs a separate licence.
The dataset publisher advertises commercial access, creator profiles and daily updates through
a commercial provider; that route is unverified here and must be contracted before use.

---

## 2. TikTok Research API

- Endpoint documentation: `https://developers.tiktok.com/products/research-api/`
- Provides views, likes, shares, comments, hashtags and duration for public videos.
- **Access requires approved researcher eligibility in supported regions.**
- Status in this project: **not integrated.** No approval is held, so no code path claims it
  works. A connector stub is documented but intentionally not presented as functional.

Do not attempt to circumvent eligibility requirements.

---

## 3. TikTok Display API

- Endpoint documentation: `https://developers.tiktok.com/products/display-api/`
- Displays a **consenting creator's own** content. It is not a public video search service.
- Status: **not integrated.** Useful only for first-party creator analytics with explicit
  authorisation.

---

## 4. TikTok Creative Center

- `https://ads.tiktok.com/business/creativecenter/`
- Useful for current hashtag and sound trend research through the official interface.
- Status: **not integrated.** There is no general-purpose public API; automated access must
  respect the site's terms.

---

## 5. Licensed data providers

- `https://datasocial.ai/` (publisher of the primary dataset)
- Potential route to fresher data and creator-level attributes.
- Status: **not integrated.** Requires a commercial agreement; access, cost and licensing are
  unverified.

---

## 6. Manual import (implemented fallback)

`POST /api/v1/data/import` (local path) and `POST /api/v1/data/upload` (file) accept Parquet or
CSV, validate the required schema, copy into `data/samples/`, and record an ingestion event that
updates the freshness report.

This is the supported way to add newer data without any API credentials.

---

## Freshness policy

| Operating mode | Condition |
|---|---|
| `live` | Latest publication within 7 days of now |
| `recently_refreshed` | 7–30 days |
| `historical` | More than 30 days |
| `unavailable` | No ingestion recorded |

The mode is computed from measured timestamps, never configured by hand. The dashboard renders
it as a badge and the API exposes it at `/api/v1/health` and `/api/v1/data/status`.

## Attribution

Source dataset: `datasocial/tiktok-5.6B-videos`, CC BY-NC 4.0. Figures quoted in reports are
measured at run time and the measurement timestamp is recorded alongside them.
