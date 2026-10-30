# ViralIQ — TikTok Virality Intelligence & Prediction Platform

## Project Mission

Build a professional, production-oriented TikTok Virality Intelligence Platform that combines large-scale data engineering, machine learning, trend intelligence, predictive analytics, explainable AI and interactive dashboard visualisation.

The objective is to develop a statistically rigorous machine learning system capable of identifying and predicting TikTok video virality using historical and continuously refreshed information.

The project is inspired by Maggie's reported CatBoost experiment using a 5.6-billion-record TikTok dataset, with a reported 6.9% hit rate compared with a 2.0% baseline without engagement metrics.

Treat those reported figures as research hypotheses to investigate and reproduce, not guaranteed results.

**The final product must go substantially beyond replicating that experiment.**

Build a complete platform that can:

1. Discover patterns associated with TikTok virality.
2. Predict the probability of a video becoming viral.
3. Estimate video performance and ranking within its peer group.
4. Identify emerging trends and accelerating sounds and hashtags.
5. Incorporate recent 2025–2026 TikTok data.
6. Monitor data freshness and changing platform behaviour.
7. Compare different machine learning models.
8. Generate explainable predictions and research-quality reports.
9. Provide actual predictions through a functioning backend API.
10. Automatically evaluate whether model performance deteriorates over time.

---

# 1. PROJECT DIRECTORY AND FILE MANAGEMENT

Create the entire project inside:

`/Users/derr/Documents/Random/ViralIQ`

All source code, datasets, scripts, model artifacts, configuration files, notebooks, reports, documentation, tests and dashboard files must be saved inside this directory.

Establish a professional, modular architecture.

Suggested structure:

ViralIQ/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
│
├── data/
│   ├── raw/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   ├── samples/
│   └── metadata/
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_catboost_experiment.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── labeling/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   ├── inference/
│   └── monitoring/
│
├── api/
│   ├── main.py
│   ├── routes/
│   ├── schemas/
│   └── services/
│
├── dashboard/
│
├── database/
│   ├── migrations/
│   └── schemas/
│
├── models/
│   ├── trained/
│   ├── checkpoints/
│   └── model_registry/
│
├── reports/
│   ├── research/
│   ├── evaluation/
│   ├── insights/
│   └── executive/
│
├── configs/
├── scripts/
├── tests/
└── docs/
    ├── architecture.md
    ├── data_sources.md
    ├── data_dictionary.md
    ├── methodology.md
    ├── model_card.md
    ├── evaluation_protocol.md
    └── limitations.md

Do not download or commit the complete 460 GB dataset into Git.

Raw datasets, model artifacts and credentials must be managed separately from version-controlled source code.

---

# 2. TEAM STRUCTURE AND RESPONSIBILITIES

Operate as a coordinated six-member AI-native data science and engineering team.

## NOVA — Chief Data Orchestrator / Project Lead

Primary responsibility: Overall project execution and coordination.

Responsibilities:
- Establish the project roadmap.
- Divide tasks between the other five agents.
- Establish technical architecture and dependencies.
- Coordinate data engineering, ML development and frontend implementation.
- Review outputs and resolve conflicting recommendations.
- Ensure all components integrate correctly.
- Monitor development progress.
- Challenge unsupported assumptions.
- Ensure the final application is genuinely functional.

Nova must not simply delegate tasks and report completion.

Nova must verify that the individual components work together.

## SAGE — Data Scientist / Machine Learning Engineer

Primary responsibility: Predictive modelling and statistical validity.

Responsibilities:

- Develop the complete ML pipeline.
- Investigate Maggie's reported CatBoost methodology.
- Establish baseline performance.
- Develop CatBoost as the primary candidate model.
- Benchmark against Logistic Regression, Random Forest and LightGBM.
- Investigate content embeddings.
- Engineer point-in-time features.
- Perform hyperparameter optimisation.
- Evaluate class imbalance.
- Calibrate predictive probabilities.
- Implement SHAP explainability.
- Perform temporal validation.
- Investigate performance differences across 2024, 2025 and 2026.
- Save trained models and reusable preprocessing artifacts.

Sage must distinguish genuine predictive capability from correlations caused by data leakage.

## QUINN — SQL Analyst / Data Engineering Specialist

Primary responsibility: Large-scale dataset processing and data infrastructure.

Responsibilities:

- Integrate the Hugging Face dataset.
- Build efficient Parquet ingestion.
- Use DuckDB and Polars for analytical processing.
- Establish PostgreSQL tables where appropriate.
- Implement feature extraction and data transformations.
- Establish data quality checks.
- Build repeatable ETL pipelines.
- Handle incremental data ingestion.
- Develop SQL queries for trend and engagement analysis.
- Maintain a reproducible data dictionary.

Avoid loading the entire dataset into memory.

## RIKA — Knowledge Retrieval / Research Specialist

Primary responsibility: Dataset research, source verification and methodological investigation.

Responsibilities:

- Verify all relevant datasets.
- Investigate alternative TikTok datasets.
- Review academic literature about short-form video virality.
- Investigate TikTok trend dynamics.
- Research platform and API limitations.
- Verify licensing and permitted data acquisition methods.
- Investigate CatBoost, multimodal prediction and modern NLP approaches.
- Document research findings with source references.
- Identify changes in TikTok content and engagement behaviour.
- Review potential data leakage and survivorship bias.

Do not invent API access, datasets, papers or research results.

## DASH — Visual Analytics Designer / Frontend Engineer

Primary responsibility: Interactive analytics dashboard.

Responsibilities:

- Build a polished, professional dashboard.
- Establish consistent design language.
- Connect the frontend to actual backend endpoints.
- Implement prediction visualisations.
- Implement trend monitoring.
- Display model comparisons.
- Display model explanations.
- Implement filters for time, region, language and content type.
- Support data freshness and model status indicators.
- Provide meaningful loading, error and empty states.

Do not build a static dashboard populated with fabricated metrics.

Every important interactive feature must communicate with a real backend endpoint.

## REX — Research Report Composer / Analytics Specialist

Primary responsibility: Reporting, insights and communication.

Responsibilities:

- Produce executive summaries.
- Document modelling methodology.
- Explain experimental results.
- Generate feature importance reports.
- Identify significant temporal changes.
- Document model limitations.
- Produce model performance comparisons.
- Generate weekly trend reports.
- Produce reproducible analytical reports with charts and tables.

All findings must be supported by actual analytical results.

---

# 3. DATASET ACQUISITION AND VERIFICATION

## Primary dataset

Hugging Face:

https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos

Explore:

https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos/tree/main/data

This dataset contains billions of TikTok video records in monthly Parquet partitions.

Available fields include:

- video_id
- posted_at
- country
- language
- type
- duration_ms
- caption
- hashtags
- hashtag_ids
- mentions
- on_screen_text
- sound_id
- is_ad
- branded_content
- product_id
- seller_id
- is_shop_video
- is_ai_generated
- not_recommended_for_fyp
- views
- likes
- comments
- shares
- saves
- downloads
- engagement_rate
- stats_updated_at

## Dataset research requirements

Before modelling, Rika and Quinn must:

1. Verify the current repository and schema.
2. Inspect available monthly Parquet partitions.
3. Determine the actual latest publication date.
4. Determine the latest metric observation date.
5. Calculate dataset freshness and observation delays.
6. Investigate missing values and inconsistent records.
7. Identify historical archive limitations.
8. Verify licensing and permitted usage.
9. Confirm the actual availability of recent 2025 and 2026 records.
10. Produce a documented dataset audit.

Never assume the dataset is continuously updated just because its repository was recently updated.

Use DuckDB and Polars to inspect and process selected partitions without downloading the entire dataset.

Prioritise 2024–2026 records for current modelling, while investigating whether older historical records add predictive value.

## Additional data sources

Investigate the following:

1. TikTok Research API:
https://developers.tiktok.com/products/research-api/

2. TikTok Display API:
https://developers.tiktok.com/products/display-api/

3. TikTok Creative Center:
https://ads.tiktok.com/business/creativecenter/

4. Potential licensed daily data sources:
https://datasocial.ai/

Build independent source adapters so the application can support multiple sources.

Research API access must only be used if the project satisfies TikTok's eligibility and approval requirements.

The Display API must use the appropriate creator authorization.

Never bypass API restrictions, authentication, anti-bot measures or access controls. Do not assume that a third-party scraping service is authorised to provide any particular data.

Provide Parquet and CSV import functionality as a fallback.

## Critical data freshness functionality

Implement a data freshness monitoring service.

It must report:

- Latest available publication timestamp.
- Latest recorded engagement observation.
- Last successful ingestion.
- Number of new records.
- Source availability.
- Data collection delay.
- Number of missing or invalid records.
- Dataset schema changes.
- Last successful model retraining.
- Current model version.

The dashboard must distinguish between live, recently refreshed and historical datasets.

If live data is unavailable, clearly identify the application as operating in historical-data mode.

Never represent historical measurements as live TikTok statistics.

---

# 4. TARGET DEFINITION AND VIRALITY METHODOLOGY

Virality must be defined statistically rather than using an arbitrary universal view threshold.

Develop at least two separate prediction tasks.

## Task A: Pre-publication virality prediction

Objective:

Predict whether a newly published video will enter the top 1% of its comparable video cohort within seven days.

Permitted features must be available before publication.

Examples:

- Caption characteristics.
- Caption embeddings.
- Hashtags.
- Language.
- Duration.
- Publication time.
- Historical sound popularity.
- Creator's prior publishing history.
- Historical creator performance derived only from earlier videos.

Do not use subsequent views, likes, shares or comments.

## Task B: Early-traction virality prediction

Objective:

Predict whether a video will become viral using information available shortly after publication.

Potential observation windows:

- 30 minutes.
- 1 hour.
- 6 hours.
- 24 hours.

Possible features:

- Initial views.
- View growth velocity.
- Like rate.
- Share rate.
- Comment rate.
- Early engagement growth.
- Sound and hashtag momentum.

These features are permitted only when the prediction is made after they have been observed.

## Label engineering

Create separate fields for:

- viral_at_snapshot
- viral_24h
- viral_7d
- future_view_count
- performance_percentile
- observation_age_hours

The existing Hugging Face dataset has engagement statistics associated with stats_updated_at. Do not assume that it contains repeated observations at 24 hours or seven days.

Build genuine fixed-age targets only when suitable observations are available.

For the retrospective dataset, use a clearly named snapshot-based proxy label where fixed-age outcomes cannot be established.

Never silently mix the two tasks or present a retrospective classification result as prospective forecasting accuracy.

---

# 5. FEATURE ENGINEERING

Investigate the predictive contribution of the following feature groups.

## Content features

- Caption length.
- Caption complexity.
- Caption sentiment.
- Caption language.
- Caption embeddings.
- Hashtag count.
- Hashtag combinations.
- Topic classifications.
- On-screen text.
- Text originality and similarity.

## Publishing features

- Hour of publication.
- Day of week.
- Weekend versus weekday.
- Country.
- Language.
- Historical posting patterns.

## Sound features

- Sound identifier.
- Historical usage volume.
- Recent usage acceleration.
- Sound popularity by country.
- Sound popularity by language.
- Sound adoption velocity.

Compute sound popularity using only videos published before the prediction timestamp.

Do not include future sound usage.

## Creator features

Where legally and technically available:

- Historical video count.
- Historical median views.
- Previous viral-video frequency.
- Publishing consistency.
- Historical average engagement.

Creator features must be calculated using earlier information only.

The primary Hugging Face dataset does not directly provide historical follower counts. Do not fabricate them.

## Optional multimodal extension

Investigate:

- Video frame embeddings.
- CLIP or comparable visual encoders.
- Audio representations.
- Video scene characteristics.
- Visual content categorisation.

This is a separate extension because the primary dataset contains metadata, not the underlying video and audio files.

Only implement this when there is an authorised media source.

---

# 6. MODEL DEVELOPMENT AND EXPERIMENT TRACKING

Implement reproducible experiments.

## Baselines

- Majority-class classifier.
- Random-ranking baseline.
- Logi
