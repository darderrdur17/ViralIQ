TikTok Viral Prediction Lab — Project Blueprint

Recommended approach Build this as a 2026-ready TikTok Virality Intelligence Platform, rather than simply reproducing Maggie’s CatBoost experiment.
The central objective should be:
Predict which TikTok videos are likely to become viral using information available before or shortly after publication, identify the content and distribution characteristics associated with virality, and continuously test whether those predictions remain effective as TikTok trends evolve.
There are three distinct capabilities to build:
Historical prediction: Learn from historical TikTok videos and their observed performance.
Current trend intelligence: Incorporate recent videos, trending sounds, hashtags, posting patterns, and content characteristics.
Prospective prediction: Score newly observed videos using only information that would genuinely have been available at prediction time.
An important distinction: a model trained on historical engagement counts can identify historical viral videos, but that does not automatically make it capable of predicting a newly published video's future performance. We need timestamped observations and a proper prediction horizon to prove that capability.
1. Dataset selection and availability
Hugging Face Logo Png | Hugging Face Logo Vector  - freepng
Primary dataset: TikTok 5.6B Videos
Hugging Face · datasocial
Verified repository
5.598 billion
Reported dataset rows
Attribute
Details
Format
Parquet
Repository size
460 GB
Partitioning
One Parquet file per month
Coverage
Historical and recent TikTok records
Engagement
Views, likes, comments, shares, saves
Content metadata
Captions, hashtags, sounds, country, language
License
CC BY-NC 4.0
The dataset reports these figures on its current Hugging Face page. 
Datasets at Hugging Face
+1
What information can we extract?
Field
Why it matters
posted_at
Publication date and time
country
Geographic segmentation
language
Audience and language differences
caption
Textual content analysis
hashtags
Topic and trend identification
on_screen_text
Text embedded in the content
duration_ms
Video duration
sound_id
Sound-level popularity analysis
is_ad
Advertising versus organic content
branded_content
Commercial content classification
views
Observed video reach
likes
Engagement
comments
Audience interaction
shares
Distribution and sharing behaviour
saves
Content retention signal
stats_updated_at
When the engagement counts were updated
The particularly useful feature is stats_updated_at. Views and engagement metrics are recorded as of that timestamp, so we can calculate the approximate age of a video when its statistics were observed.

Datasets at Hugging Face
+1
One important limitation: the dataset contains structured text and metadata, not the actual video files. It also does not provide historical follower counts for creators. Therefore, visual embeddings, audio embeddings and creator follower counts will require additional, properly sourced data.
How to handle 460 GB without downloading everything
I strongly recommend DuckDB and Polars rather than loading the entire dataset into Pandas.
Recommended data access stack
DuckDB
Remote Parquet querying and analytical SQL
Recommended
Polars
Lazy processing, filtering and transformation
Apache Parquet
Columnar storage for efficient data access
Both DuckDB and Polars support direct Hugging Face dataset access. 
DuckDB
+2
For the first experiment, have the agents inspect the newest available monthly partitions and use a manageable, stratified sample. Prioritise 2024–2026 data, with separate emphasis on 2026. Do not download 460 GB simply to get started.
2. How to make it genuinely current in October 2026
This is the most important architectural decision.
The existing dataset is useful for historical modelling, but a single engagement snapshot cannot tell us how quickly a video gained its views.
For example:
Video
Age when statistics were recorded
Views
A
2 hours
50,000
B
7 days
500,000
C
30 days
800,000
A naïve model might consider C the most successful. However, A may actually be experiencing much faster initial growth.
I would build two separate prediction engines.
Engine 01
Pre-Publication Virality Predictor
Before engagement
Predict potential virality based on information available before publication.
Features:
Caption and language embeddings.
Hashtag selection and topic.
Video duration and format.
Publication time and day.
Historical sound popularity.
Creator's previous publishing history.
Engine 02
Early-Traction Virality Predictor
After publication
Predict whether a video will become viral using its early performance.
Features:
Views after 30 minutes, 1 hour and 6 hours.
Early view velocity.
Like and share rates.
Comment growth.
Early audience retention, where legitimately available.
Sound and hashtag momentum.
This separation is important because the second engine is allowed to use engagement metrics observed before its prediction timestamp, whereas the first engine must not use them at all.
The ongoing data collection system
I recommend maintaining a video observation table rather than overwriting video statistics.
Example: video observation timeline
Initial observation — T0
Capture publication metadata, caption, duration, sound and initial counts where available.
30-minute observation
Record initial distribution.
1-hour / 6-hour observations
Measure early acceleration and engagement.
24-hour observation
Measure first-day performance.
72-hour / 7-day observations
Establish subsequent growth and the actual prediction outcome.
The system should store each observation with its actual collection timestamp, source and video ID. It must never manufacture missing historical observations.
Where will current data come from?
Source
Application
Limitation
Hugging Face 5.6B
Historical training and research
Not a guaranteed live feed
TikTok Research API
Public-video research and metrics
Restricted researcher eligibility
TikTok Display API
Authorized creator videos
User authorization required
TikTok Creative Center
Current hashtag and trend research
Do not assume a general-purpose API
Licensed data provider
Potential daily datasets and monitoring
Check access, cost and licensing
The official Research API provides fields including views, likes, shares, comments, hashtags and video duration. However, access is limited to qualifying research organisations in specified regions, not all developers. 

TikTok for Developers
+1
The Display API is intended for displaying a consenting creator's own TikTok content. It is not an unrestricted public-video search service. 

TikTok for Developers
+1
TikTok Creative Center is another useful research source for current hashtag trends and industry insights. Official Trends documentation.
For the Hugging Face publisher, the dataset card specifically states that commercial access, creator profiles and daily updates are available through Datasocial. That is worth investigating as a potential licensed route to more current data. 

Datasets at Hugging Face
+1
My recommendation is to implement a connector architecture supporting all these sources, with the Hugging Face dataset as the immediate starting point. Where live access is unavailable, the application must clearly display historical-data mode rather than pretend it has today's complete TikTok feed.
3. GitHub repositories worth studying
I found several real repositories that can provide useful architectural and modelling references. None should be represented as Maggie's original implementation.
Deep Neural Network viewer | dnnviewer
1. TikTok Virality Predictor
juanls1 · Multimodal deep learning
GitHub repository
Video, audio and text modelling. Useful for understanding multimodal feature engineering and prediction architecture. 

GitHub
+1
Recommended for research
Multimodal AI: Integrating Text, Image, and Video for Business Transformation
2. Virality Prediction
casparbreloh · PyTorch
GitHub repository
A cleaner multimodal modelling reference, worth reviewing for implementation structure.

GitHub
Recommended for architecture
GitHub - sanjeevnara7/DL-ViViT-Tensorflow: This repo contains a minimal Tensorflow implementation of the Video Vision Transformer (ViViT)
3. TikTok Virality Prediction
harbarex · ViViT / PyTorch
GitHub repository
A video-transformer approach for visual virality classification. Useful for a later multimodal extension, although its dataset is older. 

GitHub
Transforming Data Analysis with AI Agent: Meet Streamline Analyst | by Zhe LIN | Medium
4. TikTok Day 30 Views Prediction
NadaMIS · Ensemble modelling
GitHub repository
Predicts 30-day views using the first five days of engagement. This is useful for studying early-traction modelling, but it is not a pre-publication virality model. 

Bootcamp Kaggle · GitHub
I would not simply fork one of these and replace the CSV file. Build a new architecture with a reproducible data pipeline, stronger temporal validation and a functional inference service.
4. Project ideation: What should ViralIQ actually do?
I would aim for something more sophisticated than a binary classifier.
AI Content Analytics Dashboard by NurPix on Dribbble
A. Viral Prediction Studio
Submit a TikTok video URL or metadata and receive a predicted virality probability, estimated performance percentile, and supporting explanations.
How To Find Trending TikTok Sounds: 8 Ways (2026) - Shopify Australia
B. Trend Intelligence Radar
Identify accelerating hashtags, sounds, topics and content categories. Compare current trends across countries and languages.
Push the limits of explainability — an ultimate guide to SHAP library | by Ula La Paris | The Startup | Medium
C. Explainable Virality Analytics
Use SHAP and feature attribution to identify why the model considers a video promising or unlikely to perform.
Monitor ML Models | Deepchecks
D. Model Performance Laboratory
Compare CatBoost against LightGBM, logistic regression and other approaches, including historical versus recent training.
Social Media Reporting Guide for Marketing Analysts (2026)
E. Automated Insight Reports
Produce weekly reports explaining changing video characteristics, prediction performance, trend movements and model limitations.
The modelling strategy
I would build the following experiments:
Experiment
Model
Purpose
E1
Logistic Regression
Statistical baseline
E2
Random Forest
Nonlinear baseline
E3
CatBoost
Primary tabular model
E4
LightGBM
Performance comparison
E5
CatBoost + text embeddings
Enhanced content understanding
E6
Multimodal model
Optional video, audio and text fusion
Don't assume CatBoost will always win. Let the evaluation determine the champion.
How should accuracy be measured?
This is particularly important for viral prediction because viral videos are relatively rare.
I would prioritise:
Precision@K: How many of the highest-ranked predictions actually become viral?
Recall@K: How many genuinely viral videos did the model identify?
PR-AUC: How effectively does it distinguish rare positive examples?
Lift@K: How much better does it perform than random selection?
ROC-AUC: Secondary overall discrimination metric.
Brier Score and calibration: Are its predicted probabilities trustworthy?
For example, if Maggie's 6.9% represents Precision@K and the actual prevalence is 2%, then the improvement is 3.45 times the baseline. However, the two figures must come from the same evaluation population, prediction horizon and selection method. The original result should be independently reproduced before being quoted as established performance.
My preferred target definition: predict whether a video will enter the top 1% of views among comparable videos within seven days of publication. Segment by region, format, publication period and relevant age window. Don't hard-code a universal 250,000 or 400,000 view threshold.
For the existing Hugging Face dataset, use a separate historical snapshot classification task until reliable fixed-age outcome labels can be established.
5. Your complete AI Data Copilot agent prompt
The following is the prompt I would use with the six-member team. It gives every agent responsibilities, defines the ML requirements, and requires an actual functioning application rather than a collection of notebooks.
Ready to copy Six-agent collaboration
ViralIQ — TikTok Virality Intelligence & Prediction Platform
Project Mission
Build a professional, production-oriented TikTok Virality Intelligence Platform that combines large-scale data engineering, machine learning, trend intelligence, predictive analytics, explainable AI and interactive dashboard visualisation.
The objective is to develop a statistically rigorous machine learning system capable of identifying and predicting TikTok video virality using historical and continuously refreshed information.
The project is inspired by Maggie's reported CatBoost experiment using a 5.6-billion-record TikTok dataset, with a reported 6.9% hit rate compared with a 2.0% baseline without engagement metrics.
Treat those reported figures as research hypotheses to investigate and reproduce, not guaranteed results.
The final product must go substantially beyond replicating that experiment.
Build a complete platform that can:
Discover patterns associated with TikTok virality.
Predict the probability of a video becoming viral.
Estimate video performance and ranking within its peer group.
Identify emerging trends and accelerating sounds and hashtags.
Incorporate recent 2025–2026 TikTok data.
Monitor data freshness and changing platform behaviour.
Compare different machine learning models.
Generate explainable predictions and research-quality reports.
Provide actual predictions through a functioning backend API.
Automatically evaluate whether model performance deteriorates over time.
1. PROJECT DIRECTORY AND FILE MANAGEMENT
Create the entire project inside:
/Users/derr/Documents/Random/ViralIQ
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
2. TEAM STRUCTURE AND RESPONSIBILITIES
Operate as a coordinated six-member AI-native data science and engineering team.
NOVA — Chief Data Orchestrator / Project Lead
Primary responsibility: Overall project execution and coordination.
Responsibilities:
Establish the project roadmap.
Divide tasks between the other five agents.
Establish technical architecture and dependencies.
Coordinate data engineering, ML development and frontend implementation.
Review outputs and resolve conflicting recommendations.
Ensure all components integrate correctly.
Monitor development progress.
Challenge unsupported assumptions.
Ensure the final application is genuinely functional.
Nova must not simply delegate tasks and report completion.
Nova must verify that the individual components work together.
SAGE — Data Scientist / Machine Learning Engineer
Primary responsibility: Predictive modelling and statistical validity.
Responsibilities:
Develop the complete ML pipeline.
Investigate Maggie's reported CatBoost methodology.
Establish baseline performance.
Develop CatBoost as the primary candidate model.
Benchmark against Logistic Regression, Random Forest and LightGBM.
Investigate content embeddings.
Engineer point-in-time features.
Perform hyperparameter optimisation.
Evaluate class imbalance.
Calibrate predictive probabilities.
Implement SHAP explainability.
Perform temporal validation.
Investigate performance differences across 2024, 2025 and 2026.
Save trained models and reusable preprocessing artifacts.
Sage must distinguish genuine predictive capability from correlations caused by data leakage.
QUINN — SQL Analyst / Data Engineering Specialist
Primary responsibility: Large-scale dataset processing and data infrastructure.
Responsibilities:
Integrate the Hugging Face dataset.
Build efficient Parquet ingestion.
Use DuckDB and Polars for analytical processing.
Establish PostgreSQL tables where appropriate.
Implement feature extraction and data transformations.
Establish data quality checks.
Build repeatable ETL pipelines.
Handle incremental data ingestion.
Develop SQL queries for trend and engagement analysis.
Maintain a reproducible data dictionary.
Avoid loading the entire dataset into memory.
RIKA — Knowledge Retrieval / Research Specialist
Primary responsibility: Dataset research, source verification and methodological investigation.
Responsibilities:
Verify all relevant datasets.
Investigate alternative TikTok datasets.
Review academic literature about short-form video virality.
Investigate TikTok trend dynamics.
Research platform and API limitations.
Verify licensing and permitted data acquisition methods.
Investigate CatBoost, multimodal prediction and modern NLP approaches.
Document research findings with source references.
Identify changes in TikTok content and engagement behaviour.
Review potential data leakage and survivorship bias.
Do not invent API access, datasets, papers or research results.
DASH — Visual Analytics Designer / Frontend Engineer
Primary responsibility: Interactive analytics dashboard.
Responsibilities:
Build a polished, professional dashboard.
Establish consistent design language.
Connect the frontend to actual backend endpoints.
Implement prediction visualisations.
Implement trend monitoring.
Display model comparisons.
Display model explanations.
Implement filters for time, region, language and content type.
Support data freshness and model status indicators.
Provide meaningful loading, error and empty states.
Do not build a static dashboard populated with fabricated metrics.
Every important interactive feature must communicate with a real backend endpoint.
REX — Research Report Composer / Analytics Specialist
Primary responsibility: Reporting, insights and communication.
Responsibilities:
Produce executive summaries.
Document modelling methodology.
Explain experimental results.
Generate feature importance reports.
Identify significant temporal changes.
Document model limitations.
Produce model performance comparisons.
Generate weekly trend reports.
Produce reproducible analytical reports with charts and tables.
All findings must be supported by actual analytical results.
3. DATASET ACQUISITION AND VERIFICATION
Primary dataset
Hugging Face:
https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos
Explore:
https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos/tree/main/data
This dataset contains billions of TikTok video records in monthly Parquet partitions.
Available fields include:
video_id
posted_at
country
language
type
duration_ms
caption
hashtags
hashtag_ids
mentions
on_screen_text
sound_id
is_ad
branded_content
product_id
seller_id
is_shop_video
is_ai_generated
not_recommended_for_fyp
views
likes
comments
shares
saves
downloads
engagement_rate
stats_updated_at
Dataset research requirements
Before modelling, Rika and Quinn must:
Verify the current repository and schema.
Inspect available monthly Parquet partitions.
Determine the actual latest publication date.
Determine the latest metric observation date.
Calculate dataset freshness and observation delays.
Investigate missing values and inconsistent records.
Identify historical archive limitations.
Verify licensing and permitted usage.
Confirm the actual availability of recent 2025 and 2026 records.
Produce a documented dataset audit.
Never assume the dataset is continuously updated just because its repository was recently updated.
Use DuckDB and Polars to inspect and process selected partitions without downloading the entire dataset.
Prioritise 2024–2026 records for current modelling, while investigating whether older historical records add predictive value.
Additional data sources
Investigate the following:
TikTok Research API:
https://developers.tiktok.com/products/research-api/
TikTok Display API:
https://developers.tiktok.com/products/display-api/
TikTok Creative Center:
https://ads.tiktok.com/business/creativecenter/
Potential licensed daily data sources:
https://datasocial.ai/
Build independent source adapters so the application can support multiple sources.
Research API access must only be used if the project satisfies TikTok's eligibility and approval requirements.
The Display API must use the appropriate creator authorization.
Never bypass API restrictions, authentication, anti-bot measures or access controls. Do not assume that a third-party scraping service is authorised to provide any particular data.
Provide Parquet and CSV import functionality as a fallback.
Critical data freshness functionality
Implement a data freshness monitoring service.
It must report:
Latest available publication timestamp.
Latest recorded engagement observation.
Last successful ingestion.
Number of new records.
Source availability.
Data collection delay.
Number of missing or invalid records.
Dataset schema changes.
Last successful model retraining.
Current model version.
The dashboard must distinguish between live, recently refreshed and historical datasets.
If live data is unavailable, clearly identify the application as operating in historical-data mode.
Never represent historical measurements as live TikTok statistics.
4. TARGET DEFINITION AND VIRALITY METHODOLOGY
Virality must be defined statistically rather than using an arbitrary universal view threshold.
Develop at least two separate prediction tasks.
Task A: Pre-publication virality prediction
Objective:
Predict whether a newly published video will enter the top 1% of its comparable video cohort within seven days.
Permitted features must be available before publication.
Examples:
Caption characteristics.
Caption embeddings.
Hashtags.
Language.
Duration.
Publication time.
Historical sound popularity.
Creator's prior publishing history.
Historical performance derived only from earlier videos.
Do not use subsequent views, likes, shares or comments.
Task B: Early-traction virality prediction
Objective:
Predict whether a video will become viral using information available shortly after publication.
Potential observation windows:
30 minutes.
1 hour.
6 hours.
24 hours.
Possible features:
Initial views.
View growth velocity.
Like rate.
Share rate.
Comment rate.
Early engagement growth.
Sound and hashtag momentum.
These features are permitted only when the prediction is made after they have been observed.
Label engineering
Create separate fields for:
viral_at_snapshot
viral_24h
viral_7d
future_view_count
performance_percentile
observation_age_hours
The existing Hugging Face dataset has engagement statistics associated with stats_updated_at. Do not assume that it contains repeated observations at 24 hours or seven days.
Build genuine fixed-age targets only when suitable observations are available.
For the retrospective dataset, use a clearly named snapshot-based proxy label where fixed-age outcomes cannot be established.
Never silently mix the two tasks or present a retrospective classification result as prospective forecasting accuracy.
5. FEATURE ENGINEERING
Investigate the predictive contribution of the following feature groups.
Content features
Caption length.
Caption complexity.
Caption sentiment.
Caption language.
Caption embeddings.
Hashtag count.
Hashtag combinations.
Topic classifications.
On-screen text.
Text originality and similarity.
Publishing features
Hour of publication.
Day of week.
Weekend versus weekday.
Country.
Language.
Historical posting patterns.
Sound features
Sound identifier.
Historical usage volume.
Recent usage acceleration.
Sound popularity by country.
Sound popularity by language.
Sound adoption velocity.
Compute sound popularity using only videos published before the prediction timestamp.
Do not include future sound usage.
Creator features
Where legally and technically available:
Historical video count.
Historical median views.
Previous viral-video frequency.
Publishing consistency.
Historical average engagement.
Creator features must be calculated using earlier information only.
The primary Hugging Face dataset does not directly provide historical follower counts. Do not fabricate them.
Optional multimodal extension
Investigate:
Video frame embeddings.
CLIP or comparable visual encoders.
Audio representations.
Video scene characteristics.
Visual content categorisation.
This is a separate extension because the primary dataset contains metadata, not the underlying video and audio files.
Only implement this when there is an authorised media source.
6. MODEL DEVELOPMENT AND EXPERIMENT TRACKING
Implement reproducible experiments.
Baselines
Majority-class classifier.
Random-ranking baseline.
Logistic Regression.
Random Forest.
Main candidates
CatBoostClassifier.
LightGBM.
XGBoost, where appropriate.
Sentence-transformer-enhanced CatBoost.
Investigate the suitability of multilingual embedding models for videos across different regions.
Use Optuna or another appropriate optimisation framework if the dataset size and available compute justify it.
Experiment design
Create separate experiments for:
Historical data only.
2025–2026 data.
2026-focused data.
Metadata-only prediction.
Metadata plus content embeddings.
Pre-publication prediction.
Early-traction prediction.
Historical versus recent sound features.
Do not automatically give older years equal weight.
Test whether historical data improves or damages performance on the latest holdout period.
Use chronological train, validation and test splits.
Also examine creator-grouped evaluation to test performance on videos from creators not seen during training.
Never rely exclusively on random train/test splitting.
Evaluation metrics
Report:
Precision@K.
Recall@K.
PR-AUC.
ROC-AUC.
Lift@K.
F1-score.
Brier score.
Log loss.
Calibration.
Confusion matrix.
Accuracy must never be the only reported performance metric.
Include bootstrap confidence intervals where appropriate.
Record the positive-class prevalence and the precise definition of every metric.
Model artefacts
Save:
Trained model.
Feature definitions.
Preprocessing pipeline.
Training configuration.
Model version.
Evaluation results.
Training data date range.
Source dataset version.
Model metadata.
Use MLflow or an equivalent experiment-tracking system where appropriate.
Do not claim CatBoost supports an arbitrary warm-start option. Use supported incremental initialisation only when justified; otherwise retrain on the selected rolling training window.
7. EXPLAINABLE AI AND ANALYTICAL INSIGHTS
Integrate SHAP to explain model decisions.
For each prediction, provide:
Predicted probability.
Predicted performance percentile.
Expected view range, if properly calibrated.
Main positive feature contributions.
Main negative feature contributions.
Comparison with similar historical videos.
Model version.
Data freshness information.
Build analytical investigations around:
Content characteristics
Which caption, topic, hashtag and video duration characteristics are associated with higher performance?
Publishing behaviour
Which publication periods perform better within different markets?
Sound trends
Which sounds are showing acceleration rather than simply having high historical usage?
Creator behaviour
Does an established creator's previous performance predict outcomes? How well does the model generalise to new creators?
Regional differences
Do the characteristics associated with successful videos differ by country and language?
Temporal changes
Are 2026 prediction patterns different from those in 2024 and 2025?
Distinguish correlation from causation.
Do not describe a feature as causing virality merely because SHAP ranks it highly.
8. REAL BACKEND AND APPLICATION ARCHITECTURE
Build a working end-to-end application.
Suggested technology:
Python for data science.
FastAPI for inference and data services.
CatBoost and LightGBM for predictive models.
DuckDB and Polars for analytical processing.
PostgreSQL for application data and observations.
Redis/Celery or an appropriate scheduler for background processing.
React/Next.js and TypeScript for the dashboard.
Plotly or an equivalent library for analytics.
Docker for reproducible development and deployment.
Adapt this architecture when there is a demonstrably better approach.
Required backend functionality
Implement real endpoints, including:
POST /api/v1/predictions/prepublish
POST /api/v1/predictions/early-traction
GET /api/v1/predictions/history
GET /api/v1/trends
GET /api/v1/analytics/overview
GET /api/v1/models
GET /api/v1/models/evaluation
GET /api/v1/data/status
POST /api/v1/data/import
GET /api/v1/health
Frontend controls must call these backend services.
Persist prediction history, observations, model versions and ingestion records.
The dashboard must not rely on hard-coded sample metrics.
9. DASHBOARD DESIGN
Build a professional, modern analytical dashboard.
Use a clear visual hierarchy, consistent typography, useful filters and responsive layouts.
Main navigation
1. Overview
Display:
Total dataset records.
Latest data observation.
Number of videos analysed.
Viral-video prevalence.
Current model performance.
Latest model version.
Recent trends.
Data freshness.
2. Virality Prediction Studio
Allow users to:
Enter video metadata.
Submit an authorised video URL for supported ingestion.
Select pre-publication or early-traction prediction.
Enter available features.
Request a genuine prediction.
Review prediction explanations.
Compare against similar historical videos.
3. Trend Intelligence
Display:
Trending sounds.
Accelerating hashtags.
Content categories.
Geographic trend differences.
Language differences.
Trend velocity.
Historical trend comparisons.
Distinguish trending observations from model-generated estimates.
4. Content Performance Analytics
Display:
Video performance distribution.
Duration versus performance.
Publication-time analysis.
Caption and hashtag analysis.
Creator historical performance.
Content format comparisons.
5. Model Laboratory
Display:
Model comparisons.
Precision@K.
Recall@K.
PR-AUC.
Lift@K.
Calibration.
Historical versus recent performance.
Feature importance.
Experiment history.
6. Data Operations
Display:
Data sources.
Last successful ingestion.
Latest publication timestamp.
Last metric observation.
Missing-data indicators.
Data quality checks.
Pipeline status.
7. Research and Reports
Enable report generation containing:
Executive summary.
Methodology.
Data sources.
Model results.
Statistical interpretation.
Significant insights.
Limitations.
Recommendations.
10. AUTOMATED DATA REFRESH AND MODEL MONITORING
Create a configurable ingestion and monitoring pipeline.
When a compliant data connector is configured:
Detect new data.
Validate incoming schemas.
Deduplicate video IDs.
Preserve historical observations.
Update sound and hashtag trends.
Recalculate time-dependent features.
Track data distribution changes.
Schedule model evaluation.
Retrain only when sufficient new labelled data exists.
Establish configurable daily data collection and weekly or monthly model evaluation/retraining workflows.
The platform must also support manual ingestion and historical reproducibility.
Monitor:
Data drift.
Feature drift.
Prediction distribution changes.
Precision@K deterioration.
Calibration deterioration.
Missing-data rates.
Source failures.
Never automatically replace a production model simply because a newer model has been trained.
A candidate model must pass validation before promotion.
11. REPORTING AND DELIVERABLES
Rex must produce a professional research and results package.
Required deliverables:
Project architecture report.
Dataset audit.
Data dictionary.
Exploratory data analysis.
Feature engineering documentation.
Model methodology.
Model benchmarking report.
Model evaluation report.
SHAP interpretation.
Current trend analysis.
Model card.
Data limitations report.
Executive summary.
Reproducibility instructions.
Every numeric result must be obtained from an actual execution.
Reports must state:
Sample size.
Dataset date range.
Training and test periods.
Target definition.
Evaluation methodology.
Model settings.
Actual performance.
Confidence intervals where applicable.
Limitations.
Include charts and tables that communicate substantive findings, rather than decorative graphics.
12. QUALITY ASSURANCE AND FUNCTIONAL ACCEPTANCE
Quinn, Sage and Dash must coordinate an end-to-end validation.
Test the following:
Dataset ingestion works.
Data schema validation works.
Feature engineering produces the expected fields.
Training completes successfully.
Model artifacts are saved.
Models can be loaded independently.
Inference endpoints return real predictions.
Predictions are persisted correctly.
Dashboard requests reach the backend.
Filters change the underlying query results.
Model comparison values come from actual experiments.
Data freshness indicators reflect actual ingestion metadata.
Error handling works.
Missing-data conditions are handled correctly.
Tests pass in a clean development environment.
Do not mark a feature complete simply because its frontend has been implemented.
Provide a clear distinction between:
Implemented and tested.
Implemented but not fully validated.
Blocked by external access.
Planned but not implemented.
API credentials must never be required for the historical-data MVP. Provide an .env.example for optional integrations without including actual secrets.
13. EXECUTION ROADMAP
Follow this order.
Phase 1 — Research and data audit
Rika and Quinn verify the dataset, schema, licensing, data coverage, source freshness and processing feasibility.
Deliverable: Dataset audit and usable sample.
Phase 2 — Exploratory analysis and labelling
Sage and Quinn establish target definitions, data cleaning, feature engineering and statistical analysis.
Deliverable: Reproducible training dataset.
Phase 3 — Baseline and model development
Sage develops the model experiments and evaluation pipeline.
Deliverable: Benchmarked models and documented evaluation results.
Phase 4 — Trend intelligence and freshness infrastructure
Rika and Quinn research compliant data sources and implement the ingestion and monitoring architecture.
Deliverable: Historical and current-data connectors with explicit freshness status.
Phase 5 — API and dashboard
Dash, Quinn and Sage integrate inference, data storage and interactive analytics.
Deliverable: Functional Virality Intelligence Platform.
Phase 6 — Validation and reporting
Nova coordinates end-to-end testing. Rex produces the final analytical reports.
Deliverable: Tested, documented and reproducible project.
14. FINAL EXPECTATIONS
This project must be treated as a serious data science, machine learning and software engineering product.
Prioritise statistical accuracy, reproducibility, maintainability and functionality over unnecessary complexity.
Do not:
Invent performance metrics.
Fabricate dataset coverage.
Use future engagement as a pre-publication feature.
Claim a model predicts virality when it merely classifies historical views.
Hide model limitations.
Present a static frontend as a working application.
Download enormous datasets without a clear reason.
Claim to have implemented live ingestion without a verified data source.
Make unsupported causal claims.
When something cannot be implemented because an external service requires approval, finish the available functionality and clearly document the limitation.
Final definition of success
ViralIQ should be able to take a TikTok video or its permitted metadata, process it through a real machine learning pipeline, return an interpretable prediction, display supporting analytical evidence, and measure its eventual prediction accuracy against independently collected outcomes.
The finished platform must demonstrate that its predictive performance is genuine, its data is traceable, its insights are explainable and its architecture can accommodate continuously changing TikTok trends.
Build the complete working product, not just a proof of concept or a collection of notebooks.
6. What I would prioritise during execution
I would avoid having all six agents independently build everything at once. Give them these deliverables in sequence:
Priority
Agent
Required milestone
1
Rika + Quinn
Dataset audit and 2026 data availability
2
Sage
Reproducible CatBoost baseline
3
Sage + Quinn
Leakage-free features and temporal testing
4
Quinn + Rika
Fresh-data ingestion and observation storage
5
Dash + Quinn
Functional prediction API and dashboard
6
Rex + Nova
Validation, findings and final reports
My recommended MVP acceptance criteria
What should actually work before expanding the project
Dataset audit and sample processing
Required
CatBoost baseline and comparison models
Required
Historical pre-publication prediction
Required
Real backend inference and dashboard
Required
Point-in-time observation collection
Required
Actual seven-day forecasting validation
Next milestone
Multimodal video and audio analysis
Advanced
My final recommendation: The most valuable contribution of this project will not simply be achieving a higher historical classification score than Maggie. It will be demonstrating that a model trained on previous TikTok content can successfully identify emerging viral videos in a genuinely later, unseen period.
That distinction makes the project considerably more credible as a data science portfolio project and potentially useful as a real analytics product.
I would also establish a monthly dataset freshness and model-drift review so that changes in source coverage, TikTok trends and actual prediction performance are independently checked.
