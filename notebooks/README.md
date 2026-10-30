# Notebooks

The reproducible pipeline lives in `scripts/` and `src/`, not here. This is deliberate: a
notebook cannot be tested, diffed, or scheduled, and the blueprint explicitly asks for a working
product rather than a collection of notebooks.

Use these commands instead of the notebook sequence suggested in the original spec:

| Original notebook | Reproducible equivalent |
|---|---|
| `01_dataset_audit` | `python scripts/audit.py --tail 2` → `reports/research/dataset_audit.md` |
| `02_exploratory_analysis` | `GET /api/v1/analytics/overview` and `/api/v1/analytics/content` |
| `03_feature_engineering` | `src/feature_engineering/features.py` (unit-tested) |
| `04_baseline_models` | `python scripts/train.py` → `reports/evaluation/model_benchmark.md` |
| `05_catboost_experiment` | same run; CatBoost is experiment `E3` |
| `06_model_comparison` | `GET /api/v1/models/evaluation` |

If you want an interactive session, open one against the real modules:

```python
from src.ingestion.hf_source import HFDatasetSource
from src.ingestion import duckdb_engine as db
from src.preprocessing import quality

src = HFDatasetSource()
part = src.partitions_for(months=["2026-10"])[0]
url = src.config.resolve_url(part.path)

with db.session() as con:
    print(quality.audit_relation(con, f"read_parquet('{url}')", part.month))
```

Anything worth keeping from such a session should be promoted into `src/` with a test.
