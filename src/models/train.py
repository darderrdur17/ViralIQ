"""Model training, temporal validation and experiment recording.

Protocol
--------
* Thresholds and encoders are fitted on the training window only.
* Validation is a *later* slice of the training month, used for probability
  calibration and early stopping.
* Test is the remainder of the following month.
* The current partial month is a separate, untouched "recent" holdout used to
  test whether performance survives a genuine forward time shift.

Random train/test splitting is deliberately absent: it would let the model learn
from videos published after the ones it is scored on.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.evaluation import metrics as M
from src.explainability.attribution import global_importance
from src.feature_engineering.features import build_prepublication_features, fit_sound_stats
from src.labeling import labels as L
from src.models.encoder import FeatureEncoder
from src.models.reference import build_reference_bundle, save_bundle

log = get_logger("train")

VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
# Columns the pre-publication pipeline actually needs. Projecting early keeps the
# working set small on memory-constrained machines.
TRAINING_COLUMNS: tuple[str, ...] = (
    "video_id",
    "posted_at",
    "country",
    "language",
    "type",
    "duration_ms",
    "image_count",
    "caption",
    "hashtags",
    "sound_id",
    "is_ad",
    "branded_content",
    "is_shop_video",
    "is_ai_generated",
    "not_recommended_for_fyp",
    "views",
    "stats_updated_at",
)


def load_frames(paths: list[Path], file_row_cap: int | None = None, seed: int = 42) -> pl.DataFrame:
    """Load sample files, projecting to the training columns.

    ``file_row_cap`` bounds the in-memory working set on machines where holding
    several million rows across concatenation and feature frames would exhaust RAM.
    Sampling is seeded, so a capped load is reproducible.
    """
    frames = []
    for p in paths:
        df = pl.read_parquet(p)
        keep = [c for c in TRAINING_COLUMNS if c in df.columns]
        df = df.select(keep)
        if file_row_cap and df.height > file_row_cap:
            df = df.sample(n=file_row_cap, seed=seed)
        frames.append(df)
        log.info("loaded %s: %d rows x %d cols", p.name, df.height, df.width)
        del df
    return pl.concat(frames, how="vertical_relaxed")


def temporal_slices(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Chronological split by publication timestamp."""
    df = df.sort("posted_at")
    month_expr = pl.col("posted_at").dt.strftime("%Y-%m")
    months = (
        df.select(month_expr.alias("_month")).unique().sort("_month")["_month"].to_list()
    )

    if len(months) >= 3:
        # train = M, val/test = halves of M+1, recent = M+2. Anything later than M+2
        # is deliberately unused: a forward holdout must be the *next* unseen month,
        # not whichever month happens to be last.
        train_m, val_m, recent_m = months[0], months[1], months[2]
        train = df.filter(pl.col("posted_at").dt.strftime("%Y-%m") == train_m)
        rest = df.filter(pl.col("posted_at").dt.strftime("%Y-%m") == val_m)
        cutoff = rest["posted_at"].quantile(0.5)
        val = rest.filter(pl.col("posted_at") <= cutoff)
        test = rest.filter(pl.col("posted_at") > cutoff)
        recent = df.filter(pl.col("posted_at").dt.strftime("%Y-%m") == recent_m)
    elif len(months) == 2:
        train = df.filter(pl.col("posted_at").dt.strftime("%Y-%m") == months[0])
        rest = df.filter(pl.col("posted_at").dt.strftime("%Y-%m") == months[1])
        cutoff = rest["posted_at"].quantile(0.5)
        val = rest.filter(pl.col("posted_at") <= cutoff)
        test = rest.filter(pl.col("posted_at") > cutoff)
        recent = pl.DataFrame()
    else:
        # Single month: hold out the last 30% and split it chronologically so that
        # calibration and test never share a row.
        cutoff = df["posted_at"].quantile(0.7)
        train = df.filter(pl.col("posted_at") <= cutoff)
        rest = df.filter(pl.col("posted_at") > cutoff)
        mid = rest["posted_at"].quantile(0.5)
        val = rest.filter(pl.col("posted_at") <= mid)
        test = rest.filter(pl.col("posted_at") > mid)
        recent = pl.DataFrame()

    out = {"train": train, "val": val, "test": test, "recent": recent}
    for name, part in out.items():
        if part.height:
            log.info(
                "split %-6s rows=%9d  %s .. %s",
                name,
                part.height,
                part["posted_at"].min(),
                part["posted_at"].max(),
            )
    return out


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def build_estimators(seed: int, cat_features: list[str]) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = list(FeatureEncoder().numeric)
    categorical = list(FeatureEncoder().categorical)

    def onehot_pipeline(scale: bool) -> Pipeline:
        num_steps = [("scale", StandardScaler())] if scale else []
        ct = ColumnTransformer(
            [
                ("num", Pipeline(num_steps) if num_steps else "passthrough", numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), categorical),
            ],
            remainder="drop",
        )
        return ct

    return {
        "E1_logistic_regression": Pipeline(
            [
                ("prep", onehot_pipeline(scale=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "E2_random_forest": Pipeline(
            [
                ("prep", onehot_pipeline(scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=14,
                        min_samples_leaf=30,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "E3_catboost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            thread_count=3,
            cat_features=cat_features,
            # All categorical vocabularies are small (<=25 values), so one-hot is
            # both sufficient and far cheaper than CatBoost's default target-statistic
            # feature combinations, which are the main memory risk here.
            one_hot_max_size=32,
            max_ctr_complexity=1,
            allow_writing_files=False,
        ),
        "E4_lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
    }


def calibrate(model, X_val, y_val, method: str = "sigmoid"):
    """Wrap a fitted model in probability calibration on the validation slice.

    Platt scaling (``sigmoid``) is used rather than isotonic regression. Isotonic
    fits a step function, which for a ~1% base rate collapses wide ranges of the
    score into one plateau — distinct videos then receive *identical* probabilities,
    destroying the ranking resolution the product depends on. Platt scaling is
    strictly monotonic, so ordering is preserved exactly.

    scikit-learn 1.6+ replaced ``cv="prefit"`` with an explicit ``FrozenEstimator``
    wrapper, which states the intent more honestly: the base model is not refitted
    during calibration. Older versions still accept ``cv="prefit"``.
    """
    from sklearn.calibration import CalibratedClassifierCV

    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:  # pragma: no cover - older scikit-learn
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    try:
        cal.fit(X_val, y_val)
        return cal, True
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("calibration failed (%s); using raw scores", exc)
        return model, False


def _predict_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainingRun:
    run_id: str
    started_at: str
    sample_files: list[str]
    metrics: dict[str, dict]
    label_diagnostics: dict[str, Any]
    thresholds: dict[str, Any]
    splits: dict[str, dict]
    feature_importance: dict[str, list[dict]]
    artifacts: dict[str, str]
    duration_seconds: float
    environment: dict[str, str]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def run_training(
    sample_paths: list[Path] | None = None,
    output_tag: str = "primary",
    seed: int | None = None,
    max_rows_per_split: int | None = None,
    file_row_cap: int | None = 400_000,
) -> TrainingRun:
    started = time.time()
    seed = seed if seed is not None else SETTINGS.model.random_state
    sample_paths = sample_paths or sorted(SETTINGS.paths.data_samples.glob("*.parquet"))
    sample_paths = [p for p in sample_paths if not p.name.startswith(".")]
    if not sample_paths:
        raise FileNotFoundError("No sample parquet files found in data/samples")

    raw = load_frames(sample_paths, file_row_cap=file_row_cap, seed=seed)
    raw = L.add_observation_clock(raw)

    splits = temporal_slices(raw)
    if max_rows_per_split:
        splits = {
            k: (v.sample(n=min(max_rows_per_split, v.height), seed=seed) if v.height else v)
            for k, v in splits.items()
        }
    # The full frame is no longer needed once splits are materialised.
    del raw
    gc.collect()

    train = splits["train"]
    months = tuple(
        sorted(
            train.select(pl.col("posted_at").dt.strftime("%Y-%m").alias("_month"))["_month"]
            .unique()
            .to_list()
        )
    )
    thresholds = L.fit_cohort_thresholds(train, months=months)

    labeled = {k: L.apply_labels(v, thresholds) for k, v in splits.items() if v.height}
    diag = {k: L.label_diagnostics(v) for k, v in labeled.items()}
    log.info("label diagnostics (train): %s", json.dumps(diag["train"]))

    # Features — sound stats frozen on the training window only.
    sound_stats = fit_sound_stats(train)
    featured = {
        k: build_prepublication_features(v, sound_stats) for k, v in labeled.items()
    }

    encoder = FeatureEncoder().fit(featured["train"])
    X = {k: encoder.transform(v) for k, v in featured.items()}
    y = {k: v["viral_at_snapshot"].cast(pl.Int8).to_numpy() for k, v in labeled.items()}

    train_mask = labeled["train"]["observation_valid"].to_numpy()
    X_train, y_train = X["train"][train_mask], y["train"][train_mask]
    val_mask = labeled["val"]["observation_valid"].to_numpy()
    X_val, y_val = X["val"][val_mask], y["val"][val_mask]

    log.info(
        "training rows=%d (pos=%d, %.3f%%)  val rows=%d (pos=%d)",
        len(X_train), y_train.sum(), 100 * y_train.mean(), len(X_val), y_val.sum(),
    )

    cat_cols = list(encoder.categorical)
    estimators = build_estimators(seed, cat_cols)

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{output_tag}_{uuid.uuid4().hex[:6]}"
    run_dir = SETTINGS.paths.model_registry / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(run_dir / "encoder.json")
    sound_stats.write_parquet(run_dir / "sound_stats.parquet")
    # A background sample of the feature frame. SHAP explainers need a reference
    # distribution; explaining a row against itself yields all-zero attributions.
    featured["train"].select(encoder.feature_names).head(500).write_parquet(
        run_dir / "reference_features.parquet"
    )
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds.as_dict(), indent=2), encoding="utf-8")

    k_values = SETTINGS.model.top_k_values
    results: dict[str, dict] = {}
    importance: dict[str, list[dict]] = {}
    artifacts: dict[str, str] = {}

    # ---- baselines --------------------------------------------------------
    # Baselines must be scored on exactly the same rows as the models, otherwise
    # the comparison silently mixes two different evaluation populations.
    valid_masks = {
        split: labeled[split]["observation_valid"].to_numpy()
        for split in ("test", "recent")
        if split in labeled
    }
    results["E0_random_ranking"] = {
        "test": M.random_ranking_baseline(
            y["test"][valid_masks["test"]], k_values, seed=seed
        ),
        "recent": (
            M.random_ranking_baseline(
                y["recent"][valid_masks["recent"]], k_values, seed=seed
            )
            if "recent" in y and len(y["recent"])
            else None
        ),
        "note": "Uniform random scores on the identical evaluation rows; the floor any model must clear.",
    }
    prev = float(y["train"].mean())
    y_test_valid = y["test"][valid_masks["test"]]
    results["E0_majority_prevalence"] = {
        "test": {
            "n": int(len(y_test_valid)),
            "prevalence": round(float(y_test_valid.mean()), 6),
            "note": "Constant predictor at train prevalence; PR-AUC degenerates to prevalence.",
        },
        "train_prevalence": round(prev, 6),
    }

    # ---- supervised models ------------------------------------------------
    for name, est in estimators.items():
        t0 = time.time()
        log.info("fitting %s on %d rows", name, len(X_train))
        try:
            if name == "E3_catboost":
                est.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                )
            elif name == "E4_lightgbm":
                est.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="average_precision",
                )
            else:
                est.fit(X_train, y_train)
        except Exception as exc:
            log.error("%s failed: %s", name, exc)
            results[name] = {"error": str(exc)}
            continue

        model, calibrated = calibrate(est, X_val, y_val)
        model_path = run_dir / f"{name}.bin"
        import joblib

        joblib.dump(model, model_path)
        artifacts[name] = str(model_path)

        # Reference distribution for percentile + expected view range reporting.
        try:
            val_scores = _predict_scores(model, X_val)
            val_views = labeled["val"].filter(pl.Series(val_mask))["views"].to_numpy()
            bundle = build_reference_bundle(val_scores, val_views)
            save_bundle(bundle, run_dir / "reference" / f"{name}.json")
        except Exception as exc:
            log.warning("reference bundle failed for %s: %s", name, exc)

        entry: dict[str, Any] = {
            "calibrated": calibrated,
            "fit_seconds": round(time.time() - t0, 1),
        }
        for split in ("test", "recent"):
            if split in X and len(X[split]):
                mask = labeled[split]["observation_valid"].to_numpy()
                scores = _predict_scores(model, X[split][mask])
                entry[split] = M.evaluate(y[split][mask], scores, k_values)

                # Clustered CI on PR-AUC, clustering by sound (correlated rows).
                clusters = (
                    labeled[split].filter(pl.Series(mask))["sound_id"].fill_null(0).to_numpy()
                )
                entry[f"{split}_pr_auc_ci"] = M.cluster_bootstrap_ci(
                    y[split][mask],
                    scores,
                    clusters,
                    metric_fn=average_precision_score,
                    iterations=SETTINGS.model.bootstrap_iterations,
                    seed=seed,
                )
        results[name] = entry

        # Feature importance — every model family, so the champion never shows an
        # empty panel just because it happens to be linear rather than a tree.
        try:
            ranked = global_importance(model, top_n=20)
            if ranked:
                importance[name] = ranked
            else:
                log.info("no global importance available for %s", name)
        except Exception as exc:
            log.warning("importance failed for %s: %s", name, exc)

        log.info("%s done in %.1fs", name, time.time() - t0)
        # Free the fitted estimator before the next model. Holding four ensembles
        # resident at once is the dominant peak-memory cost of the whole run.
        del model, est
        gc.collect()

    # ---- persist ----------------------------------------------------------
    split_meta = {
        k: {
            "rows": v.height,
            "min_posted_at": str(v["posted_at"].min()) if v.height else None,
            "max_posted_at": str(v["posted_at"].max()) if v.height else None,
        }
        for k, v in splits.items()
    }

    run = TrainingRun(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        sample_files=[p.name for p in sample_paths],
        metrics=results,
        label_diagnostics=diag,
        thresholds=thresholds.as_dict(),
        splits=split_meta,
        feature_importance=importance,
        artifacts=artifacts,
        duration_seconds=round(time.time() - started, 1),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "viraliq_version": VERSION,
        },
    )
    (run_dir / "run.json").write_text(json.dumps(run.as_dict(), indent=2, default=str), encoding="utf-8")

    champion = select_champion(results)
    registry_path = SETTINGS.paths.model_registry / "index.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"runs": []}
    registry["runs"].append(
        {
            "run_id": run_id,
            "started_at": run.started_at,
            "champion": champion,
            "duration_seconds": run.duration_seconds,
            "path": str(run_dir),
        }
    )
    registry["active_run"] = run_id
    registry["active_model"] = champion
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    log.info("run %s complete in %.1fs; champion=%s", run_id, run.duration_seconds, champion)
    return run


def select_champion(results: dict[str, dict]) -> str | None:
    """Pick the best model by test PR-AUC, then Precision@K."""
    scored = []
    for name, entry in results.items():
        test = entry.get("test") or {}
        pr_auc = test.get("pr_auc")
        if pr_auc is None:
            continue
        p_at = max(
            (v.get("precision_at_k", 0) or 0 for v in (test.get("ranking") or {}).values()),
            default=0,
        )
        scored.append((pr_auc, p_at, name))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]
