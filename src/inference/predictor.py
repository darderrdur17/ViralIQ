"""Inference service.

Loads the active model from the registry and produces a prediction object that
states what it knows, what it does not know, and how confident it is entitled to
be. Every field is traceable to a stored artefact.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from src.common.config import SETTINGS
from src.common.logging import get_logger
from src.explainability.attribution import local_attributions
from src.feature_engineering.features import build_prepublication_features
from src.models.encoder import FeatureEncoder
from src.models.reference import load_bundle, percentile_of_views, view_range_for

log = get_logger("predictor")

TIER_BANDS = (
    (95.0, "very_high", "Top 5% of scored videos — strong virality candidate"),
    (80.0, "high", "Above the peer-group norm"),
    (50.0, "moderate", "Around the peer-group median"),
    (20.0, "low", "Below the peer-group norm"),
    (0.0, "very_low", "Unlikely to reach peer-group top 1%"),
)


class ModelNotAvailable(RuntimeError):
    """Raised when no trained model can serve a request."""


class Predictor:
    def __init__(self, run_id: str | None = None, model_name: str | None = None) -> None:
        self.registry_path = SETTINGS.paths.model_registry / "index.json"
        if not self.registry_path.exists():
            raise ModelNotAvailable(
                "No model registry found. Run `python scripts/train.py` to train the first model."
            )
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.run_id = run_id or registry.get("active_run")
        self.model_name = model_name or registry.get("active_model")
        if not self.run_id or not self.model_name:
            raise ModelNotAvailable("Registry has no active run/model pointer.")

        self.run_dir = SETTINGS.paths.model_registry / self.run_id
        self.model_path = self.run_dir / f"{self.model_name}.bin"
        if not self.model_path.exists():
            raise ModelNotAvailable(f"Model artefact missing: {self.model_path}")

        # Identifies which registry state this instance reflects, so the API layer can
        # detect that a retrain has landed and reload instead of serving a stale model.
        self.signature = (self.run_id, self.model_name, self._registry_mtime())
        # Explainers are valid for the lifetime of this loaded model; rebuilding them
        # per request costs seconds.
        self._attribution_cache: dict = {}

        import joblib

        self.model = joblib.load(self.model_path)
        self.encoder = FeatureEncoder.load(self.run_dir / "encoder.json")
        self.thresholds = json.loads((self.run_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.reference = load_bundle(self.run_dir / "reference" / f"{self.model_name}.json")
        self.run_meta = (
            json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
            if (self.run_dir / "run.json").exists()
            else {}
        )

        sound_path = self.run_dir / "sound_stats.parquet"
        self.sound_stats = pl.read_parquet(sound_path) if sound_path.exists() else None

        # Background distribution for SHAP. Without it, a row explained against
        # itself produces zero attributions.
        ref_path = self.run_dir / "reference_features.parquet"
        self.reference_features = pl.read_parquet(ref_path) if ref_path.exists() else None
        log.info("predictor ready: run=%s model=%s", self.run_id, self.model_name)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _registry_mtime() -> float:
        try:
            return (SETTINGS.paths.model_registry / "index.json").stat().st_mtime
        except OSError:
            return 0.0

    @classmethod
    def current_signature(cls) -> tuple[str, str, float] | None:
        """Signature of the registry as it exists on disk right now."""
        registry_path = SETTINGS.paths.model_registry / "index.json"
        if not registry_path.exists():
            return None
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        run_id = registry.get("active_run")
        model_name = registry.get("active_model")
        if not run_id or not model_name:
            return None
        return (run_id, model_name, cls._registry_mtime())

    @property
    def version(self) -> str:
        return f"{self.model_name}@{self.run_id}"

    def _row_from_payload(self, payload: dict[str, Any]) -> pl.DataFrame:
        """Normalise a request into the raw columns the feature layer expects."""
        posted_at = payload.get("posted_at") or datetime.now(timezone.utc)
        if isinstance(posted_at, str):
            posted_at = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
        if posted_at.tzinfo is None:
            posted_at = posted_at.replace(tzinfo=timezone.utc)

        hashtags = payload.get("hashtags") or []
        if isinstance(hashtags, str):
            hashtags = [h.strip().lstrip("#") for h in hashtags.split(",") if h.strip()]

        sound_id = payload.get("sound_id")
        video_id = payload.get("video_id") or 0

        row = {
            "video_id": int(video_id or 0),
            "posted_at": posted_at,
            "country": payload.get("country") or "__na__",
            "language": payload.get("language") or "__na__",
            "type": payload.get("type") or "video",
            "duration_ms": int(payload.get("duration_ms") or 0),
            "image_count": int(payload.get("image_count") or 0),
            "caption": payload.get("caption") or "",
            "hashtags": hashtags,
            # None (not 0) when the caller did not supply a sound: 0 is a real
            # identifier and would be treated as an extremely obscure sound.
            "sound_id": int(sound_id) if sound_id is not None else None,
            "is_ad": int(bool(payload.get("is_ad"))),
            "branded_content": payload.get("branded_content") or None,
            "is_shop_video": int(bool(payload.get("is_shop_video"))),
            "is_ai_generated": int(bool(payload.get("is_ai_generated"))),
            "not_recommended_for_fyp": int(bool(payload.get("not_recommended_for_fyp"))),
        }
        return pl.DataFrame([row]).with_columns(
            pl.col("posted_at").cast(pl.Datetime("us", "UTC")),
            pl.col("sound_id").cast(pl.Int64, strict=False),
            pl.col("video_id").cast(pl.Int64, strict=False),
        )

    def _tier(self, percentile: float | None, prob: float) -> tuple[str, str]:
        """Band the prediction by its position among scored videos.

        Absolute probability cut-offs are meaningless here: with a ~1% base rate a
        well-behaved model never emits 0.6, so fixed thresholds would label every
        video "very low". Ranking position is the interpretable quantity.
        """
        if percentile is None:
            return "unknown", "No reference distribution available to place this score."
        for threshold, label, desc in TIER_BANDS:
            if percentile >= threshold:
                return label, desc
        return "very_low", TIER_BANDS[-1][2]

    def _drivers(self, X) -> dict[str, list[dict]]:
        """Feature attributions, or an explicit statement that none are available."""
        try:
            result = local_attributions(
                self.model, X, self.reference_features, self.encoder, cache=self._attribution_cache
            )
            if result is None:
                raise TypeError("no attribution path for this model")
            return result
        except Exception as exc:
            log.debug("attribution unavailable: %s", exc)
            return {
                "method": None,
                "top": [],
                "positive": [],
                "negative": [],
                "note": (
                    "Per-prediction attribution is unavailable for this model. Global "
                    "importance is published in /api/v1/models/evaluation."
                ),
            }

    # ------------------------------------------------------------------ #
    def predict_prepublish(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = self._row_from_payload(payload)
        featured = build_prepublication_features(row, self.sound_stats)
        X = self.encoder.transform(featured)

        prob = float(self.model.predict_proba(X)[:, 1][0])
        view_range = view_range_for(prob, self.reference) if self.reference else None
        # Performance percentile is derived from expected reach, not from the model's
        # own score distribution: scores cluster tightly, so score-position percentiles
        # exaggerate small differences.
        percentile = (
            percentile_of_views(view_range["p50"], self.reference)
            if view_range and self.reference
            else None
        )
        percentile = round(percentile, 1) if percentile is not None else None
        tier, tier_desc = self._tier(percentile, prob)

        threshold = self.thresholds.get("global_threshold")
        drivers = self._drivers(X)
        input_warnings = _input_warnings(payload)
        limitations = [
            "This is a retrospective snapshot model: it estimates whether a video lands in its "
            "cohort's top 1% of observed views, not its exact view count after seven days.",
            "Creator follower counts are not present in the source dataset and are not used.",
            "Sound popularity is frozen at the training window and does not reflect today's trends.",
            "Feature attributions describe model behaviour, not causal effects on virality.",
        ]
        if input_warnings:
            limitations.insert(
                0,
                "This prediction was produced from an incomplete description; the fields listed "
                "in input_warnings were defaulted and the score is correspondingly less reliable.",
            )

        return {
            "prediction_id": None,
            "task": "prepublication",
            "viral_probability": round(prob, 4),
            "virality_tier": tier,
            "tier_description": tier_desc,
            "predicted_percentile": percentile,
            "percentile_basis": (
                "Percentile of this video's expected median reach within the observed view "
                "distribution of the validation population "
                f"(n={self.reference.get('n_reference') if self.reference else 0}). "
                "Resolution is limited by the width of the score band it falls in."
            ),
            "expected_views": view_range,
            "decision_threshold_views": round(threshold) if threshold else None,
            "label_definition": self._label_definition(),
            "input_warnings": input_warnings,
            "drivers": drivers,
            "model": {
                "version": self.version,
                "run_id": self.run_id,
                "model_name": self.model_name,
                "trained_at": self.run_meta.get("started_at"),
                "training_samples": self.run_meta.get("sample_files"),
            },
            "inputs_used": {k: _jsonable(v) for k, v in payload.items()},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "limitations": limitations,
        }

    def _label_definition(self) -> str:
        """Describe the label from the stored thresholds, not from a hardcoded string.

        The cohort definition changed once already (an observation-age bucket was
        added); deriving the text prevents the description from drifting out of sync
        with the model that is actually serving.
        """
        percentile = self.thresholds.get("percentile", 0.99)
        keys = self.thresholds.get("keys") or ["country", "type"]
        readable = " × ".join(str(k).replace("_", " ") for k in keys)
        return (
            f"Top {100 * (1 - percentile):.0f}% of observed views within the same "
            f"{readable} peer group, at snapshot time."
        )

    def warm(self) -> None:
        """Run one throwaway prediction so the first real request is not slow.

        Building the SHAP explainer takes seconds; paying that at startup rather than
        on a user's first click is the difference between "fast" and "hung".
        """
        self.predict_prepublish(
            {
                "caption": "warmup",
                "duration_ms": 15_000,
                "country": "US",
                "language": "en",
                "hashtags": ["warmup"],
            }
        )
        log.info("predictor warmed")

    def model_card(self) -> dict:
        """Everything a reviewer needs to judge this model, from stored artefacts."""
        metrics = (self.run_meta.get("metrics") or {}).get(self.model_name) or {}
        return {
            "version": self.version,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "trained_at": self.run_meta.get("started_at"),
            "training_samples": self.run_meta.get("sample_files"),
            "splits": self.run_meta.get("splits"),
            "label": self.run_meta.get("thresholds"),
            "label_diagnostics": self.run_meta.get("label_diagnostics"),
            "metrics": metrics,
            "environment": self.run_meta.get("environment"),
            "feature_count": len(self.encoder.feature_names),
            "features": self.encoder.feature_names,
        }


def _input_warnings(payload: dict[str, Any]) -> list[str]:
    """Name the fields a caller left empty.

    A prediction from an empty request is still a number, and a number with no
    caveat reads as confident. Listing what was defaulted keeps the score honest.
    """
    warnings: list[str] = []
    if not (payload.get("caption") or "").strip():
        warnings.append("caption was empty; caption features were defaulted")
    if not payload.get("duration_ms"):
        warnings.append("duration_ms was not supplied; duration was imputed from training data")
    if not payload.get("country"):
        warnings.append("country was not supplied; the global cohort fallback was used")
    if not payload.get("language"):
        warnings.append("language was not supplied; language was treated as unknown")
    if not payload.get("hashtags"):
        warnings.append("no hashtags were supplied")
    if not payload.get("sound_id"):
        warnings.append("no sound_id was supplied; sound features were imputed")
    return warnings


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
