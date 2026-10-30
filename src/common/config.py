"""Central configuration for ViralIQ.

All paths resolve relative to the project root so the package behaves identically
whether it is imported by the API, a script, or a notebook.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


@dataclasses.dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_external: Path = PROJECT_ROOT / "data" / "external"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_samples: Path = PROJECT_ROOT / "data" / "samples"
    data_metadata: Path = PROJECT_ROOT / "data" / "metadata"
    models_trained: Path = PROJECT_ROOT / "models" / "trained"
    models_checkpoints: Path = PROJECT_ROOT / "models" / "checkpoints"
    model_registry: Path = PROJECT_ROOT / "models" / "model_registry"
    reports_research: Path = PROJECT_ROOT / "reports" / "research"
    reports_evaluation: Path = PROJECT_ROOT / "reports" / "evaluation"
    reports_insights: Path = PROJECT_ROOT / "reports" / "insights"
    reports_executive: Path = PROJECT_ROOT / "reports" / "executive"
    database: Path = PROJECT_ROOT / "database"

    @property
    def duckdb_file(self) -> Path:
        return self.database / "viraliq.duckdb"

    @property
    def freshness_file(self) -> Path:
        return self.data_metadata / "freshness.json"

    @property
    def audit_file(self) -> Path:
        return self.data_metadata / "dataset_audit.json"

    def ensure(self) -> None:
        for f in dataclasses.fields(self):
            if f.name == "root":
                continue
            value = getattr(self, f.name)
            if isinstance(value, Path) and not value.suffix:
                value.mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass(frozen=True)
class SourceConfig:
    """Hugging Face dataset source definition."""

    repo_id: str = "datasocial/tiktok-5.6B-videos"
    revision: str = "main"
    data_prefix: str = "data"
    license: str = "CC BY-NC 4.0"

    @property
    def api_tree_url(self) -> str:
        return (
            f"https://huggingface.co/api/datasets/{self.repo_id}"
            f"/tree/{self.revision}/{self.data_prefix}?recursive=true"
        )

    def resolve_url(self, path: str) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}/{path}"


@dataclasses.dataclass(frozen=True)
class LabelConfig:
    """Virality target definition.

    `percentile` defines the cohort-relative virality threshold. The blueprint is
    explicit that a hard-coded global view count is not acceptable, so the label is
    always derived from a within-cohort percentile.
    """

    percentile: float = 0.99
    # Age bucket is part of the peer group: views at 3 hours and views at 20 days are
    # not comparable, so a video is only ranked against videos of similar age.
    cohort_keys: tuple[str, ...] = ("country", "type", "observation_age_bucket")
    min_cohort_size: int = 200
    # Observation-age validity window. Negative ages are impossible (counters
    # timestamped before publication) and are always excluded.
    min_observation_age_hours: float = 0.0
    max_observation_age_hours: float = 720.0


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Training knobs that are actually consumed.

    Split boundaries are not configurable: they are derived from the calendar in
    ``temporal_slices``. Exposing a ``test_fraction`` here that nothing reads would
    imply a control that does not exist.
    """

    random_state: int = 42
    top_k_values: tuple[int, ...] = (1000, 5000, 20000)
    bootstrap_iterations: int = 200


@dataclasses.dataclass(frozen=True)
class Settings:
    paths: Paths = dataclasses.field(default_factory=Paths)
    source: SourceConfig = dataclasses.field(default_factory=SourceConfig)
    labels: LabelConfig = dataclasses.field(default_factory=LabelConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)

    def ensure_dirs(self) -> None:
        self.paths.ensure()


def _coerce(cls: type, payload: dict[str, Any] | None):
    """Build a frozen dataclass from a dict, ignoring unknown keys."""
    payload = payload or {}
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in payload.items() if k in valid}
    for key, value in list(kwargs.items()):
        if isinstance(value, list):
            kwargs[key] = tuple(value)
    return cls(**kwargs)


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or DEFAULT_CONFIG
    payload: dict[str, Any] = {}
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    paths = _coerce(Paths, payload.get("paths"))
    source = _coerce(SourceConfig, payload.get("source"))
    labels = _coerce(LabelConfig, payload.get("labels"))
    model = _coerce(ModelConfig, payload.get("model"))

    # Environment overrides for deployment without editing YAML.
    repo_env = os.environ.get("VIRALIQ_HF_REPO")
    if repo_env:
        source = dataclasses.replace(source, repo_id=repo_env)

    settings = Settings(paths=paths, source=source, labels=labels, model=model, raw=payload)
    settings.ensure_dirs()
    return settings


SETTINGS = load_settings()
