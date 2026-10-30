"""Hugging Face dataset connector.

This adapter is deliberately read-only and licence-aware. It discovers monthly
Parquet partitions through the Hugging Face API and exposes their resolve URLs
for remote querying. It never downloads a full partition unless explicitly asked.
"""
from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone

from src.common.config import SETTINGS, SourceConfig
from src.common.logging import get_logger

log = get_logger("hf_source")


@dataclasses.dataclass(frozen=True)
class Partition:
    path: str
    month: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def year(self) -> int:
        return int(self.month.split("-")[0])


class HFDatasetSource:
    """Connector for a Hugging Face Parquet dataset."""

    name = "huggingface"

    def __init__(self, config: SourceConfig | None = None) -> None:
        self.config = config or SETTINGS.source
        self._cache: list[Partition] | None = None

    def list_partitions(self, refresh: bool = False) -> list[Partition]:
        if self._cache is not None and not refresh:
            return self._cache
        req = urllib.request.Request(
            self.config.api_tree_url, headers={"User-Agent": "ViralIQ/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        parts: list[Partition] = []
        for entry in payload:
            if entry.get("type") != "file" or not entry["path"].endswith(".parquet"):
                continue
            month = entry["path"].rsplit("/", 1)[-1].removesuffix(".parquet")
            parts.append(Partition(path=entry["path"], month=month, size_bytes=entry.get("size", 0)))
        parts.sort(key=lambda p: p.month)
        self._cache = parts
        log.info("discovered %d monthly partitions (%s .. %s)", len(parts), parts[0].month, parts[-1].month)
        return parts

    def partitions_for(self, months: list[str] | None = None, min_year: int | None = None) -> list[Partition]:
        parts = self.list_partitions()
        if months:
            wanted = set(months)
            parts = [p for p in parts if p.month in wanted]
        if min_year is not None:
            parts = [p for p in parts if p.year >= min_year]
        return parts

    def urls(self, partitions: list[Partition]) -> list[str]:
        return [self.config.resolve_url(p.path) for p in partitions]

    def describe(self) -> dict:
        parts = self.list_partitions()
        total = sum(p.size_bytes for p in parts)
        return {
            "source": self.name,
            "repo_id": self.config.repo_id,
            "license": self.config.license,
            "partition_count": len(parts),
            "total_bytes": total,
            "total_gb": round(total / 1e9, 2),
            "earliest_month": parts[0].month if parts else None,
            "latest_month": parts[-1].month if parts else None,
            "latest_partition_gb": round(parts[-1].size_gb, 2) if parts else None,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def is_available(self) -> tuple[bool, str | None]:
        try:
            self.list_partitions(refresh=True)
            return True, None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return False, str(exc)
