"""Feature encoding shared by every model.

Categorical vocabularies are fitted on the training window only. Unseen
categories at inference time map to ``__unseen__`` rather than crashing or
silently becoming a null, which keeps production behaviour identical to
evaluation behaviour.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.common.logging import get_logger
from src.feature_engineering.features import CATEGORICAL_FEATURES, PREPUBLICATION_FEATURES

log = get_logger("encoder")

UNSEEN = "__unseen__"
NULL = "__null__"


@dataclasses.dataclass
class FeatureEncoder:
    numeric: tuple[str, ...] = PREPUBLICATION_FEATURES
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES
    # Kept modest deliberately: one-hot expansion of wide categorical vocabularies
    # is the dominant memory cost on a single-machine deployment.
    max_categories: int = 25
    vocab: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    fill_values: dict[str, float] = dataclasses.field(default_factory=dict)

    # ---- fitting -----------------------------------------------------------
    def fit(self, df: pl.DataFrame) -> "FeatureEncoder":
        self.vocab = {}
        for col in self.categorical:
            counts = (
                df.select(pl.col(col).cast(pl.Utf8).fill_null(NULL).alias(col))
                .group_by(col)
                .agg(pl.len().alias("n"))
                .sort("n", descending=True)
                .head(self.max_categories)
            )
            cats = counts[col].to_list()
            if UNSEEN not in cats:
                cats.append(UNSEEN)
            self.vocab[col] = cats

        self.fill_values = {}
        for col in self.numeric:
            series = df[col].cast(pl.Float64, strict=False)
            median = series.median()
            self.fill_values[col] = float(median) if median is not None else 0.0
        log.info(
            "encoder fitted: %d numeric, %d categorical (%s)",
            len(self.numeric),
            len(self.categorical),
            ", ".join(f"{c}={len(v)}" for c, v in self.vocab.items()),
        )
        return self

    # ---- transforming ------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> pd.DataFrame:
        if not self.vocab:
            raise RuntimeError("FeatureEncoder.transform called before fit")

        numeric_part = df.select(
            [pl.col(c).cast(pl.Float32, strict=False).alias(c) for c in self.numeric]
        ).to_pandas()

        for col in self.numeric:
            numeric_part[col] = numeric_part[col].fillna(
                np.float32(self.fill_values.get(col, 0.0))
            )

        cat_part = pd.DataFrame(index=numeric_part.index)
        for col in self.categorical:
            raw = (
                df.select(pl.col(col).cast(pl.Utf8).fill_null(NULL).alias(col))[col]
                .to_pandas()
            )
            allowed = set(self.vocab[col]) - {UNSEEN}
            mapped = raw.where(raw.isin(allowed), UNSEEN).astype("string")
            cat_part[col] = pd.Categorical(mapped, categories=self.vocab[col])

        return pd.concat([numeric_part, cat_part], axis=1)

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "numeric": list(self.numeric),
                    "categorical": list(self.categorical),
                    "max_categories": self.max_categories,
                    "vocab": self.vocab,
                    "fill_values": self.fill_values,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            numeric=tuple(payload["numeric"]),
            categorical=tuple(payload["categorical"]),
            max_categories=payload.get("max_categories", 60),
            vocab=payload["vocab"],
            fill_values=payload["fill_values"],
        )

    @property
    def feature_names(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)
