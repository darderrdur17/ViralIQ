"""Feature engineering.

Two disjoint feature sets are produced, matching the two prediction engines in
the blueprint:

``PREPUBLICATION_FEATURES``
    Everything knowable before the video is published. No engagement counters,
    no future sound usage, no future hashtag usage.

``EARLY_TRACTION_FEATURES``
    Only valid when an observation *earlier than the prediction timestamp* has
    actually been recorded. These are built from the observation store, not from
    the static dataset, because the static dataset holds a single snapshot.

Anything that cannot be computed honestly is not produced. A missing feature is
recoverable; a leaked feature silently invalidates every reported number.
"""
from __future__ import annotations

import re

import polars as pl

from src.common.logging import get_logger
from src.labeling.labels import assert_no_performance_features

log = get_logger("features")

EMOJI_PATTERN = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U0001F1E6-\U0001F1FF" "\u2600-\u27BF" "\u2B00-\u2BFF" "]"
)

PREPUBLICATION_FEATURES: tuple[str, ...] = (
    # content
    "duration_s",
    "log_duration",
    "is_photo_carousel",
    "caption_len",
    "caption_words",
    "caption_emoji_count",
    "caption_exclaim_count",
    "caption_question_count",
    "caption_upper_ratio",
    "caption_has_mention",
    "hashtag_count",
    "log_hashtag_count",
    # publishing
    "posted_hour",
    "posted_dow",
    "is_weekend",
    # commercial / format flags
    "is_ad",
    "is_shop_video",
    "is_ai_generated",
    "not_recommended_for_fyp",
    "has_branded_content",
    # sound
    "sound_is_original",
    "sound_train_usage_log",
    "sound_train_median_views_log",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("country", "language", "type")

EARLY_TRACTION_FEATURES: tuple[str, ...] = (
    "views_at_obs",
    "log_views_at_obs",
    "like_rate",
    "share_rate",
    "comment_rate",
    "save_rate",
    "views_per_hour",
    "observation_age_hours",
)


def _bool_to_int(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int8).fill_null(0)


def build_prepublication_features(
    df: pl.DataFrame,
    sound_stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Derive the pre-publication feature set.

    ``sound_stats`` must be computed on the training window only (see
    :func:`fit_sound_stats`); passing test-window statistics would leak future
    sound popularity into training.
    """
    caption = pl.col("caption").fill_null("")
    # Duration nulls are preserved, not zeroed: a missing duration is unknown, not
    # a zero-length video. The encoder imputes the training median.
    duration_s = pl.col("duration_ms").cast(pl.Float64) / 1000.0
    out = df.with_columns(
        duration_s.alias("duration_s"),
        duration_s.log1p().alias("log_duration"),
        (pl.col("image_count").fill_null(0) > 0).cast(pl.Int8).alias("is_photo_carousel"),
        caption.str.len_chars().alias("caption_len"),
        caption.str.split(" ").list.len().alias("caption_words"),
        caption.str.count_matches(EMOJI_PATTERN.pattern).alias("caption_emoji_count"),
        caption.str.count_matches("!").alias("caption_exclaim_count"),
        caption.str.count_matches(r"\?").alias("caption_question_count"),
        pl.when(caption.str.len_chars() > 0)
        .then(
            caption.str.count_matches(r"[A-Z]").cast(pl.Float64)
            / caption.str.len_chars().cast(pl.Float64)
        )
        .otherwise(0.0)
        .alias("caption_upper_ratio"),
        caption.str.contains("@").cast(pl.Int8).alias("caption_has_mention"),
        pl.col("hashtags").list.len().fill_null(0).alias("hashtag_count"),
        pl.col("hashtags").list.len().fill_null(0).log1p().alias("log_hashtag_count"),
        pl.col("posted_at").dt.hour().alias("posted_hour"),
        pl.col("posted_at").dt.weekday().alias("posted_dow"),
        (pl.col("posted_at").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        _bool_to_int("is_ad").alias("is_ad"),
        _bool_to_int("is_shop_video").alias("is_shop_video"),
        _bool_to_int("is_ai_generated").alias("is_ai_generated"),
        _bool_to_int("not_recommended_for_fyp").alias("not_recommended_for_fyp"),
        (
            pl.col("branded_content").is_not_null()
            & (pl.col("branded_content").cast(pl.Utf8) != "")
        )
        .cast(pl.Int8)
        .alias("has_branded_content"),
        # Original sounds carry sound_id == video_id on TikTok. An unknown sound is
        # not an original sound, so nulls resolve to 0 rather than being imputed.
        (pl.col("sound_id") == pl.col("video_id")).cast(pl.Int8).fill_null(0).alias("sound_is_original"),
    )

    if sound_stats is not None and sound_stats.height:
        # An absent or unseen sound must stay NULL, not become zero usage: zero is a
        # real value meaning "extremely obscure sound", and the model would read it as
        # one. NULL propagates to the encoder, which imputes the training median.
        stats = sound_stats.with_columns(pl.col("sound_id").cast(pl.Int64))
        out = out.with_columns(pl.col("sound_id").cast(pl.Int64))
        out = out.join(stats, on="sound_id", how="left").with_columns(
            pl.col("sound_train_usage").log1p().alias("sound_train_usage_log"),
            pl.col("sound_train_median_views").log1p().alias("sound_train_median_views_log"),
        ).drop("sound_train_usage", "sound_train_median_views")
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("sound_train_usage_log"),
            pl.lit(None, dtype=pl.Float64).alias("sound_train_median_views_log"),
        )

    return out


def fit_sound_stats(train_df: pl.DataFrame) -> pl.DataFrame:
    """Sound popularity statistics frozen on the training window.

    Using only training-period rows means a test video's sound popularity reflects
    what was known *before* that period, not the popularity it went on to achieve.
    """
    stats = (
        train_df.group_by("sound_id")
        .agg(
            pl.len().alias("sound_train_usage"),
            pl.col("views").median().alias("sound_train_median_views"),
        )
    )
    log.info("sound stats fitted on %d distinct sounds", stats.height)
    return stats


def feature_frame(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Select the modelling matrix and validate the leakage guard."""
    cols = list(PREPUBLICATION_FEATURES) + list(CATEGORICAL_FEATURES)
    assert_no_performance_features(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from frame: {missing}")
    return df.select(cols), cols


def build_early_traction_features(observations: pl.DataFrame, horizon_hours: float = 6.0) -> pl.DataFrame:
    """Point-in-time features from the observation store.

    Returns an empty frame when no observation earlier than the horizon exists.
    This is the honest outcome for the static dataset: there is no second
    observation to measure velocity from.
    """
    if observations.height == 0:
        log.warning("no observations available; early-traction features cannot be built")
        return pl.DataFrame()

    eligible = observations.filter(pl.col("observation_age_hours") <= horizon_hours)
    if eligible.height == 0:
        return pl.DataFrame()

    return eligible.with_columns(
        pl.col("views").log1p().alias("log_views_at_obs"),
        (pl.col("likes") / pl.col("views").clip(1)).alias("like_rate"),
        (pl.col("shares") / pl.col("views").clip(1)).alias("share_rate"),
        (pl.col("comments") / pl.col("views").clip(1)).alias("comment_rate"),
        (pl.col("saves") / pl.col("views").clip(1)).alias("save_rate"),
        (pl.col("views") / pl.col("observation_age_hours").clip(0.01)).alias("views_per_hour"),
    ).rename({"views": "views_at_obs"})
