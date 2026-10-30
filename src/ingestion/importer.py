"""Import external Parquet/CSV files into the sample store.

The fallback import path has to be trustworthy: a file that is accepted here becomes
part of the analytical corpus. Three rules follow from that.

1. **Read correctly.** Parquet and CSV need different reader arguments; passing CSV
   options to the Parquet reader fails for every file.
2. **Normalise, then store.** A CSV exports ``hashtags`` as ``"a,b"`` and timestamps
   as text. Accepting that unchanged produces a sample that breaks the feature
   pipeline later, far from the point of the mistake.
3. **Never overwrite.** An import must not be able to destroy an existing sample
   that training depends on.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import polars as pl

from src.common.logging import get_logger
from src.preprocessing.quality import validate_schema

log = get_logger("importer")

NUMERIC_COLUMNS = (
    "duration_ms",
    "image_count",
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "downloads",
    "sound_id",
    "video_id",
    "is_ad",
    "is_shop_video",
    "is_ai_generated",
    "not_recommended_for_fyp",
)

TIMESTAMP_COLUMNS = ("posted_at", "stats_updated_at")


class ImportRejected(ValueError):
    """Raised when a file cannot be safely imported."""


@dataclasses.dataclass(frozen=True)
class ImportResult:
    rows: int
    columns: int
    path: str
    original_format: str
    normalisations: list[str]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _read(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path, infer_schema_length=10_000)
    except Exception as exc:
        raise ImportRejected(f"Could not parse {suffix} file: {exc}") from exc
    raise ImportRejected(f"Unsupported file type: {suffix or '(no extension)'}")


def _normalise_hashtags(df: pl.DataFrame) -> tuple[pl.DataFrame, str | None]:
    """Coerce a hashtag column to a list of strings.

    Parquet usually arrives as a proper list; CSV always arrives as text.
    """
    if "hashtags" not in df.columns:
        return df, None
    dtype = df.schema["hashtags"]
    if isinstance(dtype, pl.List):
        return df, None
    if dtype == pl.String:
        cleaned = (
            pl.col("hashtags")
            .fill_null("")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars().str.strip_chars("#"))
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        )
        return df.with_columns(cleaned.alias("hashtags")), "hashtags split from delimited text"
    return df.with_columns(pl.lit(None, dtype=pl.List(pl.String)).alias("hashtags")), (
        f"hashtags of type {dtype} discarded (not list or text)"
    )


def _normalise_types(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    notes: list[str] = []
    for col in TIMESTAMP_COLUMNS:
        if col not in df.columns:
            continue
        if df.schema[col] == pl.String:
            parsed = pl.col(col).str.to_datetime(time_zone="UTC", strict=False)
            df = df.with_columns(parsed.alias(col))
            notes.append(f"{col} parsed from text")
        elif isinstance(df.schema[col], pl.Datetime):
            if df.schema[col].time_zone is None:
                df = df.with_columns(pl.col(col).dt.replace_time_zone("UTC"))
                notes.append(f"{col} assumed UTC")

    for col in NUMERIC_COLUMNS:
        if col in df.columns and df.schema[col] == pl.String:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
            notes.append(f"{col} cast from text to integer")

    return df, notes


def load_and_normalise(path: Path) -> tuple[pl.DataFrame, ImportResult]:
    """Read, validate and normalise an external file."""
    path = Path(path).expanduser()
    if not path.exists():
        raise ImportRejected(f"File not found: {path}")

    df = _read(path)
    check = validate_schema(list(df.columns))
    if not check.valid:
        raise ImportRejected(
            "Schema validation failed; missing required columns: "
            + ", ".join(check.missing_required)
        )
    if df.height == 0:
        raise ImportRejected("File contains no rows")

    df, tag_note = _normalise_hashtags(df)
    df, type_notes = _normalise_types(df)
    notes = [n for n in [tag_note, *type_notes] if n]

    # Rows with an unparseable publication time cannot be placed in a split.
    before = df.height
    df = df.filter(pl.col("posted_at").is_not_null())
    if df.height < before:
        notes.append(f"{before - df.height} row(s) dropped for unparseable posted_at")
    if df.height == 0:
        raise ImportRejected("No rows remain after dropping unparseable publication times")

    return df, ImportResult(
        rows=df.height,
        columns=df.width,
        path=str(path),
        original_format=path.suffix.lower().lstrip("."),
        normalisations=notes,
    )


def store_sample(df: pl.DataFrame, source_path: Path, target_dir: Path) -> Path:
    """Write the normalised frame as a new Parquet sample, never overwriting.

    Imported files are always persisted as Parquet: analytics and training glob
    ``*.parquet``, so a CSV left as CSV would be silently invisible to both.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = f"import_{source_path.stem}"[:60]
    digest = hashlib.sha1(str(source_path).encode()).hexdigest()[:8]
    target = target_dir / f"{stem}_{digest}.parquet"
    if target.exists():
        raise ImportRejected(
            f"A sample named {target.name} already exists; refusing to overwrite it"
        )
    df.write_parquet(target)
    log.info("imported %d rows -> %s", df.height, target.name)
    return target
