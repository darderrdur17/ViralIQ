"""DuckDB access layer.

The 460 GB dataset is never downloaded. DuckDB reads Parquet over HTTPS with
column pruning and predicate pushdown, so a query only transfers the row groups
and columns it needs.
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import duckdb

from src.common.config import SETTINGS
from src.common.logging import get_logger

log = get_logger("duckdb")

_HTTP_READY = False


def _prepare(con: duckdb.DuckDBPyConnection) -> None:
    global _HTTP_READY
    if not _HTTP_READY:
        try:
            con.execute("INSTALL httpfs;")
            con.execute("LOAD httpfs;")
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("httpfs unavailable (%s); remote reads will fail", exc)
        _HTTP_READY = True
    else:
        with contextlib.suppress(Exception):
            con.execute("LOAD httpfs;")

    con.execute("SET TimeZone='UTC';")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET threads=4;")
    # A hard ceiling keeps a wide aggregation from taking down the process on a
    # small machine; DuckDB spills to disk instead of being OOM-killed.
    with contextlib.suppress(Exception):
        con.execute("SET memory_limit='3GB';")
    # Remote HTTPS reads benefit from metadata caching across repeated scans.
    with contextlib.suppress(Exception):
        con.execute("SET enable_http_metadata_cache=true;")


def connect(path: str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with httpfs loaded.

    Passing ``:memory:`` gives an ephemeral analytical connection; passing a file
    path opens the persistent application database.
    """
    target = path or ":memory:"
    con = duckdb.connect(target, read_only=read_only)
    _prepare(con)
    return con


@contextlib.contextmanager
def session(path: str | None = None, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def app_db(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Connection to the persistent application database (observations, history)."""
    SETTINGS.paths.database.mkdir(parents=True, exist_ok=True)
    return connect(str(SETTINGS.paths.duckdb_file), read_only=read_only)


def remote_scan(urls: list[str] | str, columns: list[str] | None = None) -> str:
    """Build a ``read_parquet`` expression for one or more remote URLs."""
    if isinstance(urls, str):
        urls = [urls]
    if len(urls) == 1:
        base = f"read_parquet('{urls[0]}')"
    else:
        joined = ", ".join(f"'{u}'" for u in urls)
        base = f"read_parquet([{joined}])"
    if columns:
        cols = ", ".join(columns)
        return f"(SELECT {cols} FROM {base})"
    return base
