"""Shared test configuration.

Tests must never touch the developer's live application database: DuckDB holds an
OS-level lock per file, so a running server and the test process would conflict.
A session-wide temporary database keeps the suite hermetic and re-runnable.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_database():
    path = Path(tempfile.mkdtemp()) / "viraliq_test.duckdb"
    os.environ["VIRALIQ_DB_PATH"] = str(path)
    yield path
    os.environ.pop("VIRALIQ_DB_PATH", None)
