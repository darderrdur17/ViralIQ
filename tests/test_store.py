"""Persistence-layer tests.

The store previously opened a new DuckDB connection per request. DuckDB allows one
handle per database file per process, so concurrent predictions raised
``Unique file handle conflict`` and returned 500. These tests keep that fixed.
"""
from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from api.services import store


@pytest.fixture(autouse=True)
def isolated_database():
    """Use a throwaway database.

    DuckDB holds an OS-level lock per file, so a running server and the test process
    cannot share the application database.
    """
    path = Path(tempfile.mkdtemp()) / "test.duckdb"
    store.configure(str(path))
    yield path
    store.configure(None)


def test_concurrent_writes_do_not_conflict():
    """Regression: 16 concurrent predictions used to fail ~80% of the time."""
    def write(i: int) -> str:
        return store.save_prediction(
            task="prepublication",
            model_version=f"test@run{i}",
            payload={"caption": f"concurrent {i}"},
            response={"viral_probability": 0.02, "virality_tier": "low"},
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        ids = list(pool.map(write, range(16)))

    assert len(set(ids)) == 16, "prediction ids must be unique"
    assert all(isinstance(pid, str) and pid for pid in ids)


def test_stats_remain_readable_after_concurrent_writes():
    """Stats must work on an empty store and after concurrent inserts."""
    empty = store.prediction_stats()
    assert empty["total_predictions"] == 0
    assert empty["latest_prediction_at"] is None

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: store.save_prediction(
                    task="prepublication",
                    model_version="test@stats",
                    payload={"i": i},
                    response={"viral_probability": 0.02, "virality_tier": "low"},
                ),
                range(8),
            )
        )

    stats = store.prediction_stats()
    assert stats["total_predictions"] == 8
    assert stats["latest_prediction_at"] is not None
    assert stats["mean_viral_probability"] == pytest.approx(0.02, abs=1e-6)


def test_history_returns_parsed_payload_and_response():
    store.save_prediction(
        task="prepublication",
        model_version="test@history",
        payload={"caption": "history check"},
        response={"viral_probability": 0.031, "virality_tier": "low"},
    )
    items = store.list_predictions(limit=5)
    assert items, "history should contain the prediction just written"
    first = items[0]
    assert isinstance(first["payload"], dict), "payload JSON must be parsed, not raw text"
    assert isinstance(first["response"], dict)
    assert first["payload"]["caption"] == "history check"


def test_history_task_filter_excludes_other_tasks():
    store.save_prediction(
        task="early_traction",
        model_version="test@filter",
        payload={},
        response={"viral_probability": 0.5},
    )
    filtered = store.list_predictions(limit=50, task="early_traction")
    assert all(item["task"] == "early_traction" for item in filtered)


def test_observation_count_is_integer():
    assert isinstance(store.observation_count(), int)


def test_concurrent_reads_and_writes_interleave_safely():
    """A reader running alongside writers must not raise."""
    def mixed(i: int) -> str:
        if i % 3 == 0:
            store.list_predictions(limit=3)
            return "read"
        store.save_prediction(
            task="prepublication",
            model_version="test@mixed",
            payload={"i": i},
            response={"viral_probability": 0.01},
        )
        return "write"

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(mixed, range(24)))

    assert results.count("write") == 16
    assert results.count("read") == 8
