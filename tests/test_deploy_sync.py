"""Guard against the serverless mirror drifting from the real source.

`deploy/vercel/src/` is a generated copy of `src/` that the serverless function
imports. If someone edits `src/` without re-running the export, the deployed site
keeps executing the old feature-engineering code — and `tests/test_portable.py`
would not catch it, because that test imports from `src/`, not from the mirror.

This test makes the drift explicit and tells you the fix.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
MIRROR = ROOT / "deploy" / "vercel" / "src"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not MIRROR.exists(), reason="serverless deployment not generated")
def test_serverless_mirror_matches_source():
    problems: list[str] = []

    for src_file in sorted(SOURCE.rglob("*.py")):
        if "__pycache__" in src_file.parts:
            continue
        mirror_file = MIRROR / src_file.relative_to(SOURCE)
        if not mirror_file.exists():
            problems.append(f"missing in mirror: {src_file.relative_to(ROOT)}")
        elif _digest(src_file) != _digest(mirror_file):
            problems.append(f"differs: {src_file.relative_to(ROOT)}")

    for mirror_file in sorted(MIRROR.rglob("*.py")):
        if "__pycache__" in mirror_file.parts:
            continue
        if not (SOURCE / mirror_file.relative_to(MIRROR)).exists():
            problems.append(f"stale in mirror: {mirror_file.relative_to(ROOT)}")

    assert not problems, (
        "deploy/vercel/src is out of sync with src/. "
        "Run `python scripts/export_portable.py` and commit the result.\n  "
        + "\n  ".join(problems)
    )


@pytest.mark.skipif(not (ROOT / "deploy" / "vercel" / "public").exists(),
                    reason="serverless deployment not generated")
def test_serverless_dashboard_copy_matches_source():
    source = ROOT / "dashboard" / "index.html"
    copy = ROOT / "deploy" / "vercel" / "public" / "index.html"
    assert _digest(source) == _digest(copy), (
        "deploy/vercel/public/index.html is stale. "
        "Run `python scripts/export_portable.py` and commit the result."
    )
