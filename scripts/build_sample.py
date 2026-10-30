"""Build a local sample from remote monthly partitions.

Usage:
    python scripts/build_sample.py --months 2026-08 2026-09 --fraction 0.01
    python scripts/build_sample.py --months 2026-10 --fraction 0.5 --name recent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.sample_builder import build_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reproducible local sample")
    parser.add_argument("--months", nargs="+", required=True, help="e.g. 2026-08 2026-09")
    parser.add_argument("--fraction", type=float, default=0.01, help="Hash-sampling fraction (0-1]")
    parser.add_argument("--name", default=None, help="Output file stem")
    args = parser.parse_args()

    result = build_sample(months=args.months, fraction=args.fraction, name=args.name)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
