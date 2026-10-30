"""Uniform logging setup."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str = "viraliq") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("VIRALIQ_LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("viraliq")
        root.setLevel(getattr(logging, level, logging.INFO))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("viraliq") else f"viraliq.{name}")
