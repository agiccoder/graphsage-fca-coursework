"""Minimal, consistent logging setup."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "graphsage_fca", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)
