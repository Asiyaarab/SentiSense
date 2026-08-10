"""Logging, NLTK setup, and small reusable helpers."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

import nltk

from sentisense.config import NLTK_RESOURCES

_LOG_FORMAT: Final[str] = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a configured logger. Use this everywhere — no raw print()."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    if level is not None:
        logger.setLevel(level)
    elif os.getenv("SENTISENSE_DEBUG"):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def ensure_nltk_resources(quiet: bool = True) -> None:
    """Download NLTK data if missing. Idempotent and safe to call repeatedly."""
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            if not quiet:
                print(f"Downloading NLTK resource: {name} ...")
            nltk.download(name, quiet=quiet)


def ensure_dir(path: os.PathLike | str) -> None:
    """Create a directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
