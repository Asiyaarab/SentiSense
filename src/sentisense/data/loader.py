"""Dataset loaders for the SentiSense pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentisense.config import (
    DEFAULT_DATA_FILE,
    LABEL_COLUMN,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_DATA_FILE,
    TEXT_COLUMN,
)
from sentisense.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the full IMDB dataset from CSV.

    Args:
        path: Explicit path to the CSV. If None, looks in
            ``$RAW_DATA_DIR/$DEFAULT_DATA_FILE``.

    Returns:
        DataFrame with columns ``text`` and ``label`` (renamed for consistency).

    Raises:
        FileNotFoundError: If the dataset is not in any expected location.
    """
    if path is None:
        path = RAW_DATA_DIR / DEFAULT_DATA_FILE

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            f"Place the IMDB CSV there, or pass an explicit path."
        )

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df.rename(columns={TEXT_COLUMN: "text", LABEL_COLUMN: "label"}, inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    logger.info("Loaded %d rows", len(df))
    return df.reset_index(drop=True)


def load_sample_dataset() -> pd.DataFrame:
    """Load the small built-in sample CSV (for smoke tests)."""
    sample = RAW_DATA_DIR / SAMPLE_DATA_FILE
    if not sample.exists():
        raise FileNotFoundError(
            f"Sample dataset missing at {sample}. Did the repo install correctly?"
        )
    df = pd.read_csv(sample)
    df.rename(columns={TEXT_COLUMN: "text", LABEL_COLUMN: "label"}, inplace=True)
    logger.info("Loaded %d sample rows", len(df))
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str = "processed.csv") -> Path:
    """Persist a preprocessed DataFrame to disk (for audit / reuse)."""
    out = PROCESSED_DATA_DIR / name
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    logger.info("Saved processed data to %s", out)
    return out
