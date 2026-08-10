"""Centralized configuration for SentiSense.

All paths, hyperparameters, and environment-specific settings live here.
Override via environment variables (e.g. SENTISENSE_DATA_DIR) without
touching source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root = parent of src/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Read a path from an env var, falling back to a default Path."""
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


DATA_DIR: Path = _env_path("SENTISENSE_DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
MODELS_DIR: Path = _env_path("SENTISENSE_MODELS_DIR", PROJECT_ROOT / "models")
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# Default dataset file (full IMDb 50K)
DEFAULT_DATA_FILE: str = "IMDB Dataset.csv"
SAMPLE_DATA_FILE: str = "sample_reviews.csv"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

TEXT_COLUMN: str = "review"
LABEL_COLUMN: str = "sentiment"
POSITIVE_LABEL: str = "positive"
NEGATIVE_LABEL: str = "negative"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorizerConfig:
    """TF-IDF vectorizer hyperparameters."""

    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    strip_accents: str = "unicode"


@dataclass(frozen=True)
class TrainConfig:
    """Training pipeline hyperparameters."""

    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Per-model hyperparameters."""

    name: str = "logreg"
    max_iter: int = 1000
    C: float = 1.0
    n_jobs: int = -1  # Only used for SVM (CalibratedClassifierCV). Ignored by LogReg.


@dataclass(frozen=True)
class AppConfig:
    """Aggregate app-level config (everything in one immutable bundle)."""

    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Singleton — easy to import: `from sentisense.config import settings`
settings: AppConfig = AppConfig()


# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------

NLTK_RESOURCES: dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
