"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

# Make `import sentisense` work when running `pytest` from project root
# without a prior `pip install -e .`.
# BUGFIX: this pointed at the repo root; `sentisense` lives under
# `<repo root>/src/sentisense`, so this line alone never made the import
# work (CI's `pip install -e ".[dev]"` was doing that job silently).
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def sample_texts() -> list[str]:
    return [
        "This was an absolutely brilliant movie!",
        "Boring, predictable, a complete waste of time.",
        "I loved every minute of it.",
        "Painful to watch. Avoid at all costs.",
    ]


@pytest.fixture(scope="session")
def sample_labels() -> list[str]:
    return ["positive", "negative", "positive", "negative"]
