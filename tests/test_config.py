"""Tests for sentisense.config and sentisense.utils."""

import shutil
from pathlib import Path

import pytest

from sentisense.config import (
    MODELS_DIR,
    RAW_DATA_DIR,
    AppConfig,
    settings,
)
from sentisense.utils import ensure_dir, ensure_nltk_resources, get_logger


def test_settings_is_app_config() -> None:
    assert isinstance(settings, AppConfig)


def test_sub_configs_are_frozen() -> None:
    """Frozen dataclasses prevent accidental mutation in long-running services."""
    from dataclasses import FrozenInstanceError


    with pytest.raises(FrozenInstanceError):
        settings.vectorizer.max_features = 1  # type: ignore[misc]


def test_logger_has_handler() -> None:
    log = get_logger("test.logger")
    assert len(log.handlers) >= 1
    assert log.level in (10, 20, 30, 40, 50)  # one of DEBUG..CRITICAL


def test_ensure_dir_creates_nested() -> None:
    target = Path("/tmp/sentisense_test_nested/sub")
    if target.exists():
        shutil.rmtree(target.parent)
    ensure_dir(target)
    assert target.exists()
    # Idempotent
    ensure_dir(target)
    assert target.exists()
    shutil.rmtree(target.parent)


def test_ensure_nltk_resources_idempotent() -> None:
    # Should not raise on a second call.
    ensure_nltk_resources(quiet=True)
    ensure_nltk_resources(quiet=True)


def test_path_helpers() -> None:
    assert isinstance(RAW_DATA_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
    assert RAW_DATA_DIR.name == "raw"
