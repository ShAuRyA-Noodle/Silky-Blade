"""Shared pytest fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Pin MLflow tracking to a local file:// store BEFORE pydantic-settings
# loads the Settings singleton (which is module-level in quant.config).
# Without this, tests inherit `http://mlflow:5000` from the dev .env and
# fail to resolve the Docker hostname when the stack isn't running.
_MLFLOW_TEST_DIR = Path(tempfile.gettempdir()) / "quant-tests-mlruns"
_MLFLOW_TEST_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = _MLFLOW_TEST_DIR.as_uri()


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never accidentally hit production services."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("APP_DEBUG", "true")
    monkeypatch.setenv("TRADING_ENABLED", "false")
    # The Settings singleton may already be populated with the prod URI.
    # Force the field on the live object so trainer.py reads our local path.
    from quant.config import settings as _settings

    monkeypatch.setattr(_settings, "mlflow_tracking_uri", _MLFLOW_TEST_DIR.as_uri())
