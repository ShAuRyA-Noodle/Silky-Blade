"""Integration tests for the /api/v1/paper read endpoints."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from quant.config import get_settings, settings
from quant.core.dependencies import get_current_user
from quant.db.models import User, UserRole, UserTier
from quant.main import app

pytestmark = pytest.mark.integration


def _stub_user() -> User:
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        hashed_password="!",
        full_name="Test User",
        role=UserRole.viewer,
        tier=UserTier.free,
        is_active=True,
        is_verified=True,
    )


def _override_settings_paper(*, paper: bool = True, has_keys: bool = True) -> object:
    settings.alpaca_paper = paper
    settings.alpaca_api_key_id = SecretStr("k" if has_keys else "")
    settings.alpaca_api_secret_key = SecretStr("s" if has_keys else "")
    return settings


@pytest.fixture()
def client_paper() -> Iterator[TestClient]:
    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_settings] = lambda: _override_settings_paper(paper=True, has_keys=True)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_settings, None)


@pytest.fixture()
def client_no_keys() -> Iterator[TestClient]:
    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_settings] = lambda: _override_settings_paper(paper=True, has_keys=False)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_settings, None)


@pytest.fixture()
def client_live() -> Iterator[TestClient]:
    """ALPACA_PAPER=False — endpoints must refuse."""
    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_settings] = lambda: _override_settings_paper(paper=False, has_keys=True)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_settings, None)


# ------------------------------------------------------------------
# Account endpoint
# ------------------------------------------------------------------
def test_account_returns_snapshot(client_paper: TestClient) -> None:
    fake_payload: dict[str, Any] = {
        "equity": "100000.00",
        "cash": "50000.00",
        "buying_power": "150000.00",
        "status": "ACTIVE",
        "paper": True,
    }

    class _FakeAdapter:
        async def get_json(self, path: str) -> dict[str, Any]:
            assert path == "/v2/account"
            return fake_payload

        async def aclose(self) -> None:
            pass

    with patch("quant.api.v1.paper.AlpacaBrokerAdapter", _FakeAdapter):
        r = client_paper.get("/api/v1/paper/account")

    assert r.status_code == 200
    body = r.json()
    assert body["equity"] == "100000.00"
    assert body["status"] == "ACTIVE"
    assert body["paper"] is True


def test_account_503_when_no_keys(client_no_keys: TestClient) -> None:
    r = client_no_keys.get("/api/v1/paper/account")
    assert r.status_code == 503
    assert "credentials not configured" in r.json()["detail"]


def test_account_403_when_paper_off(client_live: TestClient) -> None:
    r = client_live.get("/api/v1/paper/account")
    assert r.status_code == 403
    assert "ALPACA_PAPER must be true" in r.json()["detail"]


# ------------------------------------------------------------------
# Positions endpoint
# ------------------------------------------------------------------
def test_positions_returns_open_longs(client_paper: TestClient) -> None:
    fake_positions = [
        {
            "symbol": "AAPL",
            "qty": "10",
            "current_price": "200.00",
            "avg_entry_price": "180.00",
            "market_value": "2000.00",
            "unrealized_pl": "200.00",
            "unrealized_plpc": "0.1111",
        },
        # Short position must be filtered out
        {"symbol": "TSLA", "qty": "-5", "current_price": "180.0"},
        # Zero qty must be filtered out
        {"symbol": "MSFT", "qty": "0", "current_price": "400.0"},
    ]

    class _FakeAdapter:
        async def get_json(self, path: str) -> dict[str, Any]:
            return {}

        async def positions(self) -> list[dict[str, Any]]:
            return fake_positions

        async def aclose(self) -> None:
            pass

    with patch("quant.api.v1.paper.AlpacaBrokerAdapter", _FakeAdapter):
        r = client_paper.get("/api/v1/paper/positions")

    assert r.status_code == 200
    body = r.json()
    syms = [p["symbol"] for p in body["positions"]]
    assert syms == ["AAPL"]
    assert body["positions"][0]["unrealized_pl"] == "200.00"
    assert body["total_market_value"] == "2000.00"
    assert body["total_unrealized_pl"] == "200.00"


def test_positions_empty_when_flat(client_paper: TestClient) -> None:
    class _FakeAdapter:
        async def get_json(self, path: str) -> dict[str, Any]:
            return {}

        async def positions(self) -> list[dict[str, Any]]:
            return []

        async def aclose(self) -> None:
            pass

    with patch("quant.api.v1.paper.AlpacaBrokerAdapter", _FakeAdapter):
        r = client_paper.get("/api/v1/paper/positions")

    assert r.status_code == 200
    body = r.json()
    assert body["positions"] == []
    assert body["total_market_value"] == "0"
    assert body["total_unrealized_pl"] == "0"


def test_positions_403_when_paper_off(client_live: TestClient) -> None:
    r = client_live.get("/api/v1/paper/positions")
    assert r.status_code == 403
