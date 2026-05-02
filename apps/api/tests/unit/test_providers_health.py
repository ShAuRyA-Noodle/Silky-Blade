"""Tests for the provider health-check dispatcher."""

from __future__ import annotations

import pytest

from quant.data.providers_health import (
    ProviderStatus,
    check_all,
    list_provider_names,
)


def test_provider_names_listed() -> None:
    names = list_provider_names()
    # Must cover every adapter that has a settings field.
    expected = {
        "polygon",
        "alpaca_data",
        "alpaca_broker",
        "fred",
        "tiingo",
        "finnhub",
        "groq",
        "marketaux",
        "newsapi",
        "fmp",
        "nasdaq_data_link",
        "alphavantage",
    }
    assert set(names) == expected
    # Stable order — CLI relies on consistent column-width math.
    assert len(names) == len(set(names))


def test_provider_status_dataclass_shape() -> None:
    s = ProviderStatus(name="x", configured=True, ok=False, detail="oops", latency_ms=42)
    assert s.name == "x"
    assert s.configured is True
    assert s.ok is False
    assert s.detail == "oops"
    assert s.latency_ms == 42


@pytest.mark.asyncio
async def test_check_all_skips_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no keys are set, every provider returns configured=False / ok=False."""
    from quant.config import settings as live_settings

    # Clear every API-key field on the live Settings singleton.
    blank_fields = (
        "polygon_api_key",
        "alpaca_api_key_id",
        "alpaca_api_secret_key",
        "fred_api_key",
        "tiingo_api_key",
        "finnhub_api_key",
        "groq_api_key",
        "marketaux_api_key",
        "newsapi_key",
        "fmp_api_key",
        "nasdaq_data_link_api_key",
        "alphavantage_api_key",
    )
    for field in blank_fields:
        monkeypatch.setattr(live_settings, field, "")

    results = await check_all()
    assert len(results) == len(list_provider_names())
    for r in results:
        assert r.configured is False, f"{r.name} should be skipped without key"
        assert r.ok is False
        assert "no key" in r.detail
        assert r.latency_ms is None
