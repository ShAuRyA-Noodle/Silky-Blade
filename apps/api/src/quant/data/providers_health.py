"""
Provider health-check — ping every keyed data provider, report PASS/FAIL.

Catches misconfigured / expired / rate-limited API keys in seconds, before
a 30-minute backfill discovers it the hard way. Each check is the smallest
possible authenticated request that the provider will accept — usually a
single-symbol single-day bar fetch or an account-info endpoint.

Public surface: `check_all(settings)` returns a list of `ProviderStatus`.
The CLI prints them; tests construct settings with stubbed keys and assert
the dispatch table.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from quant.config import settings as live_settings
from quant.config.settings import Settings

log = logging.getLogger("quant.data.providers_health")


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    configured: bool
    ok: bool
    detail: str  # short human-readable message
    latency_ms: int | None = None


# ------------------------------------------------------------------
# Per-provider checks. Each returns a ProviderStatus.
# ------------------------------------------------------------------
async def _check_polygon(s: Settings) -> ProviderStatus:
    if not s.polygon_api_key:
        return ProviderStatus("polygon", False, False, "no key set")
    try:
        from quant.adapters.polygon import PolygonAdapter

        async with PolygonAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.market_status()
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        market = str(res.get("market", "?")) if isinstance(res, dict) else "?"
        return ProviderStatus("polygon", True, True, f"market={market}", ms)
    except Exception as exc:
        return ProviderStatus("polygon", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_alpaca_data(s: Settings) -> ProviderStatus:
    if not s.alpaca_api_key_id or not s.alpaca_api_secret_key:
        return ProviderStatus("alpaca_data", False, False, "no key set")
    try:
        from quant.adapters.alpaca import AlpacaDataAdapter

        async with AlpacaDataAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            now = datetime.now(UTC)
            res = await a.bars(
                ["AAPL"],
                timeframe="1Day",
                start=now - timedelta(days=3),
                end=now,
            )
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        n = len(res.get("AAPL", [])) if isinstance(res, dict) else 0
        return ProviderStatus("alpaca_data", True, True, f"AAPL bars in last 3d: {n}", ms)
    except Exception as exc:
        return ProviderStatus("alpaca_data", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_alpaca_broker(s: Settings) -> ProviderStatus:
    if not s.alpaca_api_key_id or not s.alpaca_api_secret_key:
        return ProviderStatus("alpaca_broker", False, False, "no key set")
    try:
        from quant.adapters.alpaca import AlpacaBrokerAdapter

        async with AlpacaBrokerAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.get_json("/v2/account")
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        status = str(res.get("status", "?")) if isinstance(res, dict) else "?"
        equity = str(res.get("equity", "?"))[:12] if isinstance(res, dict) else "?"
        paper = " (paper)" if s.alpaca_paper else " (LIVE)"
        return ProviderStatus(
            "alpaca_broker",
            True,
            True,
            f"status={status} equity={equity}{paper}",
            ms,
        )
    except Exception as exc:
        return ProviderStatus("alpaca_broker", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_fred(s: Settings) -> ProviderStatus:
    if not s.fred_api_key:
        return ProviderStatus("fred", False, False, "no key set")
    try:
        from quant.adapters.fred import FredAdapter

        async with FredAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            # DGS10 = 10-year treasury constant maturity rate. Tiny payload.
            res = await a.observations("DGS10", limit=1)
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        n = len(res) if isinstance(res, list) else 0
        return ProviderStatus("fred", True, True, f"DGS10 latest obs: {n}", ms)
    except Exception as exc:
        return ProviderStatus("fred", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_tiingo(s: Settings) -> ProviderStatus:
    if not s.tiingo_api_key:
        return ProviderStatus("tiingo", False, False, "no key set")
    try:
        from quant.adapters.tiingo import TiingoAdapter

        async with TiingoAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.daily_prices(
                "AAPL",
                start=date.today() - timedelta(days=2),
                end=date.today(),
            )
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        n = len(res) if isinstance(res, list) else 0
        return ProviderStatus("tiingo", True, True, f"AAPL bars in last 2d: {n}", ms)
    except Exception as exc:
        return ProviderStatus("tiingo", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_finnhub(s: Settings) -> ProviderStatus:
    if not s.finnhub_api_key:
        return ProviderStatus("finnhub", False, False, "no key set")
    try:
        from quant.adapters.finnhub import FinnhubAdapter

        async with FinnhubAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.quote("AAPL")
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        c = res.get("c", "?") if isinstance(res, dict) else "?"
        return ProviderStatus("finnhub", True, True, f"AAPL quote.c={c}", ms)
    except Exception as exc:
        return ProviderStatus("finnhub", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_groq(s: Settings) -> ProviderStatus:
    if not s.groq_api_key:
        return ProviderStatus("groq", False, False, "no key set")
    try:
        from quant.adapters.groq import GroqAdapter

        async with GroqAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.score_sentiment(
                headline="Apple reports steady quarterly revenue.",
                summary=None,
                tickers=["AAPL"],
            )
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        ok = isinstance(res, dict) and "score" in res
        return ProviderStatus(
            "groq",
            True,
            ok,
            f"sentiment_score={res.get('score', '?')}" if ok else "unexpected shape",
            ms,
        )
    except Exception as exc:
        return ProviderStatus("groq", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_marketaux(s: Settings) -> ProviderStatus:
    if not s.marketaux_api_key:
        return ProviderStatus("marketaux", False, False, "no key set")
    try:
        from quant.adapters.marketaux import MarketauxAdapter

        async with MarketauxAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.news(symbols=["AAPL"], limit=1)
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        n = len(res) if isinstance(res, list) else 0
        return ProviderStatus("marketaux", True, True, f"news count: {n}", ms)
    except Exception as exc:
        return ProviderStatus("marketaux", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_newsapi(s: Settings) -> ProviderStatus:
    if not s.newsapi_key:
        return ProviderStatus("newsapi", False, False, "no key set")
    try:
        from quant.adapters.newsapi import NewsApiAdapter

        async with NewsApiAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.everything(query="Apple", page_size=1)
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        n = len(res) if isinstance(res, list) else 0
        return ProviderStatus("newsapi", True, True, f"articles: {n}", ms)
    except Exception as exc:
        return ProviderStatus("newsapi", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_fmp(s: Settings) -> ProviderStatus:
    if not s.fmp_api_key:
        return ProviderStatus("fmp", False, False, "no key set")
    try:
        from quant.adapters.fmp import FmpAdapter

        async with FmpAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            # /stable/quote/AAPL is on FMP's current free tier (post-2025
            # endpoint reorganization); the older /api/v3/key-metrics path
            # is paid-only.
            res = await a.get_json("/stable/quote", params={"symbol": "AAPL"})
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        ok = isinstance(res, list) and len(res) > 0
        sym = res[0].get("symbol", "?") if ok else "?"
        return ProviderStatus(
            "fmp",
            True,
            ok,
            f"quote.symbol={sym}" if ok else "unexpected shape",
            ms,
        )
    except Exception as exc:
        return ProviderStatus("fmp", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_nasdaq_data_link(s: Settings) -> ProviderStatus:
    if not s.nasdaq_data_link_api_key:
        return ProviderStatus("nasdaq_data_link", False, False, "no key set")
    try:
        from quant.adapters.nasdaq_data_link import NasdaqDataLinkAdapter

        async with NasdaqDataLinkAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            # FRED/GDP is a free, always-available dataset; we only need
            # one row to confirm the API key authenticates.
            res = await a.dataset(database_code="FRED", dataset_code="GDP", limit=1)
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        ok = isinstance(res, dict) and "dataset_data" in res
        return ProviderStatus(
            "nasdaq_data_link",
            True,
            ok,
            "dataset_data ok" if ok else "unexpected shape",
            ms,
        )
    except Exception as exc:
        return ProviderStatus("nasdaq_data_link", True, False, f"{exc.__class__.__name__}: {exc}")


async def _check_alphavantage(s: Settings) -> ProviderStatus:
    if not s.alphavantage_api_key:
        return ProviderStatus("alphavantage", False, False, "no key set")
    try:
        from quant.adapters.alphavantage import AlphaVantageAdapter

        async with AlphaVantageAdapter() as a:
            t0 = asyncio.get_event_loop().time()
            res = await a.daily_adjusted("AAPL", full=False)
            ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        ok = isinstance(res, dict) and "Time Series (Daily)" in res and len(res["Time Series (Daily)"]) > 0
        return ProviderStatus(
            "alphavantage",
            True,
            ok,
            "daily_adjusted ok" if ok else "rate-limited or empty",
            ms,
        )
    except Exception as exc:
        return ProviderStatus("alphavantage", True, False, f"{exc.__class__.__name__}: {exc}")


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------
_CHECKS: tuple[tuple[str, Any], ...] = (
    ("polygon", _check_polygon),
    ("alpaca_data", _check_alpaca_data),
    ("alpaca_broker", _check_alpaca_broker),
    ("fred", _check_fred),
    ("tiingo", _check_tiingo),
    ("finnhub", _check_finnhub),
    ("groq", _check_groq),
    ("marketaux", _check_marketaux),
    ("newsapi", _check_newsapi),
    ("fmp", _check_fmp),
    ("nasdaq_data_link", _check_nasdaq_data_link),
    ("alphavantage", _check_alphavantage),
)


async def check_all(s: Settings | None = None) -> list[ProviderStatus]:
    """Run every check concurrently. Each check is independent + safe to gather."""
    cfg = s if s is not None else live_settings
    results = await asyncio.gather(
        *(check(cfg) for _, check in _CHECKS),
        return_exceptions=False,
    )
    return list(results)


def list_provider_names() -> list[str]:
    """Helper for tests + CLI completion."""
    return [name for name, _ in _CHECKS]


__all__ = [
    "ProviderStatus",
    "check_all",
    "list_provider_names",
]
