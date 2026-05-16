"""
Sentiment feature pipeline — fetch news → score via Groq → aggregate per
(symbol, date) → write a flat CSV the live trade path can consume.

Data flow:

    Marketaux news API  ─┐
                         ├─►  Groq score_sentiment  ─►  per-article {score, label}
    NewsAPI /everything ─┘

    Aggregate by (symbol, date):
        sentiment_mean  = average of per-article scores
        sentiment_count = number of articles seen for that symbol-day
        sentiment_max   = strongest single-article score (for spikes)

The output CSV is the contract; downstream `SentimentSignal` reads it.
This module never blocks the live trade path waiting on news — the cron
job runs sentiment fetch in advance and writes the CSV; the signal just
reads.

Honest scope:
    Free-tier news APIs cover roughly the last 7-30 days. Backfilling
    historical sentiment for the 2018-2026 panel needs a paid archive
    (Marketaux Pro, NewsAPI Premium, or Bloomberg). For LIVE PAPER
    TRADING the live window is what matters; for BACKTEST RESEARCH
    the technical-only model stays the apples-to-apples baseline.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("quant.features.sentiment")


@dataclass(frozen=True)
class _ScoredArticle:
    symbol: str
    published_date: date
    score: float
    label: str
    headline: str


# ------------------------------------------------------------------
# News fetchers
# ------------------------------------------------------------------
async def _fetch_marketaux_for_symbols(
    symbols: list[str],
    *,
    days: int,
    per_call_limit: int = 3,
) -> list[dict[str, Any]]:
    """One call per symbol — Marketaux free tier filters by symbol."""
    from quant.adapters.marketaux import MarketauxAdapter

    out: list[dict[str, Any]] = []
    cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    async with MarketauxAdapter() as a:
        for sym in symbols:
            try:
                articles = await a.news(
                    symbols=[sym],
                    limit=per_call_limit,
                    published_after=cutoff,
                )
            except Exception as exc:
                log.warning("marketaux fetch %s failed: %s", sym, exc)
                continue
            for art in articles:
                if isinstance(art, dict):
                    art["__symbol__"] = sym
                    art["__source__"] = "marketaux"
                    out.append(art)
    return out


async def _fetch_newsapi_for_symbols(
    symbols: list[str],
    *,
    days: int,
    per_call_limit: int = 5,
) -> list[dict[str, Any]]:
    """NewsAPI search by symbol-as-query; rate-limited 100/day on free."""
    from quant.adapters.newsapi import NewsApiAdapter

    out: list[dict[str, Any]] = []
    from_iso = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    async with NewsApiAdapter() as a:
        for sym in symbols:
            try:
                articles = await a.everything(
                    query=sym,
                    from_iso=from_iso,
                    page_size=per_call_limit,
                )
            except Exception as exc:
                log.warning("newsapi fetch %s failed: %s", sym, exc)
                continue
            for art in articles:
                if isinstance(art, dict):
                    art["__symbol__"] = sym
                    art["__source__"] = "newsapi"
                    out.append(art)
    return out


def _article_text(art: dict[str, Any]) -> tuple[str, str | None, date | None]:
    """Extract (headline, summary, date) from either provider's payload."""
    src = art.get("__source__", "")
    if src == "marketaux":
        title = str(art.get("title") or "")
        summary = art.get("description") or art.get("snippet") or None
        published = art.get("published_at") or ""
    else:  # newsapi
        title = str(art.get("title") or "")
        summary = art.get("description") or art.get("content") or None
        published = art.get("publishedAt") or ""
    pub_date: date | None = None
    if published:
        try:
            pub_date = datetime.fromisoformat(str(published).replace("Z", "+00:00")).date()
        except ValueError:
            pub_date = None
    return title, summary, pub_date


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
async def _score_articles(
    articles: list[dict[str, Any]],
    *,
    sleep_seconds: float = 0.02,
    max_concurrent: int = 6,
    provider: str = "openrouter",
) -> list[_ScoredArticle]:
    """
    Score every article via the configured LLM provider.

    Default provider: openrouter (DeepSeek V4 Flash — $0.112/$0.224 per M tokens,
    JSON-strict, ~$2 for 3-month SP500 sentiment run).

    Fallback provider: groq (llama-3.1-8b-instant, free tier 30 req/min).
    """
    sem = asyncio.Semaphore(max_concurrent)
    results: list[_ScoredArticle] = []

    if provider == "openrouter":
        from quant.adapters.openrouter import OpenRouterAdapter

        adapter_cls: type = OpenRouterAdapter
    else:
        from quant.adapters.groq import GroqAdapter

        adapter_cls = GroqAdapter

    async with adapter_cls() as scorer:

        async def _one(art: dict[str, Any]) -> None:
            sym = str(art.get("__symbol__", ""))
            title, summary, pub_date = _article_text(art)
            if not title or pub_date is None:
                return
            async with sem:
                try:
                    res = await scorer.score_sentiment(headline=title, summary=summary, tickers=[sym])
                except Exception as exc:
                    log.warning("%s score %s failed: %s", provider, sym, exc)
                    return
                await asyncio.sleep(sleep_seconds)
            score = res.get("score") if isinstance(res, dict) else None
            label = res.get("label") if isinstance(res, dict) else None
            if not isinstance(score, int | float) or not isinstance(label, str):
                return
            results.append(
                _ScoredArticle(
                    symbol=sym,
                    published_date=pub_date,
                    score=float(score),
                    label=label,
                    headline=title[:120],
                )
            )

        await asyncio.gather(*(_one(a) for a in articles))

    return results


# Backward-compat alias — old callers use _score_with_groq
_score_with_groq = _score_articles


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def _aggregate_per_symbol_day(
    scored: list[_ScoredArticle],
) -> list[dict[str, Any]]:
    """Group by (symbol, published_date) → mean / count / max(|score|)."""
    keyed: dict[tuple[str, date], list[_ScoredArticle]] = {}
    for s in scored:
        keyed.setdefault((s.symbol, s.published_date), []).append(s)
    out: list[dict[str, Any]] = []
    for (sym, d), group in sorted(keyed.items()):
        scores = [g.score for g in group]
        mean = sum(scores) / len(scores)
        signed_max = max(scores, key=abs)
        out.append(
            {
                "symbol": sym,
                "date": d.isoformat(),
                "sentiment_mean": round(mean, 4),
                "sentiment_count": len(scores),
                "sentiment_max_abs": round(signed_max, 4),
            }
        )
    return out


# ------------------------------------------------------------------
# Top-level orchestration
# ------------------------------------------------------------------
async def fetch_and_score(
    symbols: Iterable[str],
    *,
    days: int = 7,
    use_marketaux: bool = True,
    use_newsapi: bool = True,
    provider: str = "openrouter",
) -> list[dict[str, Any]]:
    """Pull news from configured sources, score each article via the LLM provider,
    aggregate per (symbol, date). Returns rows ready for CSV.

    provider: 'openrouter' (default, DeepSeek V4 Flash) or 'groq' (fallback).
    """
    from quant.config import settings

    syms = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    if not syms:
        return []

    articles: list[dict[str, Any]] = []
    if use_marketaux:
        articles.extend(await _fetch_marketaux_for_symbols(syms, days=days))
    if use_newsapi:
        articles.extend(await _fetch_newsapi_for_symbols(syms, days=days))

    log.info("fetched %d articles across %d symbols", len(articles), len(syms))
    if not articles:
        return []

    # Auto-fallback: if OpenRouter key not configured, use Groq.
    if provider == "openrouter" and settings.openrouter_api_key is None:
        log.warning("openrouter_api_key not set — falling back to groq")
        provider = "groq"

    scored = await _score_articles(articles, provider=provider)
    log.info("scored %d articles via %s", len(scored), provider)

    return _aggregate_per_symbol_day(scored)


def write_sentiment_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol", "date", "sentiment_mean", "sentiment_count", "sentiment_max_abs"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


__all__ = [
    "fetch_and_score",
    "write_sentiment_csv",
]
