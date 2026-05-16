"""
Catalyst tagger — LLM extracts structured event tags from news headlines.

Output: catalysts.csv with columns [symbol, date, catalyst_type, severity, summary]

Catalyst types (closed vocabulary — model must pick from this list):
    earnings_beat       earnings_miss       earnings_inline
    guidance_raise      guidance_cut        guidance_inline
    fda_approval        fda_rejection       clinical_data
    upgrade             downgrade           price_target_change
    merger              acquisition         spinoff
    scandal             lawsuit             investigation
    dividend_change     stock_split         buyback_announce
    ceo_change          layoffs             restructuring
    product_launch      product_recall
    none                                                       (catch-all)

Severity: low | medium | high

Smart-tier task — runs through fallback chain (K2.5 → Flash → Pro → K2.6).
Pass `live_only=True` on the runtime; this module produces snapshot data and
must NEVER be replayed into a historical backtest.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("quant.features.catalysts")

CATALYST_SYSTEM = (
    "You are a financial catalyst tagger. You read a news headline and extract a "
    "structured event tag. Return ONLY minified JSON of the form: "
    '{"catalyst_type": "<one of the allowed types>", '
    '"severity": "low|medium|high", '
    '"summary": "<one short sentence, max 100 chars>"}. '
    "Allowed catalyst types: earnings_beat, earnings_miss, earnings_inline, "
    "guidance_raise, guidance_cut, guidance_inline, fda_approval, fda_rejection, "
    "clinical_data, upgrade, downgrade, price_target_change, merger, acquisition, "
    "spinoff, scandal, lawsuit, investigation, dividend_change, stock_split, "
    "buyback_announce, ceo_change, layoffs, restructuring, product_launch, "
    "product_recall, none. "
    "If no clear catalyst, return type=none, severity=low, summary='no specific catalyst'. "
    "Do not include any prose outside the JSON."
)

_VALID_TYPES = frozenset(
    {
        "earnings_beat",
        "earnings_miss",
        "earnings_inline",
        "guidance_raise",
        "guidance_cut",
        "guidance_inline",
        "fda_approval",
        "fda_rejection",
        "clinical_data",
        "upgrade",
        "downgrade",
        "price_target_change",
        "merger",
        "acquisition",
        "spinoff",
        "scandal",
        "lawsuit",
        "investigation",
        "dividend_change",
        "stock_split",
        "buyback_announce",
        "ceo_change",
        "layoffs",
        "restructuring",
        "product_launch",
        "product_recall",
        "none",
    }
)


async def _tag_article(
    adapter: Any,
    *,
    symbol: str,
    headline: str,
    summary: str | None,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """
    Cost-optimized: catalyst tagging is a closed-vocabulary structured-output
    task, NOT reasoning. Pin to fast tier (DeepSeek V4 Flash) — saves 5x vs K2.5
    on the same accuracy. Smart chain only fires on Flash failure.
    """
    from quant.config import settings

    user = f"Ticker: {symbol}\nHeadline: {headline}\nSummary: {summary or '(none)'}"
    async with sem:
        # Try fast model first (V4 Flash) — bulk task, deterministic JSON output
        try:
            raw = await adapter._chat(
                model=settings.openrouter_model_fast,
                system=CATALYST_SYSTEM,
                user=user,
                temperature=0.0,
                max_tokens=200,  # Flash doesn't burn reasoning tokens
                response_format={"type": "json_object"},
            )
            import json as _json

            try:
                obj = _json.loads(raw)
            except _json.JSONDecodeError:
                import re as _re

                m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                if not m:
                    raise
                obj = _json.loads(m.group(0))
            model_used = settings.openrouter_model_fast
        except Exception as fast_exc:
            log.debug("catalyst Flash failed %s: %s — retrying with smart chain", symbol, fast_exc)
            try:
                obj, model_used = await adapter.smart_json(
                    system=CATALYST_SYSTEM,
                    user=user,
                    temperature=0.0,
                    max_tokens=800,
                )
            except Exception as exc:
                log.warning("catalyst tag %s failed: %s", symbol, exc)
                return None

    ctype = obj.get("catalyst_type", "none")
    if ctype not in _VALID_TYPES:
        ctype = "none"
    severity = obj.get("severity", "low")
    if severity not in ("low", "medium", "high"):
        severity = "low"
    return {
        "symbol": symbol,
        "catalyst_type": ctype,
        "severity": severity,
        "summary": str(obj.get("summary", ""))[:120],
        "model": model_used,
    }


async def tag_articles(
    articles: list[dict[str, Any]],
    *,
    max_concurrent: int = 4,
) -> list[dict[str, Any]]:
    """Tag a list of news articles. Articles must have `__symbol__`, `title`, `description`."""
    from quant.adapters.openrouter import OpenRouterAdapter

    sem = asyncio.Semaphore(max_concurrent)
    results: list[dict[str, Any]] = []

    async with OpenRouterAdapter() as adapter:

        async def _one(art: dict[str, Any]) -> None:
            sym = str(art.get("__symbol__", "")).strip()
            headline = str(art.get("title") or "")
            summary = art.get("description") or art.get("snippet") or art.get("content")
            pub = art.get("published_at") or art.get("publishedAt") or ""
            if not sym or not headline:
                return
            try:
                pub_date = datetime.fromisoformat(str(pub).replace("Z", "+00:00")).date()
            except (ValueError, TypeError):
                pub_date = datetime.now(UTC).date()
            tag = await _tag_article(
                adapter,
                symbol=sym,
                headline=headline,
                summary=summary if isinstance(summary, str) else None,
                sem=sem,
            )
            if tag is None:
                return
            tag["date"] = pub_date.isoformat()
            results.append(tag)

        await asyncio.gather(*(_one(a) for a in articles))

    return results


def write_catalysts_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol", "date", "catalyst_type", "severity", "summary", "model"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


async def fetch_and_tag(
    symbols: Iterable[str],
    *,
    days: int = 2,
    per_symbol_limit: int = 3,
) -> list[dict[str, Any]]:
    """Fetch news → tag catalysts → return per-symbol rows. Live-only."""
    from quant.features.sentiment import (
        _fetch_marketaux_for_symbols,
        _fetch_newsapi_for_symbols,
    )

    syms = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    if not syms:
        return []

    articles: list[dict[str, Any]] = []
    articles.extend(await _fetch_marketaux_for_symbols(syms, days=days, per_call_limit=per_symbol_limit))
    articles.extend(await _fetch_newsapi_for_symbols(syms, days=days, per_call_limit=per_symbol_limit))
    log.info("catalyst tagger: %d articles across %d symbols", len(articles), len(syms))
    if not articles:
        return []

    rows = await tag_articles(articles)
    log.info("catalyst tagger: tagged %d articles", len(rows))
    return rows


__all__ = ["fetch_and_tag", "tag_articles", "write_catalysts_csv"]
