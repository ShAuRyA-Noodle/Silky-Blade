"""
Macro regime classifier — one LLM call per day, classifies market regime.

Reads recent macro/market news headlines + key data points, outputs:
    {regime: "risk_on" | "risk_off" | "neutral", confidence: 0..1, rationale: "..."}

Used as a global conviction multiplier downstream:
    risk_on   → scale ML conviction × 1.0
    neutral   → scale ML conviction × 0.85
    risk_off  → scale ML conviction × 0.50 (defensive)

Smart-tier task. Fallback chain. Live-only (today's news).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger("quant.features.macro_regime")

REGIME_SYSTEM = (
    "You are a macro regime classifier for a quant trading system. You read recent "
    "market headlines, Fed announcements, and economic data, then classify the "
    "current market regime. Return ONLY minified JSON: "
    '{"regime": "risk_on"|"risk_off"|"neutral", '
    '"confidence": <float 0..1>, '
    '"rationale": "<one short sentence, max 200 chars>"}. '
    "risk_on = appetite for equities, falling vol, Fed dovish, positive macro surprises. "
    "risk_off = flight to quality, rising vol, geopolitical stress, Fed hawkish, recession fears. "
    "neutral = mixed signals, range-bound. Do not include any prose outside the JSON."
)


def _build_user_prompt(headlines: list[str]) -> str:
    bullets = "\n".join(f"- {h.strip()[:180]}" for h in headlines[:30] if h.strip())
    return (
        f"Date: {date.today().isoformat()}\n"
        f"Recent macro/market headlines (most recent first):\n{bullets}\n\n"
        "Classify the current market regime."
    )


async def _fetch_macro_headlines(*, limit: int = 25) -> list[str]:
    """Pull recent macro/market headlines from Marketaux + NewsAPI."""
    from quant.adapters.marketaux import MarketauxAdapter
    from quant.adapters.newsapi import NewsApiAdapter

    headlines: list[str] = []

    try:
        async with MarketauxAdapter() as a:
            # Macro fetch — no symbol filter, broad market news.
            # Marketaux free tier doesn't expose topics param via our adapter;
            # we get broad US-market English news and let the LLM filter relevance.
            arts = await a.news(
                symbols=None,
                limit=limit,
                published_after=None,
            )
            for art in arts:
                if isinstance(art, dict):
                    title = str(art.get("title") or "").strip()
                    if title:
                        headlines.append(title)
    except Exception as exc:
        log.warning("marketaux macro fetch failed: %s", exc)

    try:
        async with NewsApiAdapter() as a:
            arts = await a.everything(
                query="federal reserve OR inflation OR recession OR S&P 500 OR market",
                from_iso=None,
                page_size=limit,
            )
            for art in arts:
                if isinstance(art, dict):
                    title = str(art.get("title") or "").strip()
                    if title:
                        headlines.append(title)
    except Exception as exc:
        log.warning("newsapi macro fetch failed: %s", exc)

    return headlines[:limit]


async def classify_regime(*, headlines: list[str] | None = None) -> dict[str, Any]:
    """Run the LLM regime classifier. Returns the JSON result + metadata."""
    from quant.adapters.openrouter import OpenRouterAdapter

    if headlines is None:
        headlines = await _fetch_macro_headlines()

    if not headlines:
        log.warning("macro regime: no headlines available, returning neutral")
        return {
            "date": date.today().isoformat(),
            "regime": "neutral",
            "confidence": 0.0,
            "rationale": "no headlines available — defaulting to neutral",
            "model": "fallback",
            "n_headlines": 0,
        }

    async with OpenRouterAdapter() as adapter:
        obj, model_used = await adapter.smart_json(
            system=REGIME_SYSTEM,
            user=_build_user_prompt(headlines),
            temperature=0.0,
            max_tokens=600,
        )

    regime = obj.get("regime", "neutral")
    if regime not in ("risk_on", "risk_off", "neutral"):
        regime = "neutral"
    confidence = float(obj.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "date": date.today().isoformat(),
        "regime": regime,
        "confidence": confidence,
        "rationale": str(obj.get("rationale", ""))[:300],
        "model": model_used,
        "n_headlines": len(headlines),
    }


def regime_conviction_multiplier(regime: str) -> float:
    """How much to scale ML conviction by, given the regime."""
    return {"risk_on": 1.0, "neutral": 0.85, "risk_off": 0.5}.get(regime, 0.85)


def write_regime_json(result: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")


__all__ = ["classify_regime", "regime_conviction_multiplier", "write_regime_json"]
