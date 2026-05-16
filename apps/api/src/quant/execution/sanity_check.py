"""
Pre-trade sanity checker — LLM reviews ML's top-K candidates for red flags.

Workflow:
    ML model ranks 25 candidates → this module reads recent news per candidate →
    LLM outputs APPROVE / FLAG / REJECT per name → top-5 filtered from
    non-REJECT'd names.

This is the highest-value LLM layer. It catches catastrophic news the ML
model couldn't see (bankruptcy filing, fraud, recall, CEO scandal). Acts as
a brake, never as a primary signal.

Smart-tier task. Fallback chain. Live-only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

log = logging.getLogger("quant.execution.sanity_check")

SANITY_SYSTEM = (
    "You are a pre-trade risk reviewer for an ML-driven trading system. Given a "
    "ticker, its current ML conviction score, and recent news headlines, decide: "
    "should this trade go through? Return ONLY minified JSON: "
    '{"decision": "APPROVE"|"FLAG"|"REJECT", '
    '"reason": "<one short sentence, max 200 chars>"}. '
    "APPROVE = no red flags. FLAG = caution, proceed at reduced size (recent bad "
    "news but not catastrophic). REJECT = do not trade (bankruptcy filing, fraud "
    "investigation, major recall, CEO scandal, going-concern doubts, hostile "
    "takeover collapse, etc.). Be conservative — when in doubt, FLAG. "
    "Do not include any prose outside the JSON."
)

Decision = Literal["APPROVE", "FLAG", "REJECT"]


@dataclass(frozen=True)
class SanityResult:
    symbol: str
    decision: Decision
    reason: str
    model: str
    score: float  # original ML conviction
    n_headlines: int


def _format_user(symbol: str, score: float, headlines: list[str]) -> str:
    bullets = "\n".join(f"- {h.strip()[:200]}" for h in headlines[:8] if h.strip())
    if not bullets:
        bullets = "(no recent news in last 5 days)"
    return (
        f"Ticker: {symbol}\n"
        f"ML conviction score: {score:+.3f}\n"
        f"Recent news (last 5 days):\n{bullets}\n\n"
        "Review and decide."
    )


async def _fetch_recent_headlines_for(symbol: str, *, days: int = 5, limit: int = 8) -> list[str]:
    """Pull recent news headlines for one symbol."""
    from datetime import timedelta

    from quant.adapters.marketaux import MarketauxAdapter
    from quant.adapters.newsapi import NewsApiAdapter

    cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    headlines: list[str] = []

    try:
        async with MarketauxAdapter() as a:
            arts = await a.news(symbols=[symbol], limit=limit // 2, published_after=cutoff)
            for art in arts:
                if isinstance(art, dict):
                    t = str(art.get("title") or "").strip()
                    if t:
                        headlines.append(t)
    except Exception as exc:
        log.debug("marketaux sanity %s: %s", symbol, exc)

    try:
        from datetime import timedelta as _td

        from_iso = (datetime.now(UTC) - _td(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with NewsApiAdapter() as a:
            arts = await a.everything(query=symbol, from_iso=from_iso, page_size=limit // 2)
            for art in arts:
                if isinstance(art, dict):
                    t = str(art.get("title") or "").strip()
                    if t:
                        headlines.append(t)
    except Exception as exc:
        log.debug("newsapi sanity %s: %s", symbol, exc)

    return headlines[:limit]


async def _review_one(
    adapter: Any,
    *,
    symbol: str,
    score: float,
    sem: asyncio.Semaphore,
    days: int = 5,
) -> SanityResult:
    headlines = await _fetch_recent_headlines_for(symbol, days=days)
    async with sem:
        try:
            obj, model_used = await adapter.smart_json(
                system=SANITY_SYSTEM,
                user=_format_user(symbol, score, headlines),
                temperature=0.0,
                max_tokens=600,
            )
        except Exception as exc:
            log.warning("sanity review %s failed: %s — defaulting to APPROVE", symbol, exc)
            return SanityResult(
                symbol=symbol,
                decision="APPROVE",
                reason="LLM unreachable — defaulting to approve",
                model="fallback",
                score=score,
                n_headlines=len(headlines),
            )

    dec = obj.get("decision", "APPROVE")
    if dec not in ("APPROVE", "FLAG", "REJECT"):
        dec = "APPROVE"
    return SanityResult(
        symbol=symbol,
        decision=dec,
        reason=str(obj.get("reason", ""))[:300],
        model=model_used,
        score=score,
        n_headlines=len(headlines),
    )


async def review_candidates(
    candidates: Iterable[tuple[str, float]],
    *,
    max_concurrent: int = 3,
    days: int = 5,
) -> list[SanityResult]:
    """Review (symbol, score) tuples. Returns SanityResult per candidate."""
    from quant.adapters.openrouter import OpenRouterAdapter

    cands = list(candidates)
    if not cands:
        return []

    sem = asyncio.Semaphore(max_concurrent)
    async with OpenRouterAdapter() as adapter:
        results = await asyncio.gather(
            *(_review_one(adapter, symbol=s, score=sc, sem=sem, days=days) for s, sc in cands)
        )
    log.info(
        "sanity check: %d reviewed (%d APPROVE / %d FLAG / %d REJECT)",
        len(results),
        sum(1 for r in results if r.decision == "APPROVE"),
        sum(1 for r in results if r.decision == "FLAG"),
        sum(1 for r in results if r.decision == "REJECT"),
    )
    return list(results)


def filter_by_sanity(
    candidates: list[tuple[str, float]],
    results: list[SanityResult],
    *,
    flag_score_haircut: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Apply sanity decisions to candidate list.
    - REJECT: removed
    - FLAG: kept but conviction × flag_score_haircut
    - APPROVE: unchanged
    Returns filtered (symbol, adjusted_score) list, re-sorted by score.
    """
    decisions = {r.symbol: r.decision for r in results}
    out: list[tuple[str, float]] = []
    for sym, score in candidates:
        d = decisions.get(sym, "APPROVE")
        if d == "REJECT":
            continue
        adj = score * flag_score_haircut if d == "FLAG" else score
        out.append((sym, adj))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


__all__ = [
    "Decision",
    "SanityResult",
    "filter_by_sanity",
    "review_candidates",
]
