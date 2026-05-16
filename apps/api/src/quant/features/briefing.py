"""
Daily briefing writer — LLM generates plain-english summary of top picks.

One LLM call per session. Reads:
    - Top-5 picks (symbol + ML conviction + sanity-decision + reason)
    - Macro regime
    - Account state (equity, positions)
Outputs a 3-4 sentence narrative for the /paper page.

UX-only layer — zero alpha contribution. Makes the dashboard readable
to a non-technical user. Smart-tier task. Live-only.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger("quant.features.briefing")

BRIEFING_SYSTEM = (
    "You are a quant analyst writing a daily 3-4 sentence briefing for the operator "
    "of an ML-driven trading system. Given top picks, macro regime, and account "
    "state, write a clear, specific summary. Reference symbols, conviction levels, "
    "and any FLAG/REJECT decisions. No hype, no promises of returns. Plain, honest, "
    "specific. Return ONLY JSON: "
    '{"narrative": "<3-4 sentences, max 600 chars>", '
    '"headline": "<one short title, max 80 chars>"}. '
    "Do not include any prose outside the JSON."
)


def _format_user(
    *,
    as_of: date,
    top_picks: list[dict[str, Any]],
    regime: dict[str, Any] | None,
    account: dict[str, Any] | None,
) -> str:
    picks_lines: list[str] = []
    for p in top_picks[:5]:
        sym = p.get("symbol", "?")
        score = p.get("score", 0.0)
        decision = p.get("decision", "APPROVE")
        reason = p.get("reason", "")
        picks_lines.append(
            f"- {sym}: conviction={score:+.3f}, sanity={decision}"
            + (f", reason={reason[:100]}" if reason else "")
        )
    picks_block = "\n".join(picks_lines) if picks_lines else "(no picks today)"

    regime_block = "n/a"
    if regime is not None:
        regime_block = (
            f"{regime.get('regime', 'neutral')} "
            f"(confidence={regime.get('confidence', 0.0):.2f}, "
            f"why: {regime.get('rationale', '')[:120]})"
        )

    account_block = "n/a"
    if account is not None:
        account_block = (
            f"equity=${account.get('equity', 0)}, "
            f"cash=${account.get('cash', 0)}, "
            f"open_positions={account.get('n_positions', 0)}"
        )

    return (
        f"Date: {as_of.isoformat()}\n"
        f"Macro regime: {regime_block}\n"
        f"Account: {account_block}\n"
        f"Top picks (post-sanity):\n{picks_block}\n\n"
        "Write the briefing."
    )


async def write_briefing(
    *,
    as_of: date,
    top_picks: list[dict[str, Any]],
    regime: dict[str, Any] | None = None,
    account: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the daily briefing. Returns dict with narrative + headline + metadata."""
    from quant.adapters.openrouter import OpenRouterAdapter

    async with OpenRouterAdapter() as adapter:
        obj, model_used = await adapter.smart_json(
            system=BRIEFING_SYSTEM,
            user=_format_user(as_of=as_of, top_picks=top_picks, regime=regime, account=account),
            temperature=0.4,
            max_tokens=800,
        )

    return {
        "date": as_of.isoformat(),
        "headline": str(obj.get("headline", ""))[:120],
        "narrative": str(obj.get("narrative", ""))[:800],
        "model": model_used,
        "n_picks": len(top_picks),
        "regime_used": (regime or {}).get("regime", "n/a"),
    }


def write_briefing_json(result: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")


__all__ = ["write_briefing", "write_briefing_json"]
