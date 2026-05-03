"""
Order reconciliation — poll Alpaca for the terminal status of submitted
orders so the session can report fills, partial fills, and rejections.

When `submit_orders()` returns, each order has been *accepted* by the
broker but is not yet *filled*. Equity stocks on Alpaca usually fill
within seconds during market hours, but partial fills, broker rejections,
exchange holds, and post-acceptance cancellations are all real outcomes
the caller has to surface to a human reviewer.

This module is a thin polling loop. It is sequential per order to stay
inside Alpaca's free-tier rate limit (200 req/min); for our top-10
sessions that's 10 GETs in 2-3 seconds.

Terminal Alpaca order statuses (from /v2/orders/{id}):
    "filled"             — entirely filled
    "partially_filled"   — partial fill; remainder canceled
    "canceled"           — broker / venue / user canceled
    "expired"            — DAY order didn't fill, end of session
    "rejected"           — broker refused
    "done_for_day"       — venue closed before fill
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from quant.execution.broker import BrokerOrderAck

log = logging.getLogger("quant.execution.reconciliation")


_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "filled",
        "partially_filled",
        "canceled",
        "expired",
        "rejected",
        "done_for_day",
    }
)


@dataclass(frozen=True)
class FillReport:
    """Final state of one submitted order after polling."""

    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str
    submitted_qty: Decimal
    filled_qty: Decimal
    avg_fill_price: Decimal
    status: str  # final terminal status
    polls: int  # number of GETs to reach terminal


async def poll_until_terminal(
    broker_adapter: Any,
    ack: BrokerOrderAck,
    *,
    max_polls: int = 30,
    interval_seconds: float = 1.0,
) -> FillReport:
    """
    Poll one order until its status is terminal or `max_polls` is hit.

    Returns a FillReport even if we time out before terminal — the caller
    can detect a non-terminal final status (e.g. "new", "accepted") and
    decide whether to alert / re-poll later. We never raise on a
    non-terminal end state; that's a real production case, not an error.
    """
    last_payload: dict[str, Any] = {}
    polls = 0
    for _ in range(max_polls):
        polls += 1
        try:
            payload = await broker_adapter.get_json(f"/v2/orders/{ack.broker_order_id}")
        except Exception as exc:
            log.warning("poll failed for %s: %s", ack.broker_order_id, exc)
            await asyncio.sleep(interval_seconds)
            continue
        if not isinstance(payload, dict):
            log.warning("unexpected order payload type: %s", type(payload).__name__)
            await asyncio.sleep(interval_seconds)
            continue
        last_payload = payload
        status = str(payload.get("status", "")).lower()
        if status in _TERMINAL_STATUSES:
            break
        await asyncio.sleep(interval_seconds)

    return FillReport(
        broker_order_id=ack.broker_order_id,
        client_order_id=ack.client_order_id,
        symbol=str(last_payload.get("symbol", "")),
        side=str(last_payload.get("side", "")).upper(),
        submitted_qty=Decimal(str(last_payload.get("qty", "0"))),
        filled_qty=Decimal(str(last_payload.get("filled_qty", "0"))),
        avg_fill_price=Decimal(str(last_payload.get("filled_avg_price") or "0")),
        status=str(last_payload.get("status", "unknown")).lower(),
        polls=polls,
    )


async def reconcile(
    broker_adapter: Any,
    acks: list[BrokerOrderAck],
    *,
    max_polls: int = 30,
    interval_seconds: float = 1.0,
) -> list[FillReport]:
    """
    Poll every ack to terminal sequentially. Sequential is intentional —
    the rebalance volume is low and rate-limit headroom matters more than
    parallelism; concurrent polling here would buy nothing but tail risk.
    """
    out: list[FillReport] = []
    for ack in acks:
        report = await poll_until_terminal(
            broker_adapter,
            ack,
            max_polls=max_polls,
            interval_seconds=interval_seconds,
        )
        out.append(report)
        log.info(
            "  reconciled %s %s: status=%s filled=%s/%s @ avg=$%s polls=%d",
            report.side,
            report.symbol,
            report.status,
            report.filled_qty,
            report.submitted_qty,
            report.avg_fill_price,
            report.polls,
        )
    return out


def is_terminal(status: str) -> bool:
    return status.lower() in _TERMINAL_STATUSES


__all__ = [
    "FillReport",
    "is_terminal",
    "poll_until_terminal",
    "reconcile",
]
