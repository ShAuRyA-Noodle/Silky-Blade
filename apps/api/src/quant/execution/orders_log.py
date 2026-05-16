"""
Orders audit trail — CSV-as-DB.

Every order successfully submitted via run_live_session() gets appended
to orders-log.csv. This is the persistent audit trail for paper trading
when running in a no-DB context (GitHub Actions cron, local CLI).

Schema (CSV columns):
    session_id           idempotent ID for the rebalance session
    as_of                trading-day date the signal was computed for
    submitted_at_utc     wall-clock UTC timestamp of the broker.submit() call
    broker_order_id      broker-assigned ID (Alpaca order UUID)
    client_order_id      sha256-derived deterministic ID
    symbol               ticker
    side                 BUY | SELL
    quantity             share count (decimal)
    status               broker-reported status (accepted | filled | rejected)

Append-only. Idempotent: the GitHub Actions cron commit step diffs the
file so no-op runs are not committed. If two runs same day produce same
client_order_id, they'd dedupe at broker level (see session_id idempotency
fix in live_session.py).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("quant.execution.orders_log")

_FIELDS = [
    "session_id",
    "as_of",
    "submitted_at_utc",
    "broker_order_id",
    "client_order_id",
    "symbol",
    "side",
    "quantity",
    "status",
]


def append_orders_log(
    *,
    path: str | Path,
    session_id: str,
    as_of: date,
    acks: Iterable[Any],
    proposals: Iterable[Any],
) -> int:
    """
    Append successful submissions to orders-log.csv. Returns row count written.

    Joins `acks` to `proposals` by index — both lists are produced in the same
    submission order by paper_session.submit_orders (SELLs first, then BUYs).
    Skips when no acks (plan-only runs or empty proposal set).
    """
    ack_list = list(acks)
    prop_list = list(proposals)
    if not ack_list:
        return 0

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0
    now_utc = datetime.now(UTC).isoformat()

    # Build symbol/side/qty lookup from proposals by client_order_id stem.
    # client_order_id is sha256(session_id|symbol|side|qty)[:32] — we can't
    # cheaply reverse that, so we match positionally on the SELL-first sort.
    proposals_sorted = sorted(prop_list, key=lambda x: 0 if x.side.upper() == "SELL" else 1)

    rows_written = 0
    with p.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        if not exists:
            w.writeheader()
        for i, ack in enumerate(ack_list):
            prop = proposals_sorted[i] if i < len(proposals_sorted) else None
            row = {
                "session_id": session_id,
                "as_of": as_of.isoformat(),
                "submitted_at_utc": now_utc,
                "broker_order_id": getattr(ack, "broker_order_id", ""),
                "client_order_id": getattr(ack, "client_order_id", ""),
                "symbol": getattr(prop, "symbol", "") if prop else "",
                "side": getattr(prop, "side", "") if prop else "",
                "quantity": str(getattr(prop, "quantity", "")) if prop else "",
                "status": getattr(ack, "status", ""),
            }
            w.writerow(row)
            rows_written += 1
    log.info("orders-log: appended %d rows to %s", rows_written, p)
    return rows_written


__all__ = ["append_orders_log"]
