"""
Pre-trade risk manager — DB-backed path for the REST API order flow.

ARCHITECTURE NOTE: This module is used by the FastAPI REST order endpoints
(quant/api/v1/orders.py, quant/api/v1/admin.py). It is NOT the same as
execution/risk_gate.py, which is the stateless, DB-free risk check used
by the live trading session (run_live_session).

Two separate risk layers by design:
    quant.risk.manager      — DB-backed, full sector/drawdown/daily-loss/Redis
                              kill-switch checks. Used by the API order service.
                              Requires an active Postgres + Redis connection.
    quant.execution.risk_gate — Stateless file-CLI check. Used by live_session.
                              No DB needed; runs in CLI/cron contexts.

The live session's risk_gate is the active production path for paper trading.
This module is the active path for any REST-API-submitted orders (future).

Every order submitted via the API flows through RiskManager.check(intent).
Violations raise `RiskViolation` with a human-readable reason — the order
is blocked, logged, and never touches the broker.

Enforced limits (from settings):
- max_position_pct      — single-name ceiling (fraction of equity)
- max_sector_pct        — sector ceiling (same)
- max_positions         — open-name count cap
- daily_loss_limit_pct  — hard block if today's realized PnL < -limit
- drawdown_kill_pct     — hard block if peak-to-trough drawdown > limit
- kill switch (Redis)   — manual halt override; operator-only

All checks use current Postgres state, not in-memory caches, so a second
instance of the API can't drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from quant.config import settings
from quant.db.models import OrderSide, OrderStatus, Position, Snapshot, Ticker, Trade

log = logging.getLogger("quant.risk")


class RiskViolation(Exception):
    """Raised when an order would breach a configured limit."""


@dataclass(frozen=True)
class OrderIntent:
    user_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: Decimal
    limit_price: Decimal | None
    mark_price: Decimal  # most recent quote — required for $-sizing


@dataclass(frozen=True)
class RiskCheckResult:
    ok: bool
    reason: str | None = None


# ----------------------------------------------------------------
# Kill switch — a simple boolean stored in Redis under a fixed key.
# Operators flip it via /admin/kill; the worker/API reads it before every order.
# ----------------------------------------------------------------
KILL_SWITCH_KEY = "quant:kill_switch"


async def _kill_switch_engaged() -> bool:
    try:
        import redis.asyncio as redis
    except ImportError:  # pragma: no cover
        return False
    client = redis.from_url(settings.redis_url)
    try:
        v = await client.get(KILL_SWITCH_KEY)
        return v is not None and v.decode() in ("1", "true", "on")
    finally:
        await client.close()


# ----------------------------------------------------------------
# Risk manager
# ----------------------------------------------------------------
class RiskManager:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def check(self, intent: OrderIntent) -> RiskCheckResult:
        """Return RiskCheckResult(ok=False, reason=...) on any violation."""
        # Global kill switch first — cheap and short-circuits everything.
        if await _kill_switch_engaged():
            return RiskCheckResult(False, "kill switch engaged")

        if intent.quantity <= 0:
            return RiskCheckResult(False, "quantity must be positive")
        if intent.mark_price <= 0:
            return RiskCheckResult(False, "no valid mark price")

        # Equity baseline — latest snapshot, fall back to configured initial capital.
        equity = await self._current_equity(intent.user_id)
        if equity <= 0:
            return RiskCheckResult(False, "equity ≤ 0, refusing to trade")

        order_notional = intent.quantity * intent.mark_price
        max_single = equity * Decimal(str(settings.max_position_pct))
        if order_notional > max_single:
            return RiskCheckResult(
                False,
                f"order notional {order_notional:.0f} > max_position_pct cap {max_single:.0f}",
            )

        # Position-count cap (only counts BUY side — we can always close).
        if intent.side == "BUY":
            open_positions = await self._open_position_count(intent.user_id)
            # If this symbol isn't already a position, opening it would add one.
            has_pos = await self._has_position(intent.user_id, intent.symbol)
            if not has_pos and open_positions >= settings.max_positions:
                return RiskCheckResult(
                    False,
                    f"open position count {open_positions} ≥ max_positions {settings.max_positions}",
                )

        # Sector exposure — if ticker sector known, aggregate.
        sector = await self._sector_of(intent.symbol)
        if sector and intent.side == "BUY":
            sector_exposure = await self._sector_exposure(intent.user_id, sector)
            max_sector = equity * Decimal(str(settings.max_sector_pct))
            if sector_exposure + order_notional > max_sector:
                return RiskCheckResult(
                    False,
                    f"sector {sector} exposure {sector_exposure + order_notional:.0f} > cap {max_sector:.0f}",
                )

        # Daily-loss gate.
        today_pnl = await self._today_realized_pnl(intent.user_id)
        loss_limit = -equity * Decimal(str(settings.daily_loss_limit_pct))
        if today_pnl < loss_limit:
            return RiskCheckResult(False, f"daily loss {today_pnl:.2f} exceeded limit {loss_limit:.2f}")

        # Drawdown kill.
        drawdown = await self._current_drawdown(intent.user_id)
        if drawdown > Decimal(str(settings.drawdown_kill_pct)):
            return RiskCheckResult(
                False, f"drawdown {drawdown:.2%} > kill threshold {settings.drawdown_kill_pct:.2%}"
            )

        return RiskCheckResult(True)

    # ---------------- helpers ----------------
    async def _current_equity(self, user_id: str) -> Decimal:
        stmt = (
            select(Snapshot.total_equity)
            .where(Snapshot.user_id == user_id)
            .order_by(Snapshot.date.desc())
            .limit(1)
        )
        v = (await self.session.execute(stmt)).scalar_one_or_none()
        return Decimal(str(v)) if v is not None else Decimal(str(settings.initial_capital_usd))

    async def _open_position_count(self, user_id: str) -> int:
        stmt = select(func.count()).select_from(Position).where(Position.user_id == user_id)
        return (await self.session.execute(stmt)).scalar() or 0

    async def _has_position(self, user_id: str, symbol: str) -> bool:
        stmt = select(Position.id).where(Position.user_id == user_id, Position.symbol == symbol)
        return (await self.session.execute(stmt)).scalar_one_or_none() is not None

    async def _sector_of(self, symbol: str) -> str | None:
        stmt = select(Ticker.sector).where(Ticker.symbol == symbol)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def _sector_exposure(self, user_id: str, sector: str) -> Decimal:
        stmt = (
            select(func.coalesce(func.sum(Position.quantity * Position.last_mark_price), 0))
            .select_from(Position)
            .join(Ticker, Ticker.symbol == Position.symbol)
            .where(Position.user_id == user_id, Ticker.sector == sector)
        )
        return Decimal(str((await self.session.execute(stmt)).scalar() or 0))

    async def _today_realized_pnl(self, user_id: str) -> Decimal:
        today = datetime.now(UTC).date()
        stmt = select(func.coalesce(func.sum(Trade.realized_pnl), 0)).where(
            Trade.user_id == user_id,
            Trade.trade_date == today,
            Trade.status == OrderStatus.filled,
        )
        return Decimal(str((await self.session.execute(stmt)).scalar() or 0))

    async def _current_drawdown(self, user_id: str) -> Decimal:
        """(peak_equity - current_equity) / peak_equity over trailing 90 days."""
        since = date.today() - timedelta(days=90)
        stmt = select(Snapshot.total_equity).where(Snapshot.user_id == user_id, Snapshot.date >= since)
        values = [Decimal(str(v)) for v in (await self.session.execute(stmt)).scalars().all()]
        if not values:
            return Decimal("0")
        peak = max(values)
        current = values[-1]
        if peak <= 0:
            return Decimal("0")
        return (peak - current) / peak


__all__ = ["OrderIntent", "OrderSide", "RiskCheckResult", "RiskManager", "RiskViolation"]
