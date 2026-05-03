"""
Pre-trade risk gate — checks every proposed order against hard limits
before it can be submitted.

This is the safety net that sits between the signal-generated plan and
the broker. The walk-forward backtest is risk-naive on purpose (equal
weight, no caps, no kill-switch); production execution is not. Live
proposals must pass these checks or get downgraded to plan-only.

Sources of truth:
- Per-position dollar cap        =  account_equity * settings.max_position_pct
- Max number of positions        =  settings.max_positions
- Daily realized-loss kill       =  -account_equity * settings.daily_loss_limit_pct
                                    (kills new BUYs once breached;
                                     SELLs always pass — get-out trumps)
- Drawdown kill                  =  account_equity at-rest below
                                    peak * (1 - settings.drawdown_kill_pct)
                                    (peak tracked by caller — usually
                                     persisted between runs)

Sector caps are deferred until the universe carries sector tags.

Output: a `RiskCheckResult` per proposal — `accept` or `block` with a
short reason. Caller applies the filter; if any BUY blocks the whole
session SHOULD log + halt rather than partial-fill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from quant.execution.paper_session import ProposedOrder

log = logging.getLogger("quant.execution.risk_gate")


@dataclass(frozen=True)
class RiskCheckResult:
    proposal: ProposedOrder
    accepted: bool
    reason: str  # short human-readable; "" when accepted


@dataclass(frozen=True)
class RiskLimits:
    """Snapshot of risk caps at a single point in time."""

    max_position_pct: float
    max_positions: int
    daily_loss_limit_pct: float  # absolute fraction (0.02 == 2%)
    drawdown_kill_pct: float


@dataclass(frozen=True)
class AccountState:
    """Inputs the risk gate needs about the live account."""

    equity: Decimal
    realized_pnl_today: Decimal = Decimal("0")
    peak_equity: Decimal | None = None  # None disables drawdown check


# ------------------------------------------------------------------
# Individual checks — each returns (accepted, reason)
# ------------------------------------------------------------------
def _check_position_size(p: ProposedOrder, equity: Decimal, max_pct: float) -> tuple[bool, str]:
    if p.side != "BUY":
        return True, ""
    cap = equity * Decimal(str(max_pct))
    target = abs(p.target_value)
    if target > cap:
        return False, (
            f"position size ${target} > cap ${cap.quantize(Decimal('1'))} ({max_pct:.1%} of equity)"
        )
    return True, ""


def _check_max_positions(side: str, n_existing: int, n_new_buys_so_far: int, max_n: int) -> tuple[bool, str]:
    if side != "BUY":
        return True, ""
    if n_existing + n_new_buys_so_far >= max_n:
        return False, (
            f"max_positions={max_n} reached (existing={n_existing}, new_buys_in_session={n_new_buys_so_far})"
        )
    return True, ""


def _check_daily_loss_kill(
    side: str,
    realized_pnl_today: Decimal,
    equity: Decimal,
    daily_loss_pct: float,
) -> tuple[bool, str]:
    """SELLs always pass — getting out trumps. BUYs blocked once breached."""
    if side != "BUY":
        return True, ""
    threshold = -(equity * Decimal(str(daily_loss_pct)))
    if realized_pnl_today < threshold:
        return False, (
            f"daily_loss_limit breached: realized_pnl={realized_pnl_today} "
            f"< {threshold.quantize(Decimal('1'))} "
            f"({daily_loss_pct:.1%} of equity)"
        )
    return True, ""


def _check_drawdown_kill(
    side: str,
    equity: Decimal,
    peak_equity: Decimal | None,
    drawdown_pct: float,
) -> tuple[bool, str]:
    if side != "BUY" or peak_equity is None:
        return True, ""
    if peak_equity <= 0:
        return True, ""
    drawdown = (peak_equity - equity) / peak_equity
    if drawdown > Decimal(str(drawdown_pct)):
        return False, (
            f"drawdown {float(drawdown):.1%} > kill threshold "
            f"{drawdown_pct:.1%} (peak=${peak_equity}, current=${equity})"
        )
    return True, ""


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def apply_risk_gate(
    proposals: list[ProposedOrder],
    *,
    account: AccountState,
    limits: RiskLimits,
    n_existing_positions: int,
) -> list[RiskCheckResult]:
    """
    Run every proposal through every check, in order. Returns one
    RiskCheckResult per proposal (preserves input order).

    The gate is conservative: ANY check that fails on a BUY blocks it.
    SELLs are never blocked (getting out is always allowed).
    """
    results: list[RiskCheckResult] = []
    n_new_buys = 0
    for p in proposals:
        accepted = True
        reason = ""

        ok, msg = _check_drawdown_kill(p.side, account.equity, account.peak_equity, limits.drawdown_kill_pct)
        if not ok:
            accepted = False
            reason = msg

        if accepted:
            ok, msg = _check_daily_loss_kill(
                p.side, account.realized_pnl_today, account.equity, limits.daily_loss_limit_pct
            )
            if not ok:
                accepted = False
                reason = msg

        if accepted:
            ok, msg = _check_max_positions(p.side, n_existing_positions, n_new_buys, limits.max_positions)
            if not ok:
                accepted = False
                reason = msg

        if accepted:
            ok, msg = _check_position_size(p, account.equity, limits.max_position_pct)
            if not ok:
                accepted = False
                reason = msg

        if accepted and p.side == "BUY":
            n_new_buys += 1

        results.append(RiskCheckResult(proposal=p, accepted=accepted, reason=reason))
    return results


__all__ = [
    "AccountState",
    "RiskCheckResult",
    "RiskLimits",
    "apply_risk_gate",
]
