"""Unit tests for the pre-trade risk gate."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quant.execution.paper_session import ProposedOrder
from quant.execution.risk_gate import (
    AccountState,
    RiskLimits,
    apply_risk_gate,
)

_DEFAULT_LIMITS = RiskLimits(
    max_position_pct=0.05,
    max_positions=20,
    daily_loss_limit_pct=0.02,
    drawdown_kill_pct=0.15,
)


def _buy(sym: str, target: Decimal) -> ProposedOrder:
    return ProposedOrder(
        symbol=sym,
        side="BUY",
        quantity=Decimal(1),
        delta_shares=Decimal(1),
        target_value=target,
        current_value=Decimal(0),
    )


def _sell(sym: str, current: Decimal) -> ProposedOrder:
    return ProposedOrder(
        symbol=sym,
        side="SELL",
        quantity=Decimal(1),
        delta_shares=Decimal(-1),
        target_value=Decimal(0),
        current_value=current,
    )


# ------------------------------------------------------------------
# Position-size cap
# ------------------------------------------------------------------
def test_buy_at_cap_passes() -> None:
    out = apply_risk_gate(
        [_buy("AAPL", Decimal("5000"))],
        account=AccountState(equity=Decimal("100000")),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=0,
    )
    assert out[0].accepted is True


def test_buy_over_cap_blocks() -> None:
    out = apply_risk_gate(
        [_buy("AAPL", Decimal("6000"))],  # 6% > 5% cap on $100k
        account=AccountState(equity=Decimal("100000")),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=0,
    )
    assert out[0].accepted is False
    assert "position size" in out[0].reason


# ------------------------------------------------------------------
# Max-positions
# ------------------------------------------------------------------
def test_max_positions_blocks_extra_buys() -> None:
    proposals = [_buy(f"S{i:02d}", Decimal("3000")) for i in range(5)]
    out = apply_risk_gate(
        proposals,
        account=AccountState(equity=Decimal("100000")),
        limits=RiskLimits(
            max_position_pct=0.05,
            max_positions=3,
            daily_loss_limit_pct=0.02,
            drawdown_kill_pct=0.15,
        ),
        n_existing_positions=0,
    )
    assert sum(1 for r in out if r.accepted) == 3
    assert all("max_positions" in r.reason for r in out if not r.accepted)


def test_max_positions_counts_existing() -> None:
    out = apply_risk_gate(
        [_buy("X", Decimal("3000"))],
        account=AccountState(equity=Decimal("100000")),
        limits=RiskLimits(
            max_position_pct=0.05,
            max_positions=3,
            daily_loss_limit_pct=0.02,
            drawdown_kill_pct=0.15,
        ),
        n_existing_positions=3,  # already at cap
    )
    assert out[0].accepted is False


# ------------------------------------------------------------------
# Sells always pass
# ------------------------------------------------------------------
def test_sells_always_pass_even_during_drawdown() -> None:
    out = apply_risk_gate(
        [_sell("AAPL", Decimal("10000"))],
        account=AccountState(
            equity=Decimal("70000"),
            peak_equity=Decimal("100000"),  # 30% drawdown — past kill
            realized_pnl_today=Decimal("-5000"),  # past daily loss limit
        ),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=20,  # at max_positions cap
    )
    assert out[0].accepted is True


# ------------------------------------------------------------------
# Daily loss kill
# ------------------------------------------------------------------
def test_daily_loss_kill_blocks_buys_after_breach() -> None:
    out = apply_risk_gate(
        [_buy("AAPL", Decimal("3000"))],
        account=AccountState(
            equity=Decimal("100000"),
            realized_pnl_today=Decimal("-2500"),  # > 2% loss
        ),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=0,
    )
    assert out[0].accepted is False
    assert "daily_loss" in out[0].reason


def test_daily_loss_kill_passes_buys_below_threshold() -> None:
    out = apply_risk_gate(
        [_buy("AAPL", Decimal("3000"))],
        account=AccountState(
            equity=Decimal("100000"),
            realized_pnl_today=Decimal("-1500"),  # 1.5% < 2% cap
        ),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=0,
    )
    assert out[0].accepted is True


# ------------------------------------------------------------------
# Drawdown kill
# ------------------------------------------------------------------
def test_drawdown_kill_blocks_buys_when_past_threshold() -> None:
    out = apply_risk_gate(
        [_buy("AAPL", Decimal("3000"))],
        account=AccountState(
            equity=Decimal("80000"),
            peak_equity=Decimal("100000"),  # 20% drawdown > 15% threshold
        ),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=0,
    )
    assert out[0].accepted is False
    assert "drawdown" in out[0].reason


def test_drawdown_kill_skipped_when_no_peak() -> None:
    """A fresh account with no historical peak should not be drawdown-killed."""
    out = apply_risk_gate(
        [_buy("AAPL", Decimal("3000"))],
        account=AccountState(equity=Decimal("100000"), peak_equity=None),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=0,
    )
    assert out[0].accepted is True


# ------------------------------------------------------------------
# Multiple checks compose
# ------------------------------------------------------------------
def test_first_check_to_fail_wins() -> None:
    """A proposal that violates two checks should report just the first reason."""
    out = apply_risk_gate(
        [_buy("AAPL", Decimal("10000"))],  # 10% > 5% cap
        account=AccountState(
            equity=Decimal("100000"),
            peak_equity=Decimal("100000"),
        ),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=0,
    )
    assert out[0].accepted is False
    # drawdown is checked first; but with no drawdown, position-size wins.
    assert "position size" in out[0].reason


@pytest.mark.parametrize(
    "side,expected_pass",
    [("BUY", False), ("SELL", True)],
)
def test_post_kill_buys_blocked_sells_pass(side: str, expected_pass: bool) -> None:
    p = _buy("X", Decimal("3000")) if side == "BUY" else _sell("X", Decimal("3000"))
    out = apply_risk_gate(
        [p],
        account=AccountState(
            equity=Decimal("80000"),
            peak_equity=Decimal("100000"),  # 20% drawdown
            realized_pnl_today=Decimal("-3000"),  # past daily loss
        ),
        limits=_DEFAULT_LIMITS,
        n_existing_positions=20,  # max cap
    )
    assert out[0].accepted is expected_pass
