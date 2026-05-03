"""
Paper-trading read endpoints — Alpaca paper account state over HTTP.

Surfaces live Alpaca paper account snapshot + positions so the web UI
can render a "Live Paper Trading" tile alongside the backtest results.

Routes are read-only. The order-submission path is deliberately kept on
the CLI behind explicit `--submit --confirm` + TRADING_ENABLED gates —
exposing it via HTTP would invite accidents.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from quant.adapters.alpaca import AlpacaBrokerAdapter
from quant.config import Settings, get_settings
from quant.core.dependencies import get_current_user
from quant.db.models import User
from quant.execution.live_session import (
    fetch_account_snapshot,
    fetch_current_positions,
)

router = APIRouter(prefix="/paper", tags=["paper"])


# ---------------------------------------------------------------
# Schemas — strict, all fields explicit, no `Any` in response models
# ---------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AccountOut(_Strict):
    equity: str  # Decimal serialized as string to avoid float drift
    cash: str
    buying_power: str
    status: str
    paper: bool


class PositionOut(_Strict):
    symbol: str
    quantity: str  # Decimal-as-string
    last_price: str
    avg_entry_price: str
    market_value: str
    unrealized_pl: str
    unrealized_plpc: str


class PositionsOut(_Strict):
    positions: list[PositionOut]
    total_market_value: str
    total_unrealized_pl: str


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]


def _require_paper_keys(settings: Settings) -> None:
    if (
        not settings.alpaca_api_key_id.get_secret_value()
        or not settings.alpaca_api_secret_key.get_secret_value()
    ):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Alpaca paper credentials not configured",
        )
    if not settings.alpaca_paper:
        # Refuse to expose live-broker state through the read endpoints —
        # live data has different sensitivity than paper.
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "ALPACA_PAPER must be true; paper endpoints refuse to read a live account",
        )


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@router.get(
    "/account",
    response_model=AccountOut,
    summary="Live Alpaca paper account snapshot",
    description=(
        "Returns equity, cash, buying power, broker-reported status, and the "
        "`paper` flag. Decimal fields are serialized as strings to preserve "
        "precision across JSON."
    ),
)
async def get_account(
    settings: SettingsDep,
    user: CurrentUserDep,
) -> AccountOut:
    _require_paper_keys(settings)
    del user  # auth-only

    adapter = AlpacaBrokerAdapter()
    try:
        snap = await fetch_account_snapshot(adapter)
    finally:
        await adapter.aclose()

    return AccountOut(
        equity=str(snap.equity),
        cash=str(snap.cash),
        buying_power=str(snap.buying_power),
        status=snap.status,
        paper=snap.paper,
    )


@router.get(
    "/positions",
    response_model=PositionsOut,
    summary="Live Alpaca paper open positions",
    description=(
        "Returns each open long position with mark-to-market PnL "
        "(`unrealized_pl` in dollars, `unrealized_plpc` as a fraction). "
        "Shorts and zero-qty rows are filtered out."
    ),
)
async def get_positions(
    settings: SettingsDep,
    user: CurrentUserDep,
) -> PositionsOut:
    _require_paper_keys(settings)
    del user

    from decimal import Decimal as _Decimal

    adapter = AlpacaBrokerAdapter()
    try:
        positions = await fetch_current_positions(adapter)
        raw = await adapter.positions()
    finally:
        await adapter.aclose()

    raw_by_sym: dict[str, dict[str, object]] = {
        str(r.get("symbol", "")): r for r in raw if isinstance(r, dict)
    }

    out: list[PositionOut] = []
    total_market = _Decimal("0")
    total_pnl = _Decimal("0")
    for p in positions:
        r = raw_by_sym.get(p.symbol, {})
        market_value = _Decimal(str(r.get("market_value", "0")))
        upl = _Decimal(str(r.get("unrealized_pl", "0")))
        upl_pc = str(r.get("unrealized_plpc", "0"))
        avg_entry = str(r.get("avg_entry_price", "0"))
        total_market += market_value
        total_pnl += upl
        out.append(
            PositionOut(
                symbol=p.symbol,
                quantity=str(p.quantity),
                last_price=str(p.last_price),
                avg_entry_price=avg_entry,
                market_value=str(market_value),
                unrealized_pl=str(upl),
                unrealized_plpc=upl_pc,
            )
        )

    return PositionsOut(
        positions=out,
        total_market_value=str(total_market),
        total_unrealized_pl=str(total_pnl),
    )
