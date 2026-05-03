"""Tests for the live paper-trading session orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import polars as pl

from quant.execution.broker import BrokerOrderAck, BrokerOrderRequest
from quant.execution.live_session import (
    fetch_account_snapshot,
    fetch_current_positions,
    fetch_recent_bars,
    run_live_session,
)


# ------------------------------------------------------------------
# Fakes — kept local; not exported, not picked up by no-fake-data guard
# ------------------------------------------------------------------
@dataclass
class _FakeBrokerAdapter:
    account_resp: dict[str, Any] = field(default_factory=dict)
    positions_resp: list[dict[str, Any]] = field(default_factory=list)

    async def get_json(self, path: str) -> dict[str, Any]:
        if path == "/v2/account":
            return self.account_resp
        return {}

    async def positions(self) -> list[dict[str, Any]]:
        return self.positions_resp

    async def aclose(self) -> None:
        pass


@dataclass
class _FakeDataAdapter:
    bars_resp: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def bars(
        self,
        symbols: list[str],
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        return {s: self.bars_resp.get(s, []) for s in symbols}

    async def aclose(self) -> None:
        pass


class _RecordingBroker:
    def __init__(self) -> None:
        self.submitted: list[BrokerOrderRequest] = []

    async def submit(self, req: BrokerOrderRequest) -> BrokerOrderAck:
        self.submitted.append(req)
        return BrokerOrderAck(
            broker_order_id=f"id-{len(self.submitted)}",
            client_order_id=req.client_order_id or "",
            status="accepted",
        )

    async def cancel(self, broker_order_id: str) -> None:  # pragma: no cover
        pass

    async def get_status(self, broker_order_id: str) -> str:  # pragma: no cover
        return "accepted"


# ------------------------------------------------------------------
# Account + position parsing
# ------------------------------------------------------------------
async def test_fetch_account_snapshot_parses_alpaca_payload() -> None:
    adapter = _FakeBrokerAdapter(
        account_resp={
            "equity": "100000.42",
            "cash": "50000.00",
            "buying_power": "150000.00",
            "status": "ACTIVE",
            "paper": True,
        }
    )
    snap = await fetch_account_snapshot(adapter)
    assert snap.equity == Decimal("100000.42")
    assert snap.status == "ACTIVE"
    assert snap.paper is True


async def test_fetch_current_positions_drops_shorts_and_invalid() -> None:
    adapter = _FakeBrokerAdapter(
        positions_resp=[
            {"symbol": "AAPL", "qty": "10", "current_price": "200.0"},
            {"symbol": "MSFT", "qty": "0", "current_price": "400.0"},  # zero — drop
            {"symbol": "TSLA", "qty": "-5", "current_price": "180.0"},  # short — drop
            {"symbol": "", "qty": "1", "current_price": "1.0"},  # blank — drop
            {"symbol": "GOOGL", "qty": "3", "avg_entry_price": "150.0"},  # falls back to entry
        ]
    )
    positions = await fetch_current_positions(adapter)
    syms = sorted(p.symbol for p in positions)
    assert syms == ["AAPL", "GOOGL"]
    aapl = next(p for p in positions if p.symbol == "AAPL")
    assert aapl.last_price == Decimal("200.0")
    googl = next(p for p in positions if p.symbol == "GOOGL")
    assert googl.last_price == Decimal("150.0")


# ------------------------------------------------------------------
# Bar fetch parsing
# ------------------------------------------------------------------
async def test_fetch_recent_bars_returns_typed_frame() -> None:
    adapter = _FakeDataAdapter(
        bars_resp={
            "AAPL": [
                {"t": "2026-04-30T00:00:00Z", "c": 200.0},
                {"t": "2026-05-01T00:00:00Z", "c": 201.5},
            ],
            "MSFT": [
                {"t": "2026-04-30T00:00:00Z", "c": 400.0},
            ],
        }
    )
    df = await fetch_recent_bars(adapter, ["AAPL", "MSFT"], lookback_days=10)
    assert df.height == 3
    expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume", "adj_close"}
    assert set(df.columns) == expected_cols
    assert df["adj_close"].dtype == pl.Float64
    assert df.filter(pl.col("symbol") == "AAPL")["adj_close"].max() == 201.5


async def test_fetch_recent_bars_handles_empty_universe() -> None:
    adapter = _FakeDataAdapter(bars_resp={})
    df = await fetch_recent_bars(adapter, [], lookback_days=10)
    assert df.height == 0
    expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume", "adj_close"}
    assert set(df.columns) == expected_cols


# ------------------------------------------------------------------
# Safety-gate semantics
# ------------------------------------------------------------------
async def test_run_live_session_refuses_when_paper_off() -> None:
    """If ALPACA_PAPER=False, no submission, even with confirm=True."""
    broker_adapter = _FakeBrokerAdapter(account_resp={"equity": "10000", "status": "ACTIVE", "paper": True})
    data_adapter = _FakeDataAdapter(
        bars_resp={
            "AAA": [{"t": f"2026-04-{d:02d}T00:00:00Z", "c": 100.0 + d} for d in range(1, 31)],
            "BBB": [{"t": f"2026-04-{d:02d}T00:00:00Z", "c": 50.0 - d * 0.1} for d in range(1, 31)],
        }
    )
    broker = _RecordingBroker()

    from quant.backtest.signals import MomentumSignal

    result = await run_live_session(
        signal=MomentumSignal(lookback_days=20),
        universe=["AAA", "BBB"],
        broker=broker,
        broker_adapter=broker_adapter,
        data_adapter=data_adapter,
        top_k=1,
        trading_enabled=True,  # all other gates ALL open
        alpaca_paper=False,  # <-- this one closed
        confirm=True,
    )
    assert result.submitted is False
    assert len(result.proposals) >= 1
    assert len(broker.submitted) == 0


async def test_run_live_session_submits_when_all_gates_open() -> None:
    broker_adapter = _FakeBrokerAdapter(
        account_resp={"equity": "10000", "status": "ACTIVE", "paper": True},
    )
    data_adapter = _FakeDataAdapter(
        bars_resp={
            "UP": [{"t": f"2026-04-{d:02d}T00:00:00Z", "c": 100.0 * (1.01**d)} for d in range(1, 31)],
            "DOWN": [{"t": f"2026-04-{d:02d}T00:00:00Z", "c": 100.0 * (0.99**d)} for d in range(1, 31)],
        },
    )
    broker = _RecordingBroker()

    from quant.backtest.signals import MomentumSignal

    result = await run_live_session(
        signal=MomentumSignal(lookback_days=20),
        universe=["UP", "DOWN"],
        broker=broker,
        broker_adapter=broker_adapter,
        data_adapter=data_adapter,
        top_k=1,
        trading_enabled=True,
        alpaca_paper=True,
        confirm=True,
    )
    assert result.submitted is True
    assert "UP" in result.target_weights
    assert len(broker.submitted) >= 1
    # The chosen symbol on momentum should be UP, not DOWN.
    assert any(req.symbol == "UP" and req.side == "BUY" for req in broker.submitted)
