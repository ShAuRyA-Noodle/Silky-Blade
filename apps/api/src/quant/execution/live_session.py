"""
Live paper-trading session — pulls real positions + real bars from Alpaca,
computes a signal, computes orders, optionally submits them.

This is the bridge from research infrastructure to live (paper) execution.
The session never bypasses safety gates — submission requires every one of:

    settings.trading_enabled  is True
    settings.alpaca_paper     is True   (refuses live broker by default)
    caller's `confirm` flag   is True

If any of those is False, the session computes the plan and refuses to
submit. The plan is still printed / returned, so a human can review.

Inputs the session needs:
    - signal producer (any quant.backtest.signals.SignalProducer)
    - target portfolio_value (override or "use broker equity")
    - top_k (number of positions)
    - universe (symbols to score; defaults to DEV_UNIVERSE)
    - lookback_days (price history for the signal)

The session does NOT pull a years-long backfill — the price-history fetch
is bounded by `lookback_days` (default 240 trading days = roughly one year)
to keep latency low and not hammer Alpaca's rate limits.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import polars as pl

from quant.backtest.engine import SignalProducer
from quant.execution.broker import Broker, BrokerOrderAck
from quant.execution.paper_session import (
    Position,
    ProposedOrder,
    TargetAllocation,
    compute_target_orders,
    submit_orders,
)
from quant.execution.reconciliation import FillReport, reconcile
from quant.execution.risk_gate import (
    AccountState,
    RiskCheckResult,
    RiskLimits,
    apply_risk_gate,
)

log = logging.getLogger("quant.execution.live_session")


@dataclass(frozen=True)
class AccountSnapshot:
    """What the broker says about the account at the start of a session."""

    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    status: str  # "ACTIVE" | other broker-reported state
    paper: bool


@dataclass(frozen=True)
class LiveSessionResult:
    session_id: str
    as_of: date
    account: AccountSnapshot
    n_symbols_scored: int
    target_weights: dict[str, float]
    proposals: list[ProposedOrder]
    submitted: bool
    acks: list[BrokerOrderAck] = field(default_factory=list)
    risk_results: list[RiskCheckResult] = field(default_factory=list)
    fills: list[FillReport] = field(default_factory=list)


# ------------------------------------------------------------------
# Broker reads
# ------------------------------------------------------------------
async def fetch_account_snapshot(broker_adapter: Any) -> AccountSnapshot:
    """Read the broker's account endpoint into a strict snapshot."""
    raw = await broker_adapter.get_json("/v2/account")
    if not isinstance(raw, dict):
        raise RuntimeError(f"unexpected account payload type: {type(raw).__name__}")
    return AccountSnapshot(
        equity=Decimal(str(raw.get("equity", "0"))),
        cash=Decimal(str(raw.get("cash", "0"))),
        buying_power=Decimal(str(raw.get("buying_power", "0"))),
        status=str(raw.get("status", "?")),
        paper=bool(raw.get("paper", True)),
    )


async def fetch_current_positions(broker_adapter: Any) -> list[Position]:
    """Read open broker positions (long-only)."""
    raw = await broker_adapter.positions()
    out: list[Position] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sym = str(entry.get("symbol", "")).strip()
        qty = Decimal(str(entry.get("qty", "0")))
        # Alpaca may report qty < 0 for shorts; we ignore shorts here.
        if qty <= 0 or not sym:
            continue
        last = Decimal(str(entry.get("current_price") or entry.get("avg_entry_price") or "0"))
        out.append(Position(symbol=sym, quantity=qty, last_price=last))
    return out


# ------------------------------------------------------------------
# Price fetch (Alpaca data, daily bars, bounded lookback)
# ------------------------------------------------------------------
async def fetch_recent_bars(
    data_adapter: Any,
    symbols: Iterable[str],
    *,
    lookback_days: int = 240,
    end: date | None = None,
) -> pl.DataFrame:
    """
    Fetch daily bars for `symbols` for the most recent `lookback_days` of
    trading data. Returns a polars frame with columns [date, symbol, adj_close].

    Uses the Alpaca data adapter's `bars()` method, which accepts a list of
    symbols in one request — keeps round-trips bounded.
    """
    end_dt = datetime.combine(end, datetime.min.time(), tzinfo=UTC) if end is not None else datetime.now(UTC)
    # 1.5x calendar days to cover weekends + holidays in the lookback window.
    start_dt = end_dt - timedelta(days=int(lookback_days * 1.5))
    syms = sorted({s.strip() for s in symbols if s and s.strip()})
    empty_schema = {
        "date": pl.Date,
        "symbol": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "adj_close": pl.Float64,
    }
    if not syms:
        return pl.DataFrame(schema=empty_schema)

    bars = await data_adapter.bars(syms, timeframe="1Day", start=start_dt, end=end_dt)
    if not isinstance(bars, dict):
        raise RuntimeError(f"unexpected bars payload type: {type(bars).__name__}")

    rows: list[dict[str, Any]] = []
    for sym, sym_bars in bars.items():
        if not isinstance(sym_bars, list):
            continue
        for bar in sym_bars:
            if not isinstance(bar, dict):
                continue
            ts = bar.get("t")
            c = bar.get("c")
            if ts is None or c is None:
                continue
            try:
                d = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
            except ValueError:
                continue
            # Alpaca's `adjustment="split"` returns split-adjusted values;
            # treat `c` as both close and adj_close for downstream code.
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": float(bar.get("o") or c),
                    "high": float(bar.get("h") or c),
                    "low": float(bar.get("l") or c),
                    "close": float(c),
                    "volume": int(bar.get("v") or 0),
                    "adj_close": float(c),
                }
            )

    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
            pl.col("adj_close").cast(pl.Float64),
        )
        .sort(["symbol", "date"])
    )


# ------------------------------------------------------------------
# Top-level session
# ------------------------------------------------------------------
def _safety_gate(
    *,
    trading_enabled: bool,
    alpaca_paper: bool,
    confirm: bool,
) -> tuple[bool, str]:
    """Return (allow_submission, reason). Triple-check; loud-fail by default."""
    if not alpaca_paper:
        return False, "ALPACA_PAPER is False — refuses to send orders to a live broker"
    if not trading_enabled:
        return False, "TRADING_ENABLED is False in .env.local"
    if not confirm:
        return False, "--confirm flag was not passed (default safe)"
    return True, "all gates open"


async def run_live_session(
    *,
    signal: SignalProducer,
    universe: list[str],
    broker: Broker,
    broker_adapter: Any,  # AlpacaBrokerAdapter (for account + positions)
    data_adapter: Any,  # AlpacaDataAdapter
    top_k: int = 5,
    lookback_days: int = 240,
    portfolio_value_override: Decimal | None = None,
    trading_enabled: bool = False,
    alpaca_paper: bool = True,
    confirm: bool = False,
    session_id: str | None = None,
    risk_limits: RiskLimits | None = None,
    peak_equity: Decimal | None = None,
    reconcile_max_polls: int = 30,
    reconcile_interval_seconds: float = 1.0,
) -> LiveSessionResult:
    """
    Pull positions + bars → score signal → compute orders → maybe submit.

    Even with all safety gates passed, this only ever submits to whatever
    `broker` was passed in. If the caller wires a paper broker, it's paper.
    The function does not select a broker on its own.
    """
    sid = session_id or f"live-{datetime.now(UTC).isoformat()}"
    log.info("live_session %s starting (universe=%d, top_k=%d)", sid, len(universe), top_k)

    account = await fetch_account_snapshot(broker_adapter)
    log.info("account: equity=%s status=%s paper=%s", account.equity, account.status, account.paper)

    bars = await fetch_recent_bars(data_adapter, universe, lookback_days=lookback_days)
    if bars.is_empty():
        raise RuntimeError(f"no bars fetched for universe of {len(universe)} symbols")

    as_of_date = bars["date"].max()
    if not isinstance(as_of_date, date):
        raise RuntimeError(f"unable to determine as-of date from bars; got {as_of_date!r}")

    # Score: we pass `bars` as the history; signal filters to <= as_of.
    scores = signal(as_of_date, bars)
    if scores.is_empty():
        raise RuntimeError(f"signal returned no scores at as-of date {as_of_date}")
    top = scores.sort("score", descending=True).head(top_k)
    syms = top["symbol"].to_list()
    weight_each = 1.0 / max(len(syms), 1)
    target_weights = dict.fromkeys(syms, weight_each)

    # Latest close per symbol for share computation.
    latest_close_df = (
        bars.filter(pl.col("symbol").is_in(syms))
        .sort(["symbol", "date"])
        .group_by("symbol", maintain_order=True)
        .agg(pl.col("adj_close").last().alias("last"))
    )
    latest_prices: dict[str, Decimal] = {
        row["symbol"]: Decimal(str(row["last"])) for row in latest_close_df.iter_rows(named=True)
    }

    current_positions = await fetch_current_positions(broker_adapter)
    # Carry user-held names forward so we can compute SELL orders for them.
    for pos in current_positions:
        if pos.symbol not in latest_prices and pos.last_price > 0:
            latest_prices[pos.symbol] = pos.last_price

    portfolio_value = portfolio_value_override or account.equity
    target = TargetAllocation(weights=target_weights, portfolio_value=portfolio_value)
    proposals = compute_target_orders(
        current_positions=current_positions,
        target=target,
        latest_prices=latest_prices,
    )

    # Pre-trade risk gate. When `risk_limits` is None we skip the check —
    # callers that want backtest-equivalent (no caps) behavior can pass
    # None; production callers always pass the live limits from settings.
    risk_results: list[RiskCheckResult] = []
    accepted_proposals: list[ProposedOrder] = proposals
    if risk_limits is not None:
        risk_results = apply_risk_gate(
            proposals,
            account=AccountState(equity=account.equity, peak_equity=peak_equity),
            limits=risk_limits,
            n_existing_positions=len(current_positions),
        )
        accepted_proposals = [r.proposal for r in risk_results if r.accepted]
        n_blocked = sum(1 for r in risk_results if not r.accepted)
        if n_blocked > 0:
            log.warning("risk gate blocked %d/%d proposals", n_blocked, len(proposals))
            for r in risk_results:
                if not r.accepted:
                    log.warning("  BLOCK %s %s: %s", r.proposal.side, r.proposal.symbol, r.reason)

    allow, reason = _safety_gate(
        trading_enabled=trading_enabled,
        alpaca_paper=alpaca_paper,
        confirm=confirm,
    )
    submitted = False
    acks: list[BrokerOrderAck] = []
    fills: list[FillReport] = []
    if allow and accepted_proposals:
        log.warning(
            "submitting %d orders via paper broker (gates open: %s)",
            len(accepted_proposals),
            reason,
        )
        acks = await submit_orders(broker, accepted_proposals, session_id=sid)
        submitted = True
        # Reconcile every submitted order to a terminal state. With ~10
        # orders and ~1s polling, this completes in seconds in liquid
        # market hours and ~30s out-of-hours (orders sit in 'accepted'
        # until the next session opens).
        log.info("reconciling %d orders…", len(acks))
        fills = await reconcile(
            broker_adapter,
            acks,
            max_polls=reconcile_max_polls,
            interval_seconds=reconcile_interval_seconds,
        )
    else:
        log.info(
            "plan-only mode (%s); %d proposals not submitted",
            reason,
            len(accepted_proposals),
        )

    return LiveSessionResult(
        session_id=sid,
        as_of=as_of_date,
        account=account,
        n_symbols_scored=int(scores.height),
        target_weights=target_weights,
        proposals=proposals,
        submitted=submitted,
        acks=acks,
        risk_results=risk_results,
        fills=fills,
    )


__all__ = [
    "AccountSnapshot",
    "LiveSessionResult",
    "RiskLimits",
    "fetch_account_snapshot",
    "fetch_current_positions",
    "fetch_recent_bars",
    "run_live_session",
]
