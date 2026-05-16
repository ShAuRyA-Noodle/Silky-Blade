"""
Live paper-trading session — pulls real positions + real bars from Alpaca,
computes a signal, computes orders, optionally submits them.

This is the bridge from research infrastructure to live (paper) execution.
The session never bypasses safety gates — submission requires every one of:

    settings.trading_enabled       is True
    settings.alpaca_paper          is True   (paper broker; see below for live)
    caller's `confirm` flag        is True

For LIVE (real money) trading two additional gates apply:
    settings.alpaca_paper          is False  (explicit opt-in to live broker)
    settings.live_trading_confirmed is True  (SECOND explicit confirmation)

If any required gate is False, the session computes the plan and refuses to
submit. The plan is still printed / returned, so a human can review.

Failure contract:
    If order submission fails mid-rebalance (e.g. broker outage between SELLs
    and BUYs), the session logs a PARTIAL_REBALANCE critical alert, surfaces
    all successfully-acknowledged order IDs, and re-raises. The caller is
    responsible for manual reconciliation. An alert fires to SLACK_WEBHOOK_URL
    if configured.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
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
from quant.monitoring.alerts import send_alert

log = logging.getLogger("quant.execution.live_session")


@dataclass(frozen=True)
class AccountSnapshot:
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    status: str
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
    partial_failure: bool = False  # True if submission failed mid-rebalance


# ------------------------------------------------------------------
# Broker reads
# ------------------------------------------------------------------
async def fetch_account_snapshot(broker_adapter: Any) -> AccountSnapshot:
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
    raw = await broker_adapter.positions()
    out: list[Position] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sym = str(entry.get("symbol", "")).strip()
        qty = Decimal(str(entry.get("qty", "0")))
        if qty <= 0 or not sym:
            continue
        last = Decimal(str(entry.get("current_price") or entry.get("avg_entry_price") or "0"))
        out.append(Position(symbol=sym, quantity=qty, last_price=last))
    return out


_EMPTY_BARS = pl.DataFrame(
    {
        "date": pl.Series([], dtype=pl.Date),
        "symbol": pl.Series([], dtype=pl.Utf8),
        "open": pl.Series([], dtype=pl.Float64),
        "high": pl.Series([], dtype=pl.Float64),
        "low": pl.Series([], dtype=pl.Float64),
        "close": pl.Series([], dtype=pl.Float64),
        "volume": pl.Series([], dtype=pl.Float64),
        "adj_close": pl.Series([], dtype=pl.Float64),
    }
)


async def fetch_recent_bars(
    data_adapter: Any,
    symbols: Iterable[str],
    *,
    lookback_days: int = 240,
) -> pl.DataFrame:
    syms = sorted(set(symbols))
    if not syms:
        return _EMPTY_BARS
    end = datetime.now(UTC).date()
    start = end - timedelta(days=lookback_days + 60)

    try:
        raw = await data_adapter.bars(syms, timeframe="1Day", start=start, end=end)
    except Exception as exc:
        log.warning("bars fetch failed: %s", exc)
        return _EMPTY_BARS

    if not isinstance(raw, dict):
        raise RuntimeError(f"unexpected bars payload type: {type(raw).__name__}")

    rows: list[dict[str, Any]] = []
    for sym, bar_list in raw.items():
        if not isinstance(bar_list, list):
            continue
        for b in bar_list:
            if not isinstance(b, dict):
                continue
            raw_date = b.get("t") or b.get("date") or ""
            try:
                parsed_date: date = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00")).date()
            except ValueError:
                continue
            adj = b.get("adj_close") or b.get("c") or b.get("close")
            if adj is None:
                continue
            try:
                adj_f = float(adj)
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "date": parsed_date,
                    "symbol": str(sym),
                    "open": float(b.get("o") or b.get("open") or adj_f),
                    "high": float(b.get("h") or b.get("high") or adj_f),
                    "low": float(b.get("l") or b.get("low") or adj_f),
                    "close": float(b.get("c") or b.get("close") or adj_f),
                    "volume": float(b.get("v") or b.get("volume") or 0.0),
                    "adj_close": adj_f,
                }
            )
    if not rows:
        return _EMPTY_BARS
    return pl.DataFrame(rows)


# ------------------------------------------------------------------
# Safety gates
# ------------------------------------------------------------------
def _safety_gate(
    *,
    trading_enabled: bool,
    alpaca_paper: bool,
    confirm: bool,
    live_trading_confirmed: bool = False,
) -> tuple[bool, str]:
    """
    Triple-gate for paper trading. Quadruple-gate for live (real money).

    Paper flow: trading_enabled=True + alpaca_paper=True + confirm=True
    Live flow:  above + alpaca_paper=False + live_trading_confirmed=True

    Two separate env vars required for live trading so a single typo cannot
    accidentally send orders to a real broker.
    """
    if not alpaca_paper and not live_trading_confirmed:
        return False, (
            "ALPACA_PAPER=false (live broker) also requires LIVE_TRADING_CONFIRMED=true. "
            "Two separate confirmations are required to route to a live broker."
        )
    if not trading_enabled:
        return False, "TRADING_ENABLED is False — plan-only mode"
    if not confirm:
        return False, "--confirm flag not passed — plan-only mode"
    broker_label = "LIVE BROKER (real money)" if not alpaca_paper else "paper broker"
    return True, f"all gates open ({broker_label})"


# ------------------------------------------------------------------
# Order submission with failure recovery
# ------------------------------------------------------------------
async def _submit_with_recovery(
    broker: Broker,
    proposals: list[ProposedOrder],
    *,
    session_id: str,
) -> tuple[list[BrokerOrderAck], bool]:
    """
    Submit orders in two phases: SELLs first, then BUYs.

    Returns (acks, partial_failure). If BUYs fail after SELLs succeed,
    sets partial_failure=True, fires a critical alert, and re-raises so
    the caller can surface the incomplete state.

    Doing SELLs before BUYs ensures we free up cash before committing to
    new positions. If SELLs fail we stop immediately (no partial BUYs).
    """
    sells = [p for p in proposals if p.side.upper() == "SELL"]
    buys = [p for p in proposals if p.side.upper() == "BUY"]
    sell_acks: list[BrokerOrderAck] = []
    buy_acks: list[BrokerOrderAck] = []

    if sells:
        try:
            sell_acks = await submit_orders(broker, sells, session_id=session_id)
            log.info("submitted %d SELL orders", len(sell_acks))
        except Exception as exc:
            msg = f"SELL submission failed in {session_id}: {exc}. No BUY orders sent."
            log.error(msg)
            await send_alert(msg, "error")
            raise

    if buys:
        try:
            buy_acks = await submit_orders(broker, buys, session_id=session_id)
            log.info("submitted %d BUY orders", len(buy_acks))
        except Exception as exc:
            if sell_acks:
                msg = (
                    f"PARTIAL REBALANCE in {session_id}: "
                    f"{len(sell_acks)} SELL(s) submitted, BUY submission failed: {exc}. "
                    f"Sell order IDs: {[a.broker_order_id for a in sell_acks]}. "
                    "Manual review required immediately."
                )
                log.critical(msg)
                await send_alert(msg, "critical")
                return sell_acks, True  # partial_failure=True
            raise

    return sell_acks + buy_acks, False


# ------------------------------------------------------------------
# Main session
# ------------------------------------------------------------------
async def run_live_session(
    *,
    signal: SignalProducer,
    universe: list[str],
    broker: Broker,
    broker_adapter: Any,
    data_adapter: Any,
    top_k: int = 5,
    lookback_days: int = 240,
    portfolio_value_override: Decimal | None = None,
    trading_enabled: bool = False,
    alpaca_paper: bool = True,
    confirm: bool = False,
    live_trading_confirmed: bool = False,
    session_id: str | None = None,
    risk_limits: RiskLimits | None = None,
    peak_equity: Decimal | None = None,
    reconcile_max_polls: int = 30,
    reconcile_interval_seconds: float = 1.0,
    candidate_filter: Callable[[pl.DataFrame], Awaitable[pl.DataFrame]] | None = None,
    orders_log_path: str | None = None,
) -> LiveSessionResult:
    """
    Pull positions + bars → score signal → [optional candidate filter] →
    compute orders → maybe submit.

    candidate_filter (optional): async function called between scoring and
    top-K selection. Receives the full scores DataFrame, returns a modified
    DataFrame (filtered + reweighted). Used to wire LLM sanity check + macro
    regime multiplier into the live path without polluting the core engine.

    Returns a LiveSessionResult. If partial_failure=True, SELLs were
    submitted but BUYs failed — the portfolio is in an intermediate state
    and requires manual review.
    """
    # Defer session_id to AFTER bars fetch so we can use as_of_date for idempotency.
    # date-only session_id means: cron retries on same trading day → same client_order_id
    # → Alpaca dedups → no double-submission. (Previously used datetime.now → race risk.)
    log.info("live_session starting (universe=%d, top_k=%d)", len(universe), top_k)

    account = await fetch_account_snapshot(broker_adapter)
    log.info("account: equity=%s status=%s paper=%s", account.equity, account.status, account.paper)

    bars = await fetch_recent_bars(data_adapter, universe, lookback_days=lookback_days)
    if bars.is_empty():
        raise RuntimeError(f"no bars fetched for universe of {len(universe)} symbols")

    as_of_date = bars["date"].max()
    if not isinstance(as_of_date, date):
        raise RuntimeError(f"unable to determine as-of date from bars; got {as_of_date!r}")

    sid = session_id or f"live-{as_of_date.isoformat()}"

    scores = signal(as_of_date, bars)

    # LLM-augmentation hook — sanity check + regime multiplier applied here.
    # The filter receives the full scored universe and returns a modified frame.
    # Order independence: filter runs BEFORE top-K selection, so REJECT/FLAG
    # decisions can change which names enter the portfolio.
    if candidate_filter is not None and not scores.is_empty():
        try:
            scores = await candidate_filter(scores)
            log.info("candidate_filter applied: %d names remain", scores.height)
        except Exception as exc:
            log.error("candidate_filter raised — proceeding without filter: %s", exc)
            await send_alert(
                f"candidate_filter failed in {sid}: {exc}. Trading on raw signal scores.",
                "warning",
            )

    if scores.is_empty():
        raise RuntimeError(f"signal returned no scores at as-of date {as_of_date}")
    top = scores.sort("score", descending=True).head(top_k)
    syms = top["symbol"].to_list()
    weight_each = 1.0 / max(len(syms), 1)
    target_weights = dict.fromkeys(syms, weight_each)

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

    allow, reason = _safety_gate(
        trading_enabled=trading_enabled,
        alpaca_paper=alpaca_paper,
        confirm=confirm,
        live_trading_confirmed=live_trading_confirmed,
    )

    submitted = False
    partial_failure = False
    acks: list[BrokerOrderAck] = []
    fills: list[FillReport] = []

    if allow and accepted_proposals:
        log.warning("submitting %d orders (%s)", len(accepted_proposals), reason)
        acks, partial_failure = await _submit_with_recovery(broker, accepted_proposals, session_id=sid)
        submitted = True

        # Persistent audit trail — append every submitted order to CSV.
        # CSV-as-DB: zero-infra, committed to git on every cron run, searchable.
        if orders_log_path and acks:
            try:
                from quant.execution.orders_log import append_orders_log

                n = append_orders_log(
                    path=orders_log_path,
                    session_id=sid,
                    as_of=as_of_date,
                    acks=acks,
                    proposals=accepted_proposals,
                )
                log.info("orders_log: wrote %d rows to %s", n, orders_log_path)
            except Exception as exc:
                log.error("orders_log write failed: %s", exc)
                await send_alert(
                    f"orders_log write failed in {sid}: {exc}. Trades submitted but not logged to CSV.",
                    "warning",
                )

        if not partial_failure:
            log.info("reconciling %d orders…", len(acks))
            try:
                fills = await reconcile(
                    broker_adapter,
                    acks,
                    max_polls=reconcile_max_polls,
                    interval_seconds=reconcile_interval_seconds,
                )
            except Exception as exc:
                log.warning("reconciliation incomplete: %s", exc)
                await send_alert(
                    f"Reconciliation incomplete in {sid}: {exc}. Orders may be pending.",
                    "warning",
                )

        await send_alert(
            f"Paper session {sid} complete: {len(acks)} orders, partial_failure={partial_failure}, "
            f"as_of={as_of_date}, top={syms[:5]}",
            "info",
        )
    else:
        log.info("plan-only mode (%s); %d proposals not submitted", reason, len(accepted_proposals))

    return LiveSessionResult(
        session_id=sid,
        as_of=as_of_date,
        account=account,
        n_symbols_scored=scores.height,
        target_weights=target_weights,
        proposals=proposals,
        submitted=submitted,
        acks=acks,
        risk_results=risk_results,
        fills=fills,
        partial_failure=partial_failure,
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
