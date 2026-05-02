"""
Walk-forward backtest engine.

Given a signal producer, a price history, and a capital allocation rule,
step forward through time in fixed train/test windows, rebalance on each
test window, and record portfolio equity.

This is intentionally simple — the real-world execution path is the live
OrderService. This engine is for offline strategy evaluation:
    - daily bar granularity
    - position sizing: equal-weight top-k long signals
    - costs: fixed bps per dollar traded
    - no shorting, no leverage (add later if needed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import numpy as np
import polars as pl


class SignalProducer(Protocol):
    """Produce a DataFrame with columns [symbol, score] for a given as-of date."""

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame: ...


@dataclass
class WalkForwardConfig:
    train_days: int = 252
    test_days: int = 21
    top_k: int = 20
    cost_bps: float = 5.0  # one-way, 5 bps = 0.05%
    initial_capital: float = 100_000.0


@dataclass
class BacktestResult:
    equity_curve: pl.DataFrame
    per_period_returns: np.ndarray
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float
    metadata: dict[str, object] = field(default_factory=dict)


def walk_forward(
    prices: pl.DataFrame,
    produce_signals: SignalProducer,
    config: WalkForwardConfig,
) -> BacktestResult:
    """
    prices: polars DataFrame with columns ['date', 'symbol', 'adj_close'].
    produce_signals: callable (as_of, history_up_to_as_of) -> DataFrame[symbol, score].
    """
    if not {"date", "symbol", "adj_close"}.issubset(prices.columns):
        raise ValueError("prices must have columns [date, symbol, adj_close]")

    prices = prices.sort(["date", "symbol"])
    dates = prices.get_column("date").unique().sort().to_list()
    if len(dates) < config.train_days + config.test_days:
        raise ValueError(f"not enough dates: need {config.train_days + config.test_days}, got {len(dates)}")

    equity = config.initial_capital
    curve: list[tuple[date, float]] = []
    weights_prev: dict[str, float] = {}
    turnover_sum = 0.0

    i = config.train_days
    while i + config.test_days <= len(dates):
        rebalance_date = dates[i - 1]
        test_end = dates[min(i + config.test_days - 1, len(dates) - 1)]

        history = prices.filter(pl.col("date") <= rebalance_date)
        sigs = produce_signals(rebalance_date, history)
        if sigs.is_empty():
            i += config.test_days
            continue
        top = sigs.sort("score", descending=True).head(config.top_k)
        syms = top.get_column("symbol").to_list()
        if not syms:
            i += config.test_days
            continue
        w_new = dict.fromkeys(syms, 1.0 / len(syms))

        turnover = _turnover(weights_prev, w_new)
        turnover_sum += turnover
        equity *= 1.0 - turnover * (config.cost_bps / 10_000.0)

        test_slice = prices.filter((pl.col("date") > rebalance_date) & (pl.col("date") <= test_end))
        period_ret = _portfolio_return(test_slice, w_new)
        equity *= 1.0 + period_ret
        curve.append((test_end, equity))
        weights_prev = w_new

        i += config.test_days

    if not curve:
        raise RuntimeError("no test windows produced returns — check data")

    eq_df = pl.DataFrame({"date": [d for d, _ in curve], "equity": [e for _, e in curve]})
    eq = np.asarray([e for _, e in curve], dtype=float)
    rets = np.diff(eq) / eq[:-1] if eq.size > 1 else np.array([0.0])
    total_return = eq[-1] / config.initial_capital - 1.0
    years = len(curve) * config.test_days / 252.0
    ann_ret = (1 + total_return) ** (1 / max(years, 1e-9)) - 1
    ann_vol = float(rets.std(ddof=1) * np.sqrt(252 / config.test_days)) if rets.size > 1 else 0.0
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")
    dd = _max_drawdown(eq)

    return BacktestResult(
        equity_curve=eq_df,
        per_period_returns=rets,
        total_return=float(total_return),
        annualized_return=float(ann_ret),
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=dd,
        turnover=turnover_sum,
        metadata={
            "n_rebalances": len(curve),
            "train_days": config.train_days,
            "test_days": config.test_days,
            "top_k": config.top_k,
            "cost_bps": config.cost_bps,
        },
    )


def _turnover(prev: dict[str, float], new: dict[str, float]) -> float:
    symbols = set(prev) | set(new)
    return 0.5 * sum(abs(new.get(s, 0.0) - prev.get(s, 0.0)) for s in symbols)


def _portfolio_return(slice_df: pl.DataFrame, weights: dict[str, float]) -> float:
    if slice_df.is_empty():
        return 0.0
    # Per-symbol return from first to last close in the slice.
    # `maintain_order=True` keeps group iteration order stable. Note: this alone
    # does NOT make the run bit-exact — Polars' multi-threaded reductions still
    # produce ULP-level drift (~1e-15) in float64 sums. Bit-exact reproducibility
    # requires `POLARS_MAX_THREADS=1` in the environment. See REPRODUCE.md.
    agg = slice_df.group_by("symbol", maintain_order=True).agg(
        [pl.col("adj_close").first().alias("p0"), pl.col("adj_close").last().alias("p1")]
    )
    total = 0.0
    for row in agg.iter_rows(named=True):
        w = weights.get(row["symbol"], 0.0)
        if w == 0 or row["p0"] in (0, None):
            continue
        total += w * (float(row["p1"]) / float(row["p0"]) - 1.0)
    return total


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max())


__all__ = [
    "BacktestResult",
    "SignalProducer",
    "WalkForwardConfig",
    "walk_forward",
]
