"""
Built-in SignalProducer implementations for the backtest CLI.

The CLI is agnostic to the signal source — it takes any `SignalProducer`
(callable returning `DataFrame[symbol, score]`). The producers here are
baselines you can point at without training anything. Each is a sensible
null hypothesis the ML model must beat:

- `MomentumSignal(lookback_days)` — score = trailing total return.
  The classic factor; long-horizon (~6m) it's a documented anomaly.
- `LowVolSignal(lookback_days)` — score = -trailing volatility. The
  low-volatility anomaly: low-vol stocks have historically delivered
  comparable returns at lower risk than the cap-weighted index.
- `MeanReversionSignal(lookback_days)` — score = -trailing return on
  short horizons. Short-term reversals pay because liquidity providers
  earn the spread that displaced prices revert through.

These three are roughly *uncorrelated* over the same window — momentum
buys recent winners, mean-reversion buys recent losers, low-vol picks
boring names. Including all three in a PBO sweep is more diagnostic than
a momentum-only grid because the trial pool now spans different return
sources, not just one knob.

The ML-backed producer (load an MLflow-logged LightGBM ensemble, call
`predict_proba`, project to a long-score) is intentionally not wired here;
it requires a trained registry, which is a pipeline concern, not a CLI one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl


@dataclass(frozen=True)
class MomentumSignal:
    """Trailing-return momentum. Higher score = stronger uptrend."""

    lookback_days: int = 126  # ~6 trading months

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        if history.is_empty():
            return pl.DataFrame({"symbol": [], "score": []})

        hist = history.filter(pl.col("date") <= as_of).sort(["symbol", "date"])
        scores = (
            hist.group_by("symbol", maintain_order=True)
            .agg(pl.col("adj_close").tail(self.lookback_days).alias("_tail"))
            .with_columns(
                pl.col("_tail").list.last().alias("_p_end"),
                pl.col("_tail").list.first().alias("_p_start"),
                pl.col("_tail").list.len().alias("_n"),
            )
            .filter((pl.col("_n") >= self.lookback_days) & (pl.col("_p_start") > 0))
            .with_columns((pl.col("_p_end") / pl.col("_p_start") - 1.0).alias("score"))
            .select(["symbol", "score"])
        )
        return scores


@dataclass(frozen=True)
class LowVolSignal:
    """
    Low-volatility anomaly. Score = -stddev(log_returns over lookback).
    Higher score = lower realized volatility = stronger long candidate.
    """

    lookback_days: int = 126

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        if history.is_empty():
            return pl.DataFrame({"symbol": [], "score": []})

        hist = history.filter(pl.col("date") <= as_of).sort(["symbol", "date"])
        # Per symbol: take last N+1 closes, compute log returns, take std.
        # We need lookback_days+1 prices to get lookback_days returns.
        scores = (
            hist.group_by("symbol", maintain_order=True)
            .agg(pl.col("adj_close").tail(self.lookback_days + 1).alias("_tail"))
            .with_columns(pl.col("_tail").list.len().alias("_n"))
            .filter(pl.col("_n") >= self.lookback_days + 1)
            .with_columns(
                pl.col("_tail").map_elements(_neg_log_return_std, return_dtype=pl.Float64).alias("score")
            )
            .select(["symbol", "score"])
        )
        return scores


def _neg_log_return_std(prices: list[float]) -> float:
    """Negative std of log returns. Higher = lower vol = preferred."""
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size < 2 or (arr <= 0).any():
        return float("nan")
    rets = np.diff(np.log(arr))
    if rets.size < 2:
        return float("nan")
    return -float(np.std(rets, ddof=1))


@dataclass(frozen=True)
class MeanReversionSignal:
    """
    Short-horizon reversal. Score = -trailing total return over lookback.
    Higher score = recent loser = stronger long candidate. The spec
    typically uses a short window (1–2 weeks); we keep `lookback_days`
    parametric so the same dataclass can be reused.
    """

    lookback_days: int = 5

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        if history.is_empty():
            return pl.DataFrame({"symbol": [], "score": []})

        hist = history.filter(pl.col("date") <= as_of).sort(["symbol", "date"])
        scores = (
            hist.group_by("symbol", maintain_order=True)
            .agg(pl.col("adj_close").tail(self.lookback_days).alias("_tail"))
            .with_columns(
                pl.col("_tail").list.last().alias("_p_end"),
                pl.col("_tail").list.first().alias("_p_start"),
                pl.col("_tail").list.len().alias("_n"),
            )
            .filter((pl.col("_n") >= self.lookback_days) & (pl.col("_p_start") > 0))
            .with_columns((-1.0 * (pl.col("_p_end") / pl.col("_p_start") - 1.0)).alias("score"))
            .select(["symbol", "score"])
        )
        return scores


__all__ = ["LowVolSignal", "MeanReversionSignal", "MomentumSignal"]
