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
class MLPredictionsSignal:
    """
    Replay an ML trainer's out-of-fold predictions as a long-only signal.

    The trainer artifact bundle written by `quant.ml.trainer.train` includes
    `oof_predictions.csv` with columns `[date, symbol, prob_neg1, prob_zero,
    prob_pos1, prob_*_calibrated, in_oof, pred_class]`. This signal loads
    that file once at construction time, indexes by `(date, symbol)`, and on
    each rebalance returns `score = P(+1) - P(-1)` for symbols with an
    in-OOF prediction available at the as-of date.

    The score is a long-short conviction; we use only the long side here
    (top-k highest scores). Calibrated probabilities are preferred when
    present — calibration is a property of the predictor, not the signal,
    and using the calibrated columns is the honest default.

    Why this is the bridge that closes the loop: the LightGBM model trained
    on triple-barrier labels with purged K-fold CV + embargo is the central
    ML claim in TRUST.md. Before this signal existed, that model never
    produced a Sharpe number anyone could compare to the momentum baseline.
    With it, the same walk-forward engine that prices the baseline also
    prices the model — apples-to-apples.
    """

    predictions_csv: str
    use_calibrated: bool = True

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        # `history` is unused — predictions were generated against the same
        # price panel during training, so we trust them as-of `as_of`.
        del history
        oof = pl.read_csv(self.predictions_csv, try_parse_dates=True).with_columns(
            pl.col("date").cast(pl.Date)
        )
        # Use the latest available prediction at or before `as_of`. The
        # walk-forward engine rebalances at fixed cadences; the OOF panel
        # has predictions only on label-anchor dates.
        eligible = oof.filter((pl.col("date") <= as_of) & pl.col("in_oof"))
        if eligible.is_empty():
            return pl.DataFrame({"symbol": [], "score": []})

        # Per symbol, take the most recent in-OOF prediction.
        latest = eligible.sort(["symbol", "date"]).group_by("symbol", maintain_order=True).tail(1)

        prob_pos = "prob_pos1_calibrated" if self.use_calibrated else "prob_pos1"
        prob_neg = "prob_neg1_calibrated" if self.use_calibrated else "prob_neg1"
        if prob_pos not in latest.columns or prob_neg not in latest.columns:
            raise ValueError(
                f"{self.predictions_csv}: missing columns {prob_pos}/{prob_neg} — "
                "regenerate the artifact with the current trainer."
            )

        return latest.select(
            pl.col("symbol"),
            (pl.col(prob_pos) - pl.col(prob_neg)).alias("score"),
        )


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


__all__ = ["LowVolSignal", "MLPredictionsSignal", "MeanReversionSignal", "MomentumSignal"]
