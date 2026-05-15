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
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from quant.ml.predict import ModelBundle


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
    Replay an ML trainer's out-of-fold predictions as a signal.

    IMPORTANT — OPTIMISM WARNING:
        OOF (out-of-fold) predictions are NOT the same as true forward-deployed
        predictions. In purged K-fold CV, fold k is predicted by a model trained
        on folds {0..K-1}\{k}. For half of the folds, the training set includes
        data AFTER the prediction date (calendar-time ordering is not preserved
        across folds). The purged-K-fold embargo prevents direct look-ahead, but
        the resulting OOF Sharpe is OPTIMISTIC vs. a model deployed strictly
        forward in time.

        Use `MLBundleSignal` for a true forward simulation: it loads the saved
        artifact bundle and computes predictions on demand at each rebalance date
        using only data available at that point. That is the honest baseline for
        production deployment.

        `MLPredictionsSignal` is appropriate for comparing the ML model to the
        momentum baseline on the SAME historical window, with the understanding
        that the Sharpe is slightly optimistic due to the above.

    The trainer artifact bundle written by `quant.ml.trainer.train` includes
    `oof_predictions.csv` with columns `[date, symbol, prob_neg1, prob_zero,
    prob_pos1, prob_*_calibrated, in_oof, pred_class]`. This signal loads
    that file once at construction time and returns `score = P(+1) - P(-1)`
    for symbols with an in-OOF prediction at or before the as-of date.
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


@dataclass(frozen=True)
class MLBundleSignal:
    """
    Live ML signal — loads a trainer artifact bundle and runs calibrated
    prediction on the price history at each rebalance. Score = P(+1)_cal -
    P(-1)_cal, the same conviction the `recommend()` policy uses.

    Loading is cached the first time the signal is called (the bundle is
    immutable). For sweeps that score the same date many times, only one
    load happens per `MLBundleSignal` instance.

    Use this when you have a freshly trained model and want to backtest
    or paper-trade against ITS recommendations directly — different from
    `MLPredictionsSignal`, which only replays already-computed OOF probs.
    """

    model_dir: str

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        # Lazy import — predict.py drags in lightgbm + sklearn, which we
        # don't want in the import path of the lightweight backtest CLI.
        from quant.ml.predict import load_bundle, recommend

        bundle = _bundle_cache.get(self.model_dir)
        if bundle is None:
            bundle = load_bundle(self.model_dir)
            _bundle_cache[self.model_dir] = bundle

        recs = recommend(bundle, history, as_of=as_of, threshold=0.0)
        if not recs:
            return pl.DataFrame({"symbol": [], "score": []})
        return pl.DataFrame({"symbol": [r.symbol for r in recs], "score": [r.score for r in recs]})


# Module-level cache so repeated calls with the same model_dir avoid the
# heavy load+pickle path. Keyed by absolute model_dir string.
_bundle_cache: dict[str, ModelBundle] = {}


@dataclass(frozen=True)
class ValueSignal:
    """
    Earnings-yield value factor — score = 1 / P/E. Higher score = cheaper
    relative to earnings = stronger long candidate by the value premium.

    Reads a flat CSV produced by `scripts/fetch_fundamentals.py` (schema:
    symbol, price, pe, eps, market_cap, fetched_at_utc). The fundamentals
    snapshot is point-in-time at the moment of fetch — for backtests
    pre-2025 you'd ideally want quarterly historical fundamentals, which
    free FMP doesn't provide. For LIVE recommendations the snapshot is
    accurate; that's the use case this signal is wired for.

    Negative-earnings names (P/E < 0) are excluded — value-via-earnings
    is undefined when there are no earnings to value against.
    """

    fundamentals_csv: str

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        del as_of, history  # value snapshot is current-only
        df = pl.read_csv(self.fundamentals_csv)
        if "symbol" not in df.columns or "pe" not in df.columns:
            raise ValueError(f"{self.fundamentals_csv}: missing required columns symbol, pe")
        return (
            df.with_columns(pl.col("pe").cast(pl.Float64, strict=False))
            .filter(pl.col("pe").is_finite() & (pl.col("pe") > 0))
            .with_columns((1.0 / pl.col("pe")).alias("score"))
            .select(["symbol", "score"])
        )


@dataclass(frozen=True)
class SentimentSignal:
    """
    Aggregated news-sentiment signal — score = mean of per-article Groq
    sentiment scores over the last `lookback_days`. Reads a CSV produced
    by `quant.features.sentiment.fetch_and_score` (schema: symbol, date,
    sentiment_mean, sentiment_count, sentiment_max_abs).

    Why this exists: LLMs (Groq Llama) are good at converting unstructured
    text into a numeric sentiment score; bad at picking stocks directly.
    The honest architecture uses the LLM as a feature extractor, not a
    decision-maker. This signal is the feature in standalone form; the
    `CompositeSignal` blends it with an ML / momentum signal.

    Symbols missing from the CSV (no news in window) are silently
    omitted — the SignalProducer contract is "score whoever you can".
    """

    sentiment_csv: str
    lookback_days: int = 3

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        del history
        df = pl.read_csv(self.sentiment_csv, try_parse_dates=True).with_columns(pl.col("date").cast(pl.Date))
        if "symbol" not in df.columns or "sentiment_mean" not in df.columns:
            raise ValueError(f"{self.sentiment_csv}: missing symbol/sentiment_mean columns")
        from datetime import timedelta as _timedelta

        window_start = as_of - _timedelta(days=self.lookback_days)
        eligible = df.filter((pl.col("date") <= as_of) & (pl.col("date") >= window_start))
        if eligible.is_empty():
            return pl.DataFrame({"symbol": [], "score": []})
        return (
            eligible.group_by("symbol", maintain_order=True)
            .agg(pl.col("sentiment_mean").mean().alias("score"))
            .select(["symbol", "score"])
        )


@dataclass(frozen=True)
class CompositeSignal:
    """
    Late-stage LINEAR BLEND of two SignalProducers:

        composite_score = α * primary_score + β * secondary_score   (α + β = 1)

    ARCHITECTURE HONESTY: This is a parallel signal blend — both children
    produce independent scores which are then averaged. This is NOT the same
    as injecting one signal as a feature into a trained model. If you want
    sentiment as a true ML feature, add it to FEATURE_COLUMNS in trainer.py
    and retrain. That would give the model nonlinear interactions between
    sentiment and price features.

    The current design is appropriate for live production use where retraining
    is expensive: the LLM-scored sentiment acts as a real-time overlay that
    shifts the ML signal's conviction score up or down. It is NOT a substitute
    for adding sentiment to the training set.

    Join semantics: inner-join by default (conservative — names without
    sentiment drop out). Set outer_join=True to impute 0 for missing side.
    """

    primary: object  # SignalProducer
    secondary: object  # SignalProducer
    alpha: float = 0.7  # weight on primary
    beta: float = 0.3  # weight on secondary
    outer_join: bool = False

    def __call__(self, as_of: date, history: pl.DataFrame) -> pl.DataFrame:
        if abs(self.alpha + self.beta - 1.0) > 1e-9:
            raise ValueError(f"alpha+beta must equal 1.0, got {self.alpha + self.beta}")
        a = self.primary(as_of, history)  # type: ignore[operator]
        b = self.secondary(as_of, history)  # type: ignore[operator]
        if a.is_empty() and b.is_empty():
            return pl.DataFrame({"symbol": [], "score": []})

        a = a.rename({"score": "_a"})
        b = b.rename({"score": "_b"})
        # polars 1.x renamed "outer" → "full"; we want full outer join here.
        how: str = "full" if self.outer_join else "inner"
        joined = a.join(b, on="symbol", how=how)
        if self.outer_join:
            joined = joined.with_columns(
                pl.col("_a").fill_null(0.0).alias("_a"),
                pl.col("_b").fill_null(0.0).alias("_b"),
            )
        return joined.with_columns(
            (pl.col("_a") * self.alpha + pl.col("_b") * self.beta).alias("score")
        ).select(["symbol", "score"])


__all__ = [
    "CompositeSignal",
    "LowVolSignal",
    "MLBundleSignal",
    "MLPredictionsSignal",
    "MeanReversionSignal",
    "MomentumSignal",
    "SentimentSignal",
    "ValueSignal",
]
