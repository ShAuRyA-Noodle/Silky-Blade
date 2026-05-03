"""Unit tests for the built-in SignalProducer baselines."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quant.backtest.runner import SignalSpec, build_signal
from quant.backtest.signals import LowVolSignal, MeanReversionSignal


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
def test_build_signal_low_vol() -> None:
    s = build_signal(SignalSpec(kind="low_vol", params={"lookback_days": 30}))
    assert isinstance(s, LowVolSignal)
    assert s.lookback_days == 30


def test_build_signal_value(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[name-defined]
    from quant.backtest.signals import ValueSignal

    p = tmp_path / "fund.csv"  # type: ignore[union-attr]
    p.write_text("symbol,price,pe,eps,market_cap,fetched_at_utc\nAAPL,200,30,6.5,1e12,2026-05-01\n")
    s = build_signal(SignalSpec(kind="value", params={"fundamentals_csv": str(p)}))
    assert isinstance(s, ValueSignal)


def test_value_signal_inverse_pe(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[name-defined]
    from datetime import date as _date

    from quant.backtest.signals import ValueSignal

    p = tmp_path / "fund.csv"  # type: ignore[union-attr]
    p.write_text(
        "symbol,price,pe,eps,market_cap,fetched_at_utc\n"
        "CHEAP,10,5,2,1e9,2026-05-01\n"
        "EXPENSIVE,500,50,10,1e12,2026-05-01\n"
        "LOSS,1,-5,-0.2,1e8,2026-05-01\n"  # negative-earnings excluded
    )
    sig = ValueSignal(fundamentals_csv=str(p))
    scores = sig(_date(2026, 5, 1), pl_empty_history())
    got = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True)}
    assert "LOSS" not in got
    assert got["CHEAP"] > got["EXPENSIVE"]  # 1/5 > 1/50
    assert abs(got["CHEAP"] - 0.2) < 1e-9
    assert abs(got["EXPENSIVE"] - 0.02) < 1e-9


def pl_empty_history() -> pl.DataFrame:
    return pl.DataFrame({"date": [], "symbol": [], "adj_close": []})


def test_build_signal_mean_reversion() -> None:
    s = build_signal(SignalSpec(kind="mean_reversion", params={"lookback_days": 7}))
    assert isinstance(s, MeanReversionSignal)
    assert s.lookback_days == 7


# ------------------------------------------------------------------
# LowVolSignal
# ------------------------------------------------------------------
def _two_path_history(
    quiet_drift: float = 0.0005,
    quiet_vol: float = 0.005,
    loud_drift: float = 0.0005,
    loud_vol: float = 0.05,
    n_days: int = 80,
    seed: int = 1,
) -> pl.DataFrame:
    """Two real-arithmetic GBM paths: one quiet, one loud."""
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rows: list[dict[str, object]] = []
    p_q, p_l = 100.0, 100.0
    for d in dates:
        p_q *= math.exp(float(rng.normal(quiet_drift, quiet_vol)))
        p_l *= math.exp(float(rng.normal(loud_drift, loud_vol)))
        rows.append({"date": d, "symbol": "QUIET", "adj_close": p_q})
        rows.append({"date": d, "symbol": "LOUD", "adj_close": p_l})
    return pl.DataFrame(rows)


def test_low_vol_signal_prefers_quiet_symbol() -> None:
    hist = _two_path_history()
    sig = LowVolSignal(lookback_days=60)
    scores = sig(date(2022, 3, 21), hist)
    got = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True)}
    # Score is -std(log returns) — both negative, but the quiet path is closer
    # to zero (smaller magnitude of negation), so its score is HIGHER.
    assert got["QUIET"] > got["LOUD"]


def test_low_vol_signal_drops_symbols_below_lookback_window() -> None:
    hist = _two_path_history(n_days=20)
    sig = LowVolSignal(lookback_days=60)
    scores = sig(date(2022, 1, 19), hist)
    # Window not satisfied → empty result
    assert scores.height == 0


def test_low_vol_signal_returns_empty_on_empty_history() -> None:
    sig = LowVolSignal(lookback_days=20)
    out = sig(date(2024, 1, 1), pl.DataFrame({"date": [], "symbol": [], "adj_close": []}))
    assert out.height == 0
    assert set(out.columns) == {"symbol", "score"}


# ------------------------------------------------------------------
# MeanReversionSignal
# ------------------------------------------------------------------
def test_mean_reversion_signal_prefers_recent_loser() -> None:
    """Loser over the lookback should outrank winner."""
    dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(15)]
    rows: list[dict[str, object]] = []
    for d in dates:
        i = (d - date(2022, 1, 1)).days
        rows.append({"date": d, "symbol": "WIN", "adj_close": 100.0 * (1 + 0.01) ** i})
        rows.append({"date": d, "symbol": "LOSS", "adj_close": 100.0 * (1 - 0.01) ** i})
    hist = pl.DataFrame(rows)

    sig = MeanReversionSignal(lookback_days=10)
    scores = sig(dates[-1], hist)
    got = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True)}
    # Score = -trailing_return → LOSS should be > WIN
    assert got["LOSS"] > got["WIN"]
    # And LOSS's score should be positive (its trailing return was negative).
    assert got["LOSS"] > 0
    assert got["WIN"] < 0


def test_mean_reversion_signal_handles_short_lookback() -> None:
    """Default lookback is 5 days — verify the path is exercised."""
    dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(8)]
    rows: list[dict[str, object]] = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "symbol": "X", "adj_close": 100.0 + i})
    hist = pl.DataFrame(rows)

    sig = MeanReversionSignal()  # default lookback=5
    scores = sig(dates[-1], hist)
    assert scores.height == 1
    # Strictly increasing prices → negative score
    assert scores["score"][0] < 0


# ------------------------------------------------------------------
# Cross-signal: scores stay finite + alignable across producers
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind,params",
    [
        ("momentum", {"lookback_days": 30}),
        ("low_vol", {"lookback_days": 30}),
        ("mean_reversion", {"lookback_days": 5}),
    ],
)
def test_signals_emit_well_formed_scores(kind: str, params: dict[str, int]) -> None:
    """Every built-in signal returns a `{symbol, score}` frame with finite scores."""
    hist = _two_path_history(n_days=80)
    sig = build_signal(SignalSpec(kind=kind, params=params))
    scores = sig(date(2022, 3, 21), hist)
    assert set(scores.columns) == {"symbol", "score"}
    if scores.height > 0:
        # Every score must be finite (no NaN, no inf).
        arr = scores["score"].to_numpy()
        assert np.all(np.isfinite(arr)), f"{kind} produced non-finite scores"
