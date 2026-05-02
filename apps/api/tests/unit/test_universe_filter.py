"""Tests for walk_forward's universe_filter integration."""

from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant.backtest.engine import WalkForwardConfig, walk_forward
from quant.backtest.signals import MomentumSignal


def _write_prices(path: Path, n_days: int = 400, n_symbols: int = 6, seed: int = 5) -> Path:
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days) if (start + timedelta(days=i)).weekday() < 5]
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    drifts = np.linspace(-0.0002, 0.0008, n_symbols)
    rows: list[tuple[str, str, float]] = []
    for s_idx, sym in enumerate(symbols):
        p = 100.0
        for d in dates:
            p *= math.exp(float(rng.normal(drifts[s_idx], 0.012)))
            rows.append((d.isoformat(), sym, round(p, 4)))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "symbol", "adj_close"])
        w.writerows(rows)
    return path


@pytest.fixture()
def prices(tmp_path: Path) -> pl.DataFrame:
    p = _write_prices(tmp_path / "prices.csv")
    return pl.read_csv(p, try_parse_dates=True).with_columns(pl.col("date").cast(pl.Date))


# ------------------------------------------------------------------
# Filter narrows the eligible universe
# ------------------------------------------------------------------
def test_universe_filter_excludes_symbols(prices: pl.DataFrame) -> None:
    """Filter that excludes some symbols should change the result."""
    sig = MomentumSignal(lookback_days=20)
    cfg = WalkForwardConfig(train_days=40, test_days=5, top_k=2, cost_bps=2.0)

    # Allow only 2 symbols → those are picked regardless of momentum rank.
    def _filter(_d: date) -> set[str]:
        return {"S00", "S01"}

    res_filtered = walk_forward(prices, sig, cfg, universe_filter=_filter)
    res_raw = walk_forward(prices, sig, cfg)

    # Filtered run is a strict subset of the universe, so its results must
    # differ from the raw run (with overwhelming probability on real-arithmetic
    # GBM with 6 symbols).
    assert res_filtered.equity_curve.height > 0
    assert res_raw.equity_curve.height > 0
    assert res_filtered.total_return != res_raw.total_return


def test_universe_filter_skips_rebalance_when_eligible_set_empty(prices: pl.DataFrame) -> None:
    """Empty eligible set on every date → no rebalances → engine should raise."""
    sig = MomentumSignal(lookback_days=20)
    cfg = WalkForwardConfig(train_days=40, test_days=5, top_k=2, cost_bps=2.0)

    def _empty_filter(_d: date) -> set[str]:
        return set()

    with pytest.raises(RuntimeError, match="no test windows"):
        walk_forward(prices, sig, cfg, universe_filter=_empty_filter)


def test_universe_filter_none_means_no_restriction(prices: pl.DataFrame) -> None:
    """Passing None should produce identical output to omitting the kwarg."""
    sig = MomentumSignal(lookback_days=20)
    cfg = WalkForwardConfig(train_days=40, test_days=5, top_k=3, cost_bps=2.0)

    res_omitted = walk_forward(prices, sig, cfg)
    res_none = walk_forward(prices, sig, cfg, universe_filter=None)
    assert res_omitted.total_return == res_none.total_return
    assert res_omitted.sharpe == res_none.sharpe


def test_universe_filter_can_change_per_date(prices: pl.DataFrame) -> None:
    """Filter that varies by date should not crash and should keep running."""
    sig = MomentumSignal(lookback_days=20)
    cfg = WalkForwardConfig(train_days=40, test_days=5, top_k=2, cost_bps=2.0)
    boundary = date(2020, 6, 1)

    def _date_dependent(d: date) -> set[str]:
        if d < boundary:
            return {"S00", "S01", "S02"}
        return {"S03", "S04", "S05"}

    res = walk_forward(prices, sig, cfg, universe_filter=_date_dependent)
    assert res.equity_curve.height > 0
    assert res.metadata["n_rebalances"] > 0
