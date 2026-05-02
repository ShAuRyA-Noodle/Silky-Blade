"""Unit tests for the backtest sweep runner (multi-config + cross-config PBO)."""

from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from quant.backtest.engine import WalkForwardConfig
from quant.backtest.runner import SignalSpec, StatsSpec
from quant.backtest.sweep import (
    SweepConfig,
    SweepRunSpec,
    load_sweep_config,
    run_sweep,
)


# ------------------------------------------------------------------
# Fixtures — synthetic-but-real GBM prices (mirrors test_backtest_runner)
# ------------------------------------------------------------------
def _write_prices_csv(path: Path, n_days: int = 800, n_symbols: int = 10, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    start = date(2019, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days) if (start + timedelta(days=i)).weekday() < 5]
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    drifts = np.linspace(-0.0003, 0.0009, n_symbols)
    vol = 0.012

    rows: list[tuple[str, str, float]] = []
    for s_idx, sym in enumerate(symbols):
        price = 100.0
        for d in dates:
            r = float(rng.normal(drifts[s_idx], vol))
            price *= math.exp(r)
            rows.append((d.isoformat(), sym, round(price, 4)))

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "symbol", "adj_close"])
        w.writerows(rows)
    return path


@pytest.fixture()
def prices_csv(tmp_path: Path) -> Path:
    return _write_prices_csv(tmp_path / "prices.csv")


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------
def test_load_sweep_config_json_roundtrip(tmp_path: Path) -> None:
    body = {
        "name": "unit_sweep",
        "prices_csv": "prices.csv",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "output_dir": str(tmp_path / "out"),
        "walk_forward_base": {"train_days": 60, "test_days": 5, "top_k": 3, "cost_bps": 2.0},
        "stats_base": {"sharpes_std": 0.4},
        "n_slices": 4,
        "runs": [
            {"name": "a", "signal": {"kind": "momentum", "params": {"lookback_days": 20}}},
            {"name": "b", "signal": {"kind": "momentum", "params": {"lookback_days": 40}}},
        ],
    }
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(body), encoding="utf-8")

    cfg = load_sweep_config(p)
    assert cfg.name == "unit_sweep"
    assert len(cfg.runs) == 2
    assert cfg.runs[0].signal.params["lookback_days"] == 20
    assert cfg.runs[1].signal.params["lookback_days"] == 40
    assert cfg.n_slices == 4


def test_load_sweep_config_rejects_empty_runs(tmp_path: Path) -> None:
    body = {
        "name": "x",
        "prices_csv": "p.csv",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "runs": [],
    }
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'runs'"):
        load_sweep_config(p)


def test_load_sweep_config_rejects_duplicate_run_names(tmp_path: Path) -> None:
    body = {
        "name": "x",
        "prices_csv": "p.csv",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "runs": [
            {"name": "a", "signal": {"kind": "momentum", "params": {"lookback_days": 20}}},
            {"name": "a", "signal": {"kind": "momentum", "params": {"lookback_days": 40}}},
        ],
    }
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate run name"):
        load_sweep_config(p)


# ------------------------------------------------------------------
# Cadence-mismatch guard
# ------------------------------------------------------------------
def test_run_sweep_rejects_cadence_override(tmp_path: Path, prices_csv: Path) -> None:
    cfg = SweepConfig(
        name="cadence_clash",
        prices_csv=str(prices_csv),
        start_date=date(2019, 1, 1),
        end_date=date(2021, 12, 31),
        output_dir=str(tmp_path / "art"),
        walk_forward_base=WalkForwardConfig(train_days=60, test_days=5, top_k=3, cost_bps=2.0),
        stats_base=StatsSpec(n_trials=1, sharpes_std=0.3),
        runs=(
            SweepRunSpec(
                name="a", signal=SignalSpec(kind="momentum", params={"lookback_days": 20})
            ),
            SweepRunSpec(
                name="b",
                signal=SignalSpec(kind="momentum", params={"lookback_days": 40}),
                walk_forward_overrides={"test_days": 10},
            ),
        ),
        n_slices=4,
    )
    with pytest.raises(ValueError, match="train_days/test_days must match"):
        run_sweep(cfg)


def test_run_sweep_requires_at_least_two_runs(tmp_path: Path, prices_csv: Path) -> None:
    cfg = SweepConfig(
        name="too_small",
        prices_csv=str(prices_csv),
        start_date=date(2019, 1, 1),
        end_date=date(2021, 12, 31),
        output_dir=str(tmp_path / "art"),
        walk_forward_base=WalkForwardConfig(train_days=60, test_days=5, top_k=3, cost_bps=2.0),
        stats_base=StatsSpec(n_trials=1, sharpes_std=0.3),
        runs=(SweepRunSpec(name="solo", signal=SignalSpec(kind="momentum", params={"lookback_days": 20})),),
        n_slices=4,
    )
    with pytest.raises(ValueError, match=">= 2 runs"):
        run_sweep(cfg)


# ------------------------------------------------------------------
# End-to-end sweep on synthetic-but-real GBM
# ------------------------------------------------------------------
def test_run_sweep_writes_full_artifact_bundle(tmp_path: Path, prices_csv: Path) -> None:
    cfg = SweepConfig(
        name="e2e_sweep",
        prices_csv=str(prices_csv),
        start_date=date(2019, 1, 1),
        end_date=date(2021, 12, 31),
        output_dir=str(tmp_path / "art"),
        walk_forward_base=WalkForwardConfig(train_days=60, test_days=5, top_k=3, cost_bps=2.0),
        stats_base=StatsSpec(n_trials=1, sharpes_std=0.3),
        runs=(
            SweepRunSpec(name="mom_20", signal=SignalSpec(kind="momentum", params={"lookback_days": 20})),
            SweepRunSpec(name="mom_40", signal=SignalSpec(kind="momentum", params={"lookback_days": 40})),
            SweepRunSpec(name="mom_60", signal=SignalSpec(kind="momentum", params={"lookback_days": 60})),
            SweepRunSpec(name="mom_80", signal=SignalSpec(kind="momentum", params={"lookback_days": 80})),
        ),
        n_slices=4,
    )
    report = run_sweep(cfg)

    out_dir = Path(report["artifacts"]["dir"])
    for name in ("sweep_report.json", "sweep.config.snapshot.json", "manifest.json"):
        assert (out_dir / name).exists(), f"missing sweep artifact: {name}"

    # Cross-config PBO must be in [0, 1]
    assert 0.0 <= report["pbo"] <= 1.0
    assert report["n_configs"] == 4
    assert report["cscv_S"] == 4
    assert report["cscv_n_trials"] > 0
    assert report["n_observations_per_config"] >= cfg.n_slices

    # Each run gets full 4-file bundle
    for r in report["runs"]:
        run_dir = Path(r["artifacts_dir"])
        for name in ("report.json", "equity_curve.csv", "manifest.json", "config.snapshot.json"):
            assert (run_dir / name).exists(), f"missing per-run artifact: {run_dir}/{name}"

    # Each per-run DSR was computed with n_trials = n_runs (sweep auto-sets this).
    for r in report["runs"]:
        with open(Path(r["artifacts_dir"]) / "report.json") as fh:
            sub = json.load(fh)
        assert sub["metrics"]["dsr_n_trials"] == 4

    # Sweep manifest carries a sweep_config_hash.
    with open(out_dir / "manifest.json") as fh:
        manifest = json.load(fh)
    assert len(manifest["sweep_config_hash"]) == 64
    assert len(manifest["data_fingerprint"]) == 64
