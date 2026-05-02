"""
Multi-config backtest sweeps with cross-config Probability of Backtest
Overfitting (PBO).

A single run gets a Deflated Sharpe Ratio (DSR) — that adjusts the observed
Sharpe for the size of the trial pool you searched. PBO is the complementary
diagnostic: given a *sweep* of competing configs over the same data and time
window, how often does the in-sample winner end up below the OOS median?
PBO needs the full N-config returns matrix; you cannot compute it from one
run. This module is the gap-closer.

Schema (YAML):

    name: <sweep_name>
    prices_csv: <path>
    start_date: <iso>
    end_date:   <iso>
    output_dir: <path>

    walk_forward_base: { train_days, test_days, top_k, cost_bps,
                         initial_capital }
    stats_base:        { sharpes_std }   # n_trials is forced to len(runs)
    runs:
      - name: <run_name>
        signal: { kind, params }
        walk_forward_overrides: { ... }   # optional
        stats_overrides:        { ... }   # optional

Two enforced invariants — both kept simple on purpose so PBO is
mathematically meaningful:

1. Every run must share the same prices_csv / start_date / end_date so the
   per-period returns line up on the same time grid.
2. Every run must share the same `train_days` and `test_days` (the walk-
   forward cadence). Per-config `top_k` and `cost_bps` overrides are fine —
   those don't shift rebalance dates. Allowing different test windows would
   mean inner-joining on dates and silently dropping observations, which
   biases the PBO estimate. Fail loud instead.

Output bundle (`<output_dir>/<sweep_name>/`):

    sweep_report.json          — per-run metrics + cross-config PBO
    sweep.config.snapshot.json — exact sweep config used
    manifest.json              — code_sha, sweep_config_hash, data_fingerprint
    per_run/<run_name>/        — full 4-file artifact bundle from runner.py
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant.backtest.engine import WalkForwardConfig
from quant.backtest.reproducibility import build_manifest
from quant.backtest.runner import (
    RunConfig,
    SignalSpec,
    StatsSpec,
    load_prices_csv,
    run_backtest,
)
from quant.backtest.statistics import probability_of_backtest_overfitting

log = logging.getLogger("quant.backtest.sweep")


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass(frozen=True)
class SweepRunSpec:
    """One row in the sweep — a single config that varies signal/knobs."""

    name: str
    signal: SignalSpec = field(default_factory=SignalSpec)
    walk_forward_overrides: dict[str, Any] = field(default_factory=dict)
    stats_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepConfig:
    name: str
    prices_csv: str
    start_date: date
    end_date: date
    output_dir: str
    walk_forward_base: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    stats_base: StatsSpec = field(default_factory=StatsSpec)
    runs: tuple[SweepRunSpec, ...] = ()
    n_slices: int = 8


# ------------------------------------------------------------------
# Config I/O
# ------------------------------------------------------------------
def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load a YAML or JSON sweep config."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data: dict[str, Any]
    if p.suffix in (".yml", ".yaml"):
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(raw)
        if not isinstance(loaded, dict):
            raise ValueError(f"{p}: expected a YAML mapping at the top level")
        data = loaded
    else:
        data = json.loads(raw)
    return _coerce_sweep_config(data)


def _as_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


def _coerce_sweep_config(data: dict[str, Any]) -> SweepConfig:
    wf_base = data.get("walk_forward_base") or {}
    stats_base = data.get("stats_base") or {}
    raw_runs = data.get("runs") or []
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("sweep config must define a non-empty 'runs' list")

    runs: list[SweepRunSpec] = []
    seen: set[str] = set()
    for r in raw_runs:
        if not isinstance(r, dict):
            raise ValueError(f"runs[*] must be mappings, got {type(r).__name__}")
        rname = str(r["name"])
        if rname in seen:
            raise ValueError(f"duplicate run name: {rname!r}")
        seen.add(rname)
        sig = r.get("signal") or {}
        runs.append(
            SweepRunSpec(
                name=rname,
                signal=SignalSpec(
                    kind=str(sig.get("kind", "momentum")),
                    params=dict(sig.get("params", {})),
                ),
                walk_forward_overrides=dict(r.get("walk_forward_overrides", {})),
                stats_overrides=dict(r.get("stats_overrides", {})),
            )
        )

    return SweepConfig(
        name=str(data["name"]),
        prices_csv=str(data["prices_csv"]),
        start_date=_as_date(data["start_date"]),
        end_date=_as_date(data["end_date"]),
        output_dir=str(data.get("output_dir", "./artifacts/backtest_sweep")),
        walk_forward_base=WalkForwardConfig(
            train_days=int(wf_base.get("train_days", 252)),
            test_days=int(wf_base.get("test_days", 21)),
            top_k=int(wf_base.get("top_k", 20)),
            cost_bps=float(wf_base.get("cost_bps", 5.0)),
            initial_capital=float(wf_base.get("initial_capital", 100_000.0)),
        ),
        stats_base=StatsSpec(
            n_trials=int(stats_base.get("n_trials", 1)),
            sharpes_std=float(stats_base.get("sharpes_std", 0.5)),
        ),
        runs=tuple(runs),
        n_slices=int(data.get("n_slices", 8)),
    )


# ------------------------------------------------------------------
# Resolve one sweep row into a full RunConfig
# ------------------------------------------------------------------
def _resolve_run(sweep: SweepConfig, run: SweepRunSpec, *, n_runs: int) -> RunConfig:
    """
    Apply per-run overrides on top of the sweep base, and force
    `stats.n_trials = n_runs` — the user's grid IS the trial pool, anything
    else under-counts the selection bias.
    """
    wf_base = sweep.walk_forward_base
    wf = WalkForwardConfig(
        train_days=int(run.walk_forward_overrides.get("train_days", wf_base.train_days)),
        test_days=int(run.walk_forward_overrides.get("test_days", wf_base.test_days)),
        top_k=int(run.walk_forward_overrides.get("top_k", wf_base.top_k)),
        cost_bps=float(run.walk_forward_overrides.get("cost_bps", wf_base.cost_bps)),
        initial_capital=float(run.walk_forward_overrides.get("initial_capital", wf_base.initial_capital)),
    )

    if wf.train_days != wf_base.train_days or wf.test_days != wf_base.test_days:
        raise ValueError(
            f"run {run.name!r}: train_days/test_days must match walk_forward_base "
            f"(got train={wf.train_days} test={wf.test_days}, base train="
            f"{wf_base.train_days} test={wf_base.test_days}). Misaligned cadence "
            f"breaks PBO date alignment."
        )

    stats = replace(
        sweep.stats_base,
        n_trials=n_runs,
        sharpes_std=float(run.stats_overrides.get("sharpes_std", sweep.stats_base.sharpes_std)),
    )

    per_run_dir = str(Path(sweep.output_dir) / sweep.name / "per_run")
    return RunConfig(
        name=run.name,
        prices_csv=sweep.prices_csv,
        start_date=sweep.start_date,
        end_date=sweep.end_date,
        output_dir=per_run_dir,
        walk_forward=wf,
        signal=run.signal,
        stats=stats,
    )


# ------------------------------------------------------------------
# Returns matrix construction
# ------------------------------------------------------------------
def _per_period_returns_with_dates(report: dict[str, Any]) -> pl.DataFrame:
    """
    Read the artifact equity_curve.csv we just wrote and rebuild the
    per-period returns aligned to dates. The runner already emits dates
    for each rebalance; converting equity → returns matches what
    `BacktestResult.per_period_returns` does in memory (np.diff(eq)/eq[:-1]),
    but with the corresponding test_end dates attached.
    """
    eq_path = Path(report["artifacts"]["equity_curve"])
    df = pl.read_csv(eq_path, try_parse_dates=True).sort("date")
    if df.height < 2:
        return pl.DataFrame({"date": [], "ret": []}, schema={"date": pl.Date, "ret": pl.Float64})
    eq = df["equity"].to_numpy()
    rets = np.diff(eq) / eq[:-1]
    return pl.DataFrame({"date": df["date"][1:], "ret": rets})


def _build_returns_matrix(
    per_run_returns: dict[str, pl.DataFrame],
) -> tuple[np.ndarray, list[str], list[date]]:
    """
    Build a (T, N) matrix by inner-joining per-run return series on date.
    Because all runs share the same prices_csv and walk-forward cadence,
    the inner join should be lossless in normal operation; if a run fails
    to produce a rebalance for a given date (e.g. signal returned empty),
    we drop that date for *every* run rather than fabricate values.
    """
    if not per_run_returns:
        raise ValueError("no per-run return series — nothing to PBO over")

    names = list(per_run_returns.keys())
    joined: pl.DataFrame | None = None
    for name in names:
        df = per_run_returns[name].rename({"ret": f"ret__{name}"})
        joined = df if joined is None else joined.join(df, on="date", how="inner")
    assert joined is not None
    joined = joined.sort("date")

    dates = joined["date"].to_list()
    cols = [f"ret__{n}" for n in names]
    matrix = joined.select(cols).to_numpy()
    return matrix, names, dates


# ------------------------------------------------------------------
# Sweep runner
# ------------------------------------------------------------------
def run_sweep(cfg: SweepConfig) -> dict[str, Any]:
    """
    Execute every run in `cfg.runs`, compute cross-config PBO, write the
    sweep artifact bundle. Returns the in-memory sweep report.
    """
    n_runs = len(cfg.runs)
    if n_runs < 2:
        raise ValueError(f"sweep needs >= 2 runs to compute PBO, got {n_runs}")

    sweep_dir = Path(cfg.output_dir) / cfg.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "per_run").mkdir(parents=True, exist_ok=True)

    per_run_reports: list[dict[str, Any]] = []
    per_run_returns: dict[str, pl.DataFrame] = {}

    for run in cfg.runs:
        run_cfg = _resolve_run(cfg, run, n_runs=n_runs)
        log.info("sweep[%s] running %s", cfg.name, run.name)
        report = run_backtest(run_cfg)
        per_run_reports.append(report)
        per_run_returns[run.name] = _per_period_returns_with_dates(report)

    matrix, names, dates = _build_returns_matrix(per_run_returns)
    n_obs, n_cols = matrix.shape
    if n_obs < cfg.n_slices:
        raise ValueError(
            f"after date alignment, only {n_obs} observations remain — need >= "
            f"n_slices={cfg.n_slices} for CSCV PBO. Lengthen the window or "
            f"shorten train_days/test_days."
        )
    if n_cols != n_runs:
        raise RuntimeError(f"returns matrix has {n_cols} cols but sweep had {n_runs} runs")

    pbo_result = probability_of_backtest_overfitting(matrix, n_slices=cfg.n_slices)

    sweep_report: dict[str, Any] = {
        "name": cfg.name,
        "window": {"start": cfg.start_date.isoformat(), "end": cfg.end_date.isoformat()},
        "n_configs": n_runs,
        "n_observations_per_config": n_obs,
        "aligned_date_range": {
            "start": dates[0].isoformat() if dates else None,
            "end": dates[-1].isoformat() if dates else None,
        },
        "pbo": pbo_result["pbo"],
        "cscv_S": cfg.n_slices,
        "cscv_n_trials": pbo_result["n_trials"],
        "runs": [
            {
                "name": r["name"],
                "sharpe": r["metrics"]["sharpe"],
                "deflated_sharpe_p": r["metrics"]["deflated_sharpe_p"],
                "annualized_return": r["metrics"]["annualized_return"],
                "annualized_vol": r["metrics"]["annualized_vol"],
                "max_drawdown": r["metrics"]["max_drawdown"],
                "turnover": r["metrics"]["turnover"],
                "artifacts_dir": r["artifacts"]["dir"],
            }
            for r in per_run_reports
        ],
        "run_order": names,
    }

    out = _write_sweep_artifacts(cfg, sweep_report, per_run_reports)
    sweep_report["artifacts"] = out
    return sweep_report


def _write_sweep_artifacts(
    cfg: SweepConfig,
    sweep_report: dict[str, Any],
    per_run_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir) / cfg.name

    # Sweep manifest — fingerprint the underlying price CSV (same as runner)
    # plus a sweep_config_hash so re-running the same YAML is recognized.
    prices = load_prices_csv(cfg.prices_csv, cfg.start_date, cfg.end_date)
    data_tuples: list[tuple[Any, ...]] = [
        (r["date"].isoformat(), r["symbol"], float(r["adj_close"])) for r in prices.iter_rows(named=True)
    ]
    sweep_dict = _sweep_to_dict(cfg)
    manifest = build_manifest(config=sweep_dict, data_tuples=data_tuples)
    sweep_dict_bytes = json.dumps(sweep_dict, sort_keys=True, default=str).encode()
    sweep_config_hash = hashlib.sha256(sweep_dict_bytes).hexdigest()

    (out_dir / "sweep_report.json").write_text(
        json.dumps(sweep_report, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "sweep.config.snapshot.json").write_text(
        json.dumps(sweep_dict, indent=2, default=str), encoding="utf-8"
    )
    manifest_payload = json.loads(manifest.to_json())
    manifest_payload["sweep_config_hash"] = sweep_config_hash
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {
        "dir": str(out_dir),
        "sweep_report": str(out_dir / "sweep_report.json"),
        "sweep_config_snapshot": str(out_dir / "sweep.config.snapshot.json"),
        "manifest": str(out_dir / "manifest.json"),
        "per_run_dir": str(out_dir / "per_run"),
        "per_run": [pr["artifacts"]["dir"] for pr in per_run_reports],
    }


def _sweep_to_dict(cfg: SweepConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "prices_csv": cfg.prices_csv,
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "output_dir": cfg.output_dir,
        "walk_forward_base": {
            "train_days": cfg.walk_forward_base.train_days,
            "test_days": cfg.walk_forward_base.test_days,
            "top_k": cfg.walk_forward_base.top_k,
            "cost_bps": cfg.walk_forward_base.cost_bps,
            "initial_capital": cfg.walk_forward_base.initial_capital,
        },
        "stats_base": {
            "n_trials": cfg.stats_base.n_trials,
            "sharpes_std": cfg.stats_base.sharpes_std,
        },
        "n_slices": cfg.n_slices,
        "runs": [
            {
                "name": r.name,
                "signal": {"kind": r.signal.kind, "params": r.signal.params},
                "walk_forward_overrides": r.walk_forward_overrides,
                "stats_overrides": r.stats_overrides,
            }
            for r in cfg.runs
        ],
    }


__all__ = [
    "SweepConfig",
    "SweepRunSpec",
    "load_sweep_config",
    "run_sweep",
]
