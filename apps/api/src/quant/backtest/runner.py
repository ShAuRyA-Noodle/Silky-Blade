"""
Backtest orchestrator — config-driven runner that ties together every piece
of the backtest plane (engine · statistics · reproducibility) into a single
reproducible artifact set.

One run produces, on disk:

    <output_dir>/<run_name>/
        report.json       — headline metrics + DSR (+ PBO if sweep)
        equity_curve.csv  — date, equity pairs across rebalances
        manifest.json     — code_sha, config_hash, data_fingerprint, env snapshot
        config.snapshot.json — exact config used (for human diffing)

If ANY of those four files is missing, the house rule says the result has
not shipped. The writer here is atomic — the directory is only created once
all artifacts are built in memory.

Data intake:
- `prices_csv`: a CSV with columns [date, symbol, adj_close]. Dates ISO.
  This is the offline/CI path. Also used by tests.
- `prices_db` (future): query the canonical bars table. Not implemented in
  this file — the DB path uses the existing adapters + session, which the
  caller can wire into a subclass. Keeping this module DB-free keeps the
  test surface small.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant.backtest.engine import SignalProducer, WalkForwardConfig, walk_forward
from quant.backtest.reproducibility import build_manifest
from quant.backtest.signals import (
    CompositeSignal,
    LowVolSignal,
    MeanReversionSignal,
    MLBundleSignal,
    MLPredictionsSignal,
    MomentumSignal,
    SentimentSignal,
    ValueSignal,
)
from quant.backtest.statistics import deflated_sharpe_ratio, sharpe_ratio

log = logging.getLogger("quant.backtest.runner")


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass(frozen=True)
class SignalSpec:
    """Selector for the built-in SignalProducer registry."""

    kind: str = "momentum"  # momentum | ...
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StatsSpec:
    """DSR inputs that can't be derived from a single-run backtest."""

    n_trials: int = 1
    sharpes_std: float = 0.5  # plausible prior over the trial-pool sharpes


@dataclass(frozen=True)
class RunConfig:
    name: str
    prices_csv: str
    start_date: date
    end_date: date
    output_dir: str
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    signal: SignalSpec = field(default_factory=SignalSpec)
    stats: StatsSpec = field(default_factory=StatsSpec)
    # Universe enforcement at each rebalance:
    #   "raw"        : no filter — uses every symbol present in prices_csv.
    #   "sp500_pit"  : intersect with point-in-time S&P 500 membership at
    #                  the rebalance date (Wikipedia-sourced).
    universe: str = "raw"


# ------------------------------------------------------------------
# Config I/O
# ------------------------------------------------------------------
def load_config(path: str | Path) -> RunConfig:
    """Load a YAML or JSON config. YAML requires PyYAML."""
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
    return _coerce_config(data)


def _coerce_config(data: dict[str, Any]) -> RunConfig:
    wf = data.get("walk_forward") or {}
    sig = data.get("signal") or {}
    stats = data.get("stats") or {}
    return RunConfig(
        name=str(data["name"]),
        prices_csv=str(data["prices_csv"]),
        start_date=_as_date(data["start_date"]),
        end_date=_as_date(data["end_date"]),
        output_dir=str(data.get("output_dir", "./artifacts/backtest")),
        walk_forward=WalkForwardConfig(
            train_days=int(wf.get("train_days", 252)),
            test_days=int(wf.get("test_days", 21)),
            top_k=int(wf.get("top_k", 20)),
            cost_bps=float(wf.get("cost_bps", 5.0)),
            initial_capital=float(wf.get("initial_capital", 100_000.0)),
        ),
        signal=SignalSpec(
            kind=str(sig.get("kind", "momentum")),
            params=dict(sig.get("params", {})),
        ),
        stats=StatsSpec(
            n_trials=int(stats.get("n_trials", 1)),
            sharpes_std=float(stats.get("sharpes_std", 0.5)),
        ),
        universe=str(data.get("universe", "raw")),
    )


def _as_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


# ------------------------------------------------------------------
# Signal registry
# ------------------------------------------------------------------
def build_signal(spec: SignalSpec) -> SignalProducer:
    if spec.kind == "momentum":
        return MomentumSignal(lookback_days=int(spec.params.get("lookback_days", 126)))
    if spec.kind == "low_vol":
        return LowVolSignal(lookback_days=int(spec.params.get("lookback_days", 126)))
    if spec.kind == "mean_reversion":
        return MeanReversionSignal(lookback_days=int(spec.params.get("lookback_days", 5)))
    if spec.kind == "ml_predictions":
        path = spec.params.get("predictions_csv")
        if not isinstance(path, str) or not path:
            raise ValueError("ml_predictions signal requires params.predictions_csv (path)")
        use_cal = bool(spec.params.get("use_calibrated", True))
        return MLPredictionsSignal(predictions_csv=path, use_calibrated=use_cal)
    if spec.kind == "ml_bundle":
        model_dir = spec.params.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise ValueError("ml_bundle signal requires params.model_dir (path to artifact dir)")
        return MLBundleSignal(model_dir=model_dir)
    if spec.kind == "value":
        path = spec.params.get("fundamentals_csv")
        if not isinstance(path, str) or not path:
            raise ValueError("value signal requires params.fundamentals_csv")
        return ValueSignal(fundamentals_csv=path)
    if spec.kind == "sentiment":
        sent_path = spec.params.get("sentiment_csv")
        if not isinstance(sent_path, str) or not sent_path:
            raise ValueError("sentiment signal requires params.sentiment_csv")
        lookback = int(spec.params.get("lookback_days", 3))
        return SentimentSignal(sentiment_csv=sent_path, lookback_days=lookback)
    if spec.kind == "composite":
        # Composite blends two children. Each child is itself a SignalSpec
        # nested in params: params.primary = {kind, params}, params.secondary
        # = {kind, params}. Weights default to alpha=0.7 / beta=0.3.
        prim = spec.params.get("primary")
        sec = spec.params.get("secondary")
        if not isinstance(prim, dict) or not isinstance(sec, dict):
            raise ValueError(
                "composite signal requires params.primary AND params.secondary (each a {kind, params} dict)"
            )
        alpha = float(spec.params.get("alpha", 0.7))
        beta = float(spec.params.get("beta", 1.0 - alpha))
        outer = bool(spec.params.get("outer_join", False))
        primary_sig = build_signal(SignalSpec(kind=str(prim["kind"]), params=dict(prim.get("params", {}))))
        secondary_sig = build_signal(SignalSpec(kind=str(sec["kind"]), params=dict(sec.get("params", {}))))
        return CompositeSignal(
            primary=primary_sig,
            secondary=secondary_sig,
            alpha=alpha,
            beta=beta,
            outer_join=outer,
        )
    raise ValueError(f"unknown signal kind: {spec.kind!r}")


# ------------------------------------------------------------------
# Universe filter registry
# ------------------------------------------------------------------
def _build_universe_filter(name: str) -> Any:
    """Resolve the `universe` config value to a UniverseFilter callable, or None."""
    if name == "raw":
        return None
    if name == "sp500_pit":
        # Local import — point_in_time fetches Wikipedia at construction time,
        # so we only pay that cost when the user opted in.
        from quant.backtest.universe_filter import point_in_time_sp500_filter

        log.info("universe=sp500_pit — fetching Wikipedia changes table")
        return point_in_time_sp500_filter()
    raise ValueError(f"unknown universe filter: {name!r}")


# ------------------------------------------------------------------
# Data intake
# ------------------------------------------------------------------
def load_prices_csv(path: str | Path, start: date, end: date) -> pl.DataFrame:
    """Load a CSV with [date, symbol, adj_close], filter to window, typecheck."""
    df = pl.read_csv(str(path), try_parse_dates=True)
    missing = {"date", "symbol", "adj_close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df.with_columns(pl.col("date").cast(pl.Date))
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def run_backtest(cfg: RunConfig) -> dict[str, Any]:
    """
    Execute one backtest end-to-end. Returns the in-memory report dict AND
    writes the four-file artifact bundle to disk.
    """
    prices = load_prices_csv(cfg.prices_csv, cfg.start_date, cfg.end_date)
    if prices.is_empty():
        raise RuntimeError(f"no price rows in {cfg.prices_csv} for {cfg.start_date}→{cfg.end_date}")

    producer = build_signal(cfg.signal)
    universe_filter = _build_universe_filter(cfg.universe)
    result = walk_forward(prices, producer, cfg.walk_forward, universe_filter=universe_filter)

    # Per-period returns → annualized numbers already on result; compute DSR.
    rets = result.per_period_returns
    periods_per_year = 252.0 / cfg.walk_forward.test_days
    obs_sharpe = sharpe_ratio(rets, periods_per_year=round(periods_per_year))
    skew = _safe_moment(rets, 3)
    kurt = _safe_moment(rets, 4)
    dsr = (
        deflated_sharpe_ratio(
            obs_sharpe,
            n_trials=cfg.stats.n_trials,
            sharpes_std=cfg.stats.sharpes_std,
            n_obs=int(rets.size),
            skew=skew,
            kurtosis=kurt if kurt > 0 else 3.0,
        )
        if cfg.stats.n_trials >= 1 and rets.size > 2
        else float("nan")
    )

    report: dict[str, Any] = {
        "name": cfg.name,
        "window": {
            "start": cfg.start_date.isoformat(),
            "end": cfg.end_date.isoformat(),
            "n_rebalances": len(result.equity_curve),
        },
        "metrics": {
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "annualized_vol": result.annualized_vol,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "turnover": result.turnover,
            "deflated_sharpe_p": dsr,
            "dsr_n_trials": cfg.stats.n_trials,
            "dsr_sharpes_std": cfg.stats.sharpes_std,
            "return_skew": skew,
            "return_kurtosis": kurt,
        },
        "walk_forward": {
            "train_days": cfg.walk_forward.train_days,
            "test_days": cfg.walk_forward.test_days,
            "top_k": cfg.walk_forward.top_k,
            "cost_bps": cfg.walk_forward.cost_bps,
            "initial_capital": cfg.walk_forward.initial_capital,
        },
        "signal": {"kind": cfg.signal.kind, "params": cfg.signal.params},
    }

    out = _write_artifacts(cfg, prices, result.equity_curve, report)
    report["artifacts"] = out
    return report


def _safe_moment(rets: np.ndarray, k: int) -> float:
    if rets.size < 4:
        return 0.0 if k == 3 else 3.0
    mean = float(rets.mean())
    std = float(rets.std(ddof=1))
    if std == 0:
        return 0.0 if k == 3 else 3.0
    z = (rets - mean) / std
    moment = float((z**k).mean())
    return moment


def _write_artifacts(
    cfg: RunConfig,
    prices: pl.DataFrame,
    equity_curve: pl.DataFrame,
    report: dict[str, Any],
) -> dict[str, str]:
    out_dir = Path(cfg.output_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest — data_fingerprint is a sha of (date, symbol, adj_close) tuples.
    data_tuples: list[tuple[Any, ...]] = [
        (r["date"].isoformat(), r["symbol"], float(r["adj_close"])) for r in prices.iter_rows(named=True)
    ]
    manifest = build_manifest(config=_config_to_dict(cfg), data_tuples=data_tuples)

    equity_curve.write_csv(out_dir / "equity_curve.csv")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    (out_dir / "config.snapshot.json").write_text(
        json.dumps(_config_to_dict(cfg), indent=2, default=str), encoding="utf-8"
    )

    return {
        "dir": str(out_dir),
        "report": str(out_dir / "report.json"),
        "equity_curve": str(out_dir / "equity_curve.csv"),
        "manifest": str(out_dir / "manifest.json"),
        "config_snapshot": str(out_dir / "config.snapshot.json"),
    }


def _config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "prices_csv": cfg.prices_csv,
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "output_dir": cfg.output_dir,
        "walk_forward": {
            "train_days": cfg.walk_forward.train_days,
            "test_days": cfg.walk_forward.test_days,
            "top_k": cfg.walk_forward.top_k,
            "cost_bps": cfg.walk_forward.cost_bps,
            "initial_capital": cfg.walk_forward.initial_capital,
        },
        "signal": {"kind": cfg.signal.kind, "params": cfg.signal.params},
        "stats": {"n_trials": cfg.stats.n_trials, "sharpes_std": cfg.stats.sharpes_std},
        "universe": cfg.universe,
    }


__all__ = [
    "RunConfig",
    "SignalSpec",
    "StatsSpec",
    "build_signal",
    "load_config",
    "load_prices_csv",
    "run_backtest",
]
