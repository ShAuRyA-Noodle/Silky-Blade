"""Backtest — walk-forward engine + overfitting-aware statistics + runner."""

from quant.backtest.engine import (
    BacktestResult,
    SignalProducer,
    WalkForwardConfig,
    walk_forward,
)
from quant.backtest.reproducibility import ReproManifest, build_manifest
from quant.backtest.runner import (
    RunConfig,
    SignalSpec,
    StatsSpec,
    load_config,
    load_prices_csv,
    run_backtest,
)
from quant.backtest.signals import (
    LowVolSignal,
    MeanReversionSignal,
    MLPredictionsSignal,
    MomentumSignal,
)
from quant.backtest.statistics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from quant.backtest.sweep import (
    SweepConfig,
    SweepRunSpec,
    load_sweep_config,
    run_sweep,
)

__all__ = [
    "BacktestResult",
    "LowVolSignal",
    "MLPredictionsSignal",
    "MeanReversionSignal",
    "MomentumSignal",
    "ReproManifest",
    "RunConfig",
    "SignalProducer",
    "SignalSpec",
    "StatsSpec",
    "SweepConfig",
    "SweepRunSpec",
    "WalkForwardConfig",
    "build_manifest",
    "deflated_sharpe_ratio",
    "load_config",
    "load_prices_csv",
    "load_sweep_config",
    "probability_of_backtest_overfitting",
    "run_backtest",
    "run_sweep",
    "sharpe_ratio",
    "walk_forward",
]
