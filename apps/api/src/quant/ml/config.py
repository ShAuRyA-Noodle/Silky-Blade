"""
TrainConfig — frozen dataclass + YAML/JSON loader for the ML trainer.

Mirrors the shape of `quant.backtest.runner.RunConfig` so the two pipelines feel
the same to operators. The config is the only knob; `trainer.train` reads it,
nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Defaults are spec'd in TRUST.md so the headline claim ("LightGBM multiclass
# 3-class on triple-barrier labels with purged K-fold + embargo") is what
# actually runs when a user invokes the trainer with no overrides.
_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "deterministic": True,
    "seed": 42,
    "n_estimators": 200,
}


@dataclass(frozen=True)
class LabelSpec:
    """Triple-barrier knobs. Defaults match `TripleBarrierConfig`'s defaults."""

    horizon: int = 5
    pt_sigma: float = 2.0
    sl_sigma: float = 2.0
    vol_window: int = 21
    min_vol: float = 1e-4


@dataclass(frozen=True)
class CVSpec:
    """Purged K-fold knobs."""

    n_splits: int = 5
    embargo_frac: float = 0.01


@dataclass(frozen=True)
class ModelSpec:
    """LightGBM hyperparameters + boosting controls."""

    params: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_LGBM_PARAMS))
    num_boost_round: int = 200
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class DataSpec:
    """
    Where to read OHLCV from + how to subset symbols.

    `max_symbols` exists because the full Kaggle 5y snapshot is 505 names ×
    ~1259 days. Triple-barrier per-symbol O(N·H) plus 5-fold LightGBM training
    on the resulting matrix can OOM on a laptop. The cap is real-data
    downsampling — still real prices, fewer of them — and is logged in the
    manifest so the run is reproducible.
    """

    prices_csv: str
    start_date: date
    end_date: date
    max_symbols: int | None = None
    symbol_seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    name: str
    output_dir: str
    data: DataSpec
    label: LabelSpec = field(default_factory=LabelSpec)
    cv: CVSpec = field(default_factory=CVSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    mlflow_experiment: str | None = None  # falls back to settings.mlflow_experiment_name


def load_config(path: str | Path) -> TrainConfig:
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


def _coerce_config(data: dict[str, Any]) -> TrainConfig:
    data_block = data.get("data") or {}
    label_block = data.get("label") or {}
    cv_block = data.get("cv") or {}
    model_block = data.get("model") or {}

    if "prices_csv" not in data_block:
        raise ValueError("config.data.prices_csv is required")
    if "start_date" not in data_block or "end_date" not in data_block:
        raise ValueError("config.data.start_date and config.data.end_date are required")

    # Layered param dict: caller overrides override the trainer defaults — no
    # silent loss of any default LightGBM key the caller didn't mention.
    params = dict(_DEFAULT_LGBM_PARAMS)
    params.update(model_block.get("params") or {})

    return TrainConfig(
        name=str(data["name"]),
        output_dir=str(data.get("output_dir", "./artifacts/ml")),
        data=DataSpec(
            prices_csv=str(data_block["prices_csv"]),
            start_date=_as_date(data_block["start_date"]),
            end_date=_as_date(data_block["end_date"]),
            max_symbols=(
                int(data_block["max_symbols"]) if data_block.get("max_symbols") is not None else None
            ),
            symbol_seed=int(data_block.get("symbol_seed", 42)),
        ),
        label=LabelSpec(
            horizon=int(label_block.get("horizon", 5)),
            pt_sigma=float(label_block.get("pt_sigma", 2.0)),
            sl_sigma=float(label_block.get("sl_sigma", 2.0)),
            vol_window=int(label_block.get("vol_window", 21)),
            min_vol=float(label_block.get("min_vol", 1e-4)),
        ),
        cv=CVSpec(
            n_splits=int(cv_block.get("n_splits", 5)),
            embargo_frac=float(cv_block.get("embargo_frac", 0.01)),
        ),
        model=ModelSpec(
            params=params,
            num_boost_round=int(model_block.get("num_boost_round", 200)),
            early_stopping_rounds=int(model_block.get("early_stopping_rounds", 50)),
        ),
        mlflow_experiment=(
            str(data["mlflow_experiment"]) if data.get("mlflow_experiment") is not None else None
        ),
    )


def _as_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Plain-dict view for manifest hashing + on-disk snapshot. Stable key order."""
    return {
        "name": cfg.name,
        "output_dir": cfg.output_dir,
        "data": {
            "prices_csv": cfg.data.prices_csv,
            "start_date": cfg.data.start_date.isoformat(),
            "end_date": cfg.data.end_date.isoformat(),
            "max_symbols": cfg.data.max_symbols,
            "symbol_seed": cfg.data.symbol_seed,
        },
        "label": {
            "horizon": cfg.label.horizon,
            "pt_sigma": cfg.label.pt_sigma,
            "sl_sigma": cfg.label.sl_sigma,
            "vol_window": cfg.label.vol_window,
            "min_vol": cfg.label.min_vol,
        },
        "cv": {
            "n_splits": cfg.cv.n_splits,
            "embargo_frac": cfg.cv.embargo_frac,
        },
        "model": {
            "params": cfg.model.params,
            "num_boost_round": cfg.model.num_boost_round,
            "early_stopping_rounds": cfg.model.early_stopping_rounds,
        },
        "mlflow_experiment": cfg.mlflow_experiment,
    }


__all__ = [
    "CVSpec",
    "DataSpec",
    "LabelSpec",
    "ModelSpec",
    "TrainConfig",
    "config_to_dict",
    "load_config",
]
