"""
Hyperparameter tuning for the LightGBM trainer via Optuna TPE.

BIAS WARNING (calibration note):
    The post-tuning OOF logloss is biased downward by selection pressure.
    Optuna picks the trial that minimized OOF logloss on the TUNING WINDOW
    data. Reporting that same logloss as the model quality estimate confounds
    "best selected" with "expected future performance."

    Mitigation via `holdout_frac` (default 0.2):
    - All Optuna trials train and score on [start_date, holdout_cutoff].
    - After HPO, a SEPARATE training run with the best params is executed on
      the HOLDOUT WINDOW [holdout_cutoff, end_date] only.
    - The holdout OOF metrics (oof_logloss_holdout, oof_auc_holdout) are an
      honest post-selection generalization estimate.
    - The holdout window is excluded from ALL tuning trials — the model has
      never seen it during HPO search.

Memory profile (8GB Mac M2 Air):
- Default 200-symbol panel: ~1.5GB peak per trial.
- 30 trials × ~30s each = ~15 minutes wall time.
- Optuna's storage stays in-process; no SQLite by default.
"""

from __future__ import annotations

import dataclasses
import logging
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import optuna

from quant.ml.config import DataSpec, TrainConfig
from quant.ml.trainer import train

log = logging.getLogger("quant.ml.tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial: optuna.Trial, base_cfg: TrainConfig) -> float:
    """Sample a config, run the trainer, return OOF logloss on the tuning window."""
    params = dict(base_cfg.model.params)
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        }
    )
    num_boost_round = trial.suggest_int("num_boost_round", 100, 400)

    new_cfg = dataclasses.replace(
        base_cfg,
        name=f"{base_cfg.name}_trial_{trial.number}",
        output_dir=str(Path(base_cfg.output_dir) / "tune_trials"),
        model=type(base_cfg.model)(
            num_boost_round=num_boost_round,
            early_stopping_rounds=base_cfg.model.early_stopping_rounds,
            params={**deepcopy(params)},
        ),
    )

    report = train(new_cfg)
    logloss = float(report["oof_metrics"]["oof_logloss"])
    log.info("trial %d: logloss=%.4f (tuning window only)", trial.number, logloss)
    return logloss


def _tune_result_dict(
    n_trials: int,
    best_value: float,
    best_params: dict[str, Any],
    history: list[dict[str, Any]],
    holdout_start: Any,
    holdout_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "n_trials": n_trials,
        "best_value": best_value,
        "best_params": best_params,
        "history": history,
        "tuning_best_value_is_selection_biased": True,
        "holdout_start_date": str(holdout_start),
        "holdout_metrics": holdout_metrics or {},
    }


def tune(
    base_cfg: TrainConfig,
    *,
    n_trials: int = 30,
    seed: int = 42,
    holdout_frac: float = 0.2,
) -> dict[str, Any]:
    """
    Run Optuna TPE; return best params + honest holdout evaluation.

    holdout_frac=0.2 (default): last 20% of date range reserved for post-tuning
    evaluation. All HPO trials train on [start_date, holdout_cutoff). After HPO,
    a single training run on [holdout_cutoff, end_date] with the best params
    gives an honest generalization estimate that was never seen by Optuna.
    """
    total_days = (base_cfg.data.end_date - base_cfg.data.start_date).days
    holdout_days = max(90, int(total_days * holdout_frac))
    holdout_start = base_cfg.data.end_date - timedelta(days=holdout_days)

    log.info(
        "HPO: tuning on %s→%s | holdout: %s→%s",
        base_cfg.data.start_date,
        holdout_start,
        holdout_start,
        base_cfg.data.end_date,
    )

    # Build a tuning config that excludes the holdout period
    tuning_data: DataSpec = dataclasses.replace(base_cfg.data, end_date=holdout_start)
    tuning_cfg: TrainConfig = dataclasses.replace(base_cfg, data=tuning_data)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: _objective(t, tuning_cfg), n_trials=n_trials)

    history = [
        {
            "trial": t.number,
            "value": t.value if t.value is not None else float("nan"),
            "params": dict(t.params),
        }
        for t in study.trials
    ]
    best = study.best_trial
    best_params = dict(best.params)

    # Post-HPO holdout evaluation: train with best params on holdout window only.
    # This never overlaps with the tuning trials — honest generalization estimate.
    holdout_metrics: dict[str, float] | None = None
    try:
        num_boost_round = int(best_params.get("num_boost_round", base_cfg.model.num_boost_round))
        best_lgbm_params = {**deepcopy(base_cfg.model.params)}
        best_lgbm_params.update({k: v for k, v in best_params.items() if k != "num_boost_round"})
        holdout_model = type(base_cfg.model)(
            num_boost_round=num_boost_round,
            early_stopping_rounds=base_cfg.model.early_stopping_rounds,
            params=best_lgbm_params,
        )
        holdout_data: DataSpec = dataclasses.replace(
            base_cfg.data, start_date=holdout_start
        )
        holdout_cfg: TrainConfig = dataclasses.replace(
            base_cfg,
            name=f"{base_cfg.name}_holdout_eval",
            output_dir=str(Path(base_cfg.output_dir) / "holdout_eval"),
            data=holdout_data,
            model=holdout_model,
        )
        holdout_report = train(holdout_cfg)
        holdout_metrics = {
            "oof_logloss": float(holdout_report["oof_metrics"]["oof_logloss"]),
            "oof_macro_auc": float(holdout_report["oof_metrics"]["oof_macro_auc_ovr"]),
            "oof_balanced_accuracy": float(holdout_report["oof_metrics"]["oof_balanced_accuracy"]),
        }
        log.info(
            "holdout eval: logloss=%.4f, auc=%.4f",
            holdout_metrics["oof_logloss"],
            holdout_metrics["oof_macro_auc"],
        )
    except Exception as exc:
        log.warning("holdout evaluation failed: %s", exc)

    return _tune_result_dict(
        n_trials=n_trials,
        best_value=float(best.value if best.value is not None else float("nan")),
        best_params=best_params,
        history=history,
        holdout_start=holdout_start,
        holdout_metrics=holdout_metrics,
    )


__all__ = ["tune"]
