# ML trainer — example

End-to-end demo of `quant ml train` against the real Kaggle
[S&P 500 5-year OHLCV](https://www.kaggle.com/datasets/camnugent/sandp500)
snapshot at `data/legacy/all_stocks_5yr.csv` (2013-02-08 → 2018-02-07,
505 symbols).

This trains the model TRUST.md describes: LightGBM multiclass on
triple-barrier labels (-1 / 0 / +1) with purged K-fold + embargo, MLflow-
tracked, manifest-published.

## Run

```bash
cd apps/api
.venv/bin/python -m quant.cli ml train examples/ml/sp500_lightgbm.yaml
```

This writes the artifact bundle under
`examples/ml/artifacts/sp500_lightgbm_v1/` and creates an MLflow run on the
configured tracking URI (`settings.mlflow_tracking_uri`; defaults to
`http://mlflow:5000` in the dev compose stack — point it at `file:./mlruns`
or any other tracking URI via `MLFLOW_TRACKING_URI` for local-only runs).

## Artifact bundle

| File | What |
| --- | --- |
| `train_report.json`        | dataset shape + per-fold metrics + aggregated OOF metrics + MLflow run id |
| `oof_predictions.csv`      | one row per labeled sample with class probabilities + picked class + `in_oof` flag |
| `feature_importances.csv`  | LightGBM gain importance summed across folds, descending |
| `manifest.json`            | `code_sha`, `config_hash`, `data_fingerprint`, package versions (same schema as the backtest manifest — same `build_manifest` call) |
| `config.snapshot.json`     | exact config used — diffable, the source of truth for "what did we run" |

Per the house rule in `/TRUST.md`: if any of these is missing, the run did not ship.

## Notes

- `data.max_symbols` caps the panel to keep memory under ~4 GB on a laptop.
  The selection is deterministic via `symbol_seed`, so the data fingerprint
  is stable across reruns. Remove the cap on a workstation to train on the
  full 505-name panel.
- LightGBM hyperparameters are sane defaults straight out of TRUST.md. They
  are **not tuned** — the example exists to prove the pipeline is real, not
  to set a benchmark.
- Out-of-fold predictions come from the purged K-fold splitter (`n_splits=5`,
  `embargo_frac=0.01`). No OOF prediction is produced by a model that saw
  the validation sample's price-path at training time; that is the whole
  point of using the purged splitter over plain K-fold.
