# REPRODUCE — verifying the demo backtest from a cold start

**Date:** 2026-05-02
**Scope:** the public demo at `apps/api/examples/backtest/sp500_momentum.yaml`.
**Companion to:** [TRUST.md](./TRUST.md) — read that first for *why* the
mechanisms below matter; this file is the *how*.

---

## 1. What "reproducible" means here

A run produces four artifacts under `<output_dir>/<run_name>/`:

| File | Reproducible? | Why |
| --- | --- | --- |
| `manifest.json`         | Yes (`code_sha`, `config_hash`, `data_fingerprint`); `created_at` differs (timestamp) and `package_versions` differs only if you `uv sync` to a different lockfile. | These are pure hashes of inputs. |
| `config.snapshot.json`  | Bit-exact across runs.                                                                                                                                          | Echo of the YAML config.        |
| `report.json`           | Stable to ~9 significant digits; **last 1–2 digits drift across runs**.                                                                                          | See [§7 Known issues](#7-known-issues). |
| `equity_curve.csv`      | Stable to the cent; ULP-level drift in the trailing decimal.                                                                                                     | Same root cause.                |

The manifest's job is to identify the **inputs** (code + config + data),
not to bit-snapshot the **outputs**. It does that correctly — the three
fingerprint fields are deterministic across machines, processes, and
threads. The numerical output drifts at the floating-point ULP level
(~1e-15 relative); headline metrics (Sharpe to 3 decimals, equity to the
dollar) are stable.

**What the manifest covers:**
- `code_sha` — git HEAD at run time.
- `config_hash` — sha256 of canonical-JSON of the run config.
- `data_fingerprint` — sha256 over `(date, symbol, adj_close)` tuples of
  the **adapted** prices CSV (`sp500_5yr_adjusted.csv`), filtered to the
  config's `[start_date, end_date]` window. **Sorted**, then newline-joined.
- `package_versions` — pinned versions of polars / numpy / scikit-learn /
  lightgbm / mlflow / sqlalchemy / fastapi.
- `python_version`, `created_at`.

**What it does NOT cover:**
- The raw Kaggle source CSV (`data/legacy/all_stocks_5yr.csv`). Not
  committed to git — only the post-adapter fingerprint is. If you regenerate
  `sp500_5yr_adjusted.csv` from the Kaggle download and the
  `data_fingerprint` doesn't match, you have the wrong source data.
- The OS, CPU, BLAS implementation. ULP-level drift will occur across
  these even with identical Python packages — see §7.
- The LightGBM training run. The demo here is the **null-hypothesis
  momentum baseline**, not an ML run. The ML side (model registry +
  MLflow run IDs) is documented in TRUST.md but not exercised by this
  config.

---

## 2. Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python      | 3.12+               | `pyproject.toml` requires `>=3.12`. |
| `uv`        | any recent          | optional; `pip install -e .` works too. |
| Disk        | ~100 MB             | for the venv + adapted CSV + artifacts. |
| OS          | linux / macOS       | tested on macOS arm64 (Darwin 25.4.0). |

Node is not required — this run touches the API/quant Python only, not
the `apps/web` frontend.

### 2.1 The source dataset

The backtest needs `data/legacy/all_stocks_5yr.csv` — the
[Kaggle "S&P 500 stock data (5-year)"][kaggle] snapshot
(2013-02-08 → 2018-02-07, 505 symbols). The repo does **not** ship this
file (`.gitignore` excludes `data/legacy/*`). Download it manually:

[kaggle]: https://www.kaggle.com/datasets/camnugent/sandp500

1. `https://www.kaggle.com/datasets/camnugent/sandp500` → download.
2. Unzip; place `all_stocks_5yr.csv` at `data/legacy/all_stocks_5yr.csv`.
3. Sanity check: `wc -l data/legacy/all_stocks_5yr.csv` → ~619,041 lines
   (1 header + 619,040 data rows).

If the data fingerprint after the adapter doesn't match
`c575a81cf6ab27a94a83dfa4faba3fe8e09ff15b8b4d131e2141c927d0ac57fb`
(see §6.3), you have a different version of the Kaggle file. That is
the manifest doing its job — refuse to publish a Sharpe attached to the
wrong data.

---

## 3. Setup

```bash
# from repo root
cd apps/api

# Option A — uv (fastest, what the repo uses):
uv sync

# Option B — pip:
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Verify:

```bash
.venv/bin/python -c "import polars, lightgbm, numpy; print(polars.__version__, lightgbm.__version__, numpy.__version__)"
# expect:  1.39.3 4.6.0 2.4.4   (or your locked versions)
```

---

## 4. Run

```bash
# from apps/api
.venv/bin/python examples/backtest/prepare_sp500_5yr.py
.venv/bin/python -m quant.cli backtest run examples/backtest/sp500_momentum.yaml
```

Expected stdout from the runner:

```
Running backtest 'sp500_momentum_126' (2014-01-02 → 2018-02-07)
Done. Sharpe=1.725  DSR P=0.998  DD=8.4%  Turnover=15.1x  AnnRet=22.48%  AnnVol=13.03%
Artifacts: examples/backtest/artifacts/sp500_momentum_126
```

Artifact bundle on disk:

```
apps/api/examples/backtest/artifacts/sp500_momentum_126/
    report.json
    equity_curve.csv
    manifest.json
    config.snapshot.json
```

If any of those four files is missing, the run did not ship.

---

## 5. Expected numbers

These were observed on `code_sha cd7c8c32d40c72baa0ea740d84277c536948907c`,
Python 3.12.12, polars 1.39.3, numpy 2.4.4, lightgbm 4.6.0, on macOS arm64.

| Metric | Expected value | Tolerance |
| --- | --- | --- |
| `metrics.sharpe`              | `1.7253` | ±1e-12 |
| `metrics.deflated_sharpe_p`   | `0.9982` | ±1e-9  |
| `metrics.annualized_return`   | `0.2248` (22.48%) | ±1e-12 |
| `metrics.annualized_vol`      | `0.1303` (13.03%) | ±1e-12 |
| `metrics.max_drawdown`        | `0.0836` (8.36%)  | exact (no float reductions in the path) |
| `metrics.turnover`            | `15.06`          | exact |
| `metrics.return_skew`         | `-0.684`         | ±1e-9 |
| `metrics.return_kurtosis`     | `3.052`          | ±1e-9 |
| `window.n_rebalances`         | `37`             | exact |

If your headline Sharpe rounds to `1.725` and DSR P to `0.998`, you
reproduced it. If it rounds to `1.7` and DSR P to `0.99`, that's also
fine — see §7. If it rounds to `1.5` or `2.0`, something is wrong; see §7.

---

## 6. Verify the manifest claims

Each fingerprint in `manifest.json` should be independently
recomputable. Substitute your own paths if you ran from a different cwd.

### 6.1 `code_sha` matches `git rev-parse HEAD`

```bash
git rev-parse HEAD
grep code_sha apps/api/examples/backtest/artifacts/sp500_momentum_126/manifest.json
# both:  cd7c8c32d40c72baa0ea740d84277c536948907c
```

### 6.2 `config_hash` matches sha256 of the canonical config

```bash
cd apps/api
.venv/bin/python -c "
import hashlib, json
cfg = json.load(open('examples/backtest/artifacts/sp500_momentum_126/config.snapshot.json'))
print(hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest())
"
# expect:  e141b52e9bee08f1bda50d7721f329f0e27f97f7584ca70443b01d102d73363d
```

### 6.3 `data_fingerprint` matches sha256 over the adapted prices

```bash
cd apps/api
.venv/bin/python -c "
import hashlib
import polars as pl
from datetime import date
df = (pl.read_csv('examples/backtest/sp500_5yr_adjusted.csv', try_parse_dates=True)
        .with_columns(pl.col('date').cast(pl.Date))
        .filter((pl.col('date') >= date(2014,1,2)) & (pl.col('date') <= date(2018,2,7))))
tuples = [(r['date'].isoformat(), r['symbol'], float(r['adj_close'])) for r in df.iter_rows(named=True)]
print(hashlib.sha256('\n'.join(str(t) for t in sorted(tuples)).encode()).hexdigest())
"
# expect:  c575a81cf6ab27a94a83dfa4faba3fe8e09ff15b8b4d131e2141c927d0ac57fb
```

The fingerprint is computed in `apps/api/src/quant/backtest/reproducibility.py`
(`build_manifest`) — read the source if you don't trust this docs page.

### 6.4 `package_versions` matches the venv

```bash
.venv/bin/python -c "
from importlib.metadata import version
for p in ['polars','numpy','scikit-learn','lightgbm','mlflow','sqlalchemy','fastapi']:
    print(p, version(p))
"
# matches the package_versions block in manifest.json
```

---

## 7. Known issues

### 7.1 ULP-level drift in equity curve and Sharpe

**Symptom.** Two back-to-back runs of the *exact same config on the
exact same data with the exact same code* produce equity curves and
Sharpe ratios that differ in the **last 1–2 decimal digits**. Example
from a clean A/B run on this branch:

```diff
< sharpe   1.7253327419472226
> sharpe   1.7253327419472237      # ~1e-15 relative drift
< 2018-02-01,186849.53547712904
> 2018-02-01,186849.53547712907    # cents are the same; trailing digit drifts
```

**Root cause.** Polars executes `group_by`, `filter`, and column
reductions on a multi-threaded backend. Floating-point addition is
non-associative, so summing the same N float64 values in different
orders produces ULP-level different totals. Setting
`POLARS_MAX_THREADS=1` reduces but does not eliminate this on CPython
across processes — process-to-process hashmap iteration order in the
Polars query plan also contributes.

**Disposition.** Documented, not fixed in this audit. The drift is
~1e-15 relative — well below any threshold at which a Sharpe or an
equity curve should be compared. The three fingerprint fields in
`manifest.json` (`code_sha`, `config_hash`, `data_fingerprint`) are
unaffected and remain bit-exact across runs. A future fix would
replace the engine's Polars `group_by` + Python sum (`engine.py`
`_portfolio_return`) with a deterministic NumPy reduction in a
canonical symbol order. That's a real change in the hot path — out of
scope for a docs PR.

A small inline comment was added to `engine.py` explaining why
`maintain_order=True` is set on the `group_by` (it is necessary but
not sufficient for bit-exact reproducibility).

### 7.2 `created_at` differs across runs

By design — it's a UTC timestamp, not a fingerprint input. Ignore it
when diffing.

### 7.3 LightGBM training run is out of scope

The demo here is the **momentum null hypothesis**, not an ML run. The
LightGBM trainer (`apps/api/src/quant/models/lightgbm_trainer.py`) sets
`seed: 42`, `deterministic: True`, but a full ML reproduce would
additionally need to capture the training feature-store snapshot and an
MLflow run ID. That path is described in TRUST.md §3 but is not
exercised by `sp500_momentum.yaml`. When the
`ml/lightgbm-trainer` example lands, this section should be expanded.

---

## 8. If your numbers differ

| Difference                                                                       | Likely cause                                                                | What to do                                                                                                          |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `created_at` differs                                                             | timestamp                                                                   | Ignore.                                                                                                             |
| Last 1–2 digits of `sharpe`, `equity` differ                                     | Polars float-reduction non-determinism (§7.1)                                | Ignore — it's ULP drift.                                                                                            |
| `package_versions` differs                                                       | venv changed (`uv sync` against newer lock, or you used `pip install` direct) | Sync to the same versions or accept that small numeric differences may follow.                                       |
| `data_fingerprint` differs                                                       | wrong source CSV, or the adapter changed                                    | **Red flag.** Do not trust the resulting Sharpe. Re-download the Kaggle file; verify `wc -l` matches §2.1.            |
| `config_hash` differs                                                            | YAML edited or `_coerce_config` defaults shifted                            | **Red flag.** Diff `config.snapshot.json` against the committed YAML; resolve before claiming a result.              |
| `code_sha` differs                                                               | local git HEAD ≠ the published one                                          | `git checkout <published_sha>`, rerun, recompare.                                                                    |
| Sharpe rounds to `1.5` or `2.0` instead of `1.725`                               | engine, signal, or stats logic changed; or wrong window                     | Check `report.json.window` and `report.json.walk_forward` against the YAML. Then `git log -- src/quant/backtest/`. |
| `n_rebalances` differs from 37                                                   | window dates or `test_days` changed                                         | Diff the YAML against the canonical one in this repo.                                                                |

---

## 9. The one-line summary

`code_sha`, `config_hash`, `data_fingerprint` are bit-exact and
independently recomputable. The Sharpe rounds to **1.7253** and the
DSR P to **0.998** every time, on Python 3.12 + polars 1.39.3 + lightgbm
4.6.0; the trailing decimal of the equity curve drifts at the
floating-point ULP boundary due to Polars' multi-threaded reductions —
documented as a known issue, not a credibility issue.

The manifest's claim is honest: it tells you what *went into* the run,
not what came out. The numbers that came out are stable to nine
significant digits — more precision than any Sharpe deserves.
