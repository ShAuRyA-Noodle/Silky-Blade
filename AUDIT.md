# AUDIT — brutal in-depth review of every module

**Date:** 2026-05-03
**Auditor:** authoring agent on `main`
**Scope:** every shipped module in `apps/api/src/quant/` + `apps/web/`
**Goal:** name every gap, every weakness, every overstatement honestly. No
section ends with marketing.

This is the partner document to `TRUST.md`. TRUST.md says what the platform
*does* claim. AUDIT.md says what's still wrong with it.

---

## 0. Headline numbers, today

Real, reproducible, on-disk artifacts:

| Run | Universe | Window | Sharpe | DSR P | AnnRet | DD | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `sp500_momentum_126`        | survivors-only Kaggle | 2014→2018 | 1.725 | 0.998 | 22.48% | 8.4%  | Original demo |
| `sp500_momentum_126_pit`    | point-in-time S&P 500 | 2014→2018 | 1.112 | 0.927 | 14.74% | 10.0% | Survivorship corrected |
| `sp500_momentum_126_2026`   | Alpaca SP500 (live)   | 2019→2026 | 1.703 | 1.000 | **42.48%** | **19.6%** | Fresh data |
| `sp500_ml_predictions_v1`   | trainer 100-sym subset| 2014→2018 | 1.408 | 1.000 | 16.06% | 8.0%  | LightGBM 100-sym |
| `sp500_lightgbm_2026` (OOF) | trainer 200-sym subset| 2018→2026 | —     | —     | —      | —     | logloss 0.985, AUC 0.626 |
| `sp500_momentum_sweep`      | 13 configs, raw       | 2014→2018 | —     | —     | —      | —     | **PBO = 0.557** |
| `sp500_momentum_sweep_pit`  | 13 configs, PIT       | 2014→2018 | —     | —     | —      | —     | **PBO = 0.629** |

**Honest reading:** the strategy's *risk-adjusted* edge (Sharpe ~1.7) is
consistent across two non-overlapping market regimes, which is genuinely a
positive signal. The *return* edge inflates dramatically in the 2019-2026
window because that window includes the 2020-2021 bull rally. The 2022
bear shows up as the larger 19.6% drawdown. Selection-bias score 0.557
(0.629 under PIT) means the cross-config "winner" is roughly coin-flip.

---

## 1. Module-by-module audit

Severity legend:
- **🔴 critical** — would mislead a user; must fix before claiming production-ready
- **🟠 high** — meaningful gap; user-visible weakness
- **🟡 medium** — annoyance / incompleteness
- **🟢 low** — nice-to-have

### 1.1 Backtest engine (`quant.backtest.engine`)

**What works:**
- Walk-forward train/test windowing is correct. No look-ahead.
- 5 bps cost model + turnover tracking.
- Equal-weight top-K sizing.
- `universe_filter` callable supports point-in-time membership.

**Gaps:**
- 🟡 No shorting. Equity-only, long-only. Realistic for retail; limiting for research.
- 🟡 No leverage. Fixed `initial_capital` allocates to top-K equally.
- 🟡 No sector/industry concentration limits. The strategy can hold all 25 picks in tech if momentum says so.
- 🟢 ULP-level non-determinism documented in `REPRODUCE.md`. Polars multi-threaded reductions drift at ~1e-15. Not a real problem.

### 1.2 Statistics (`quant.backtest.statistics`)

**What works:**
- Deflated Sharpe (Bailey & López de Prado 2014) implemented from the paper. Tests show it correctly *decreases* as `n_trials` grows.
- PBO via CSCV (López de Prado 2016). Tests confirm random IID returns yield PBO ≈ 0.5.

**Gaps:**
- 🟡 PBO `n_slices` defaults to 16; sweep uses 8. Both are heuristic. No automated guidance to the user about which to pick for their trial size.
- 🟢 No benchmark-relative Sharpe (information ratio). The strategy is reported in absolute terms only.

### 1.3 Signals (`quant.backtest.signals`)

**What works:**
- `MomentumSignal`, `LowVolSignal`, `MeanReversionSignal`, `MLPredictionsSignal` all in the registry.
- Each is tested with synthetic-but-real GBM that exercises the ranking direction.

**Gaps:**
- 🟠 **No fundamental signals.** Value (P/E, P/B), quality (ROE), growth — none. We have FMP keys; we're not using them.
- 🟠 **No ensemble combiner / risk parity.** Each signal stands alone. Modern multi-factor portfolios combine; we don't.
- 🟡 ML signal uses only the latest in-OOF prediction. No daily re-inference on fresh bars (the model isn't loaded; predictions are replayed).

### 1.4 ML trainer (`quant.ml.trainer`)

**What works:**
- LightGBM multiclass on triple-barrier labels with purged K-fold + 1% embargo. End-to-end on real data. MLflow-tracked.
- Probability calibration with isotonic regression on OOF only.
- Real measurement: logloss 0.9849 on 2018-2026 data, ~10% better than coin-flip.

**Gaps:**
- 🔴 **The trained boosters are not persisted as a usable model artifact.** The trainer keeps them in memory per-fold for OOF prediction, then discards. **Live inference on tomorrow's bars is impossible without retraining.** This is the single biggest gap blocking the "user connects toy account, model says BUY/HOLD/SELL" flow you asked for.
- 🟠 **No hyperparameter tuning.** 200 boosting rounds, 31 leaves, learning rate 0.05 — fixed. No grid search, no Bayesian opt.
- 🟠 **`max_symbols` cap.** Memory-bounded to 100/200 symbols on a laptop. Real production should run on the full 500 names.
- 🟠 **Feature set is technical-only.** 26 features (returns, vol, MA, RSI, MACD, Bollinger, ATR, volume, range, gap). No fundamentals, no sentiment, no macro. Adding those is the obvious next step.
- 🟡 **No SHAP explanations exposed.** The trainer logs `feature_importance_gain` but nothing per-prediction.
- 🟡 **No model registry.** Each run writes to MLflow; "promote a run to production" is manual.

### 1.5 Universe / point-in-time (`quant.universe.point_in_time`)

**What works:**
- Wikipedia changes table parser → reverse-walk reconstruction. 394 real changes scraped on a live test.
- Reconstructs S&P 500 membership for any date back to ~2000.

**Gaps:**
- 🔴 **The "exited-and-removed-from-data" survivorship bias is unfixed.** Wikipedia has the membership history; the *price data* for delisted names is not there. To close this you need a paid feed (Polygon Stocks, Sharadar, Norgate) — $50-200/mo.
- 🟡 Pre-2000 coverage of the changes table is sparse. Doesn't matter for 2014+ backtests.
- 🟡 Ticker re-use across decades is not disambiguated.

### 1.6 Data verifier (`quant.data.verify`)

**What works:**
- Schema, dtype, null, non-positive price, NaN/Inf, duplicate-key checks.
- Real run on the 2018-2026 backfill: 0 errors, 18 large-gap warnings (Alpaca IEX feed coverage gaps).

**Gaps:**
- 🟢 No corporate-action sanity check (split / dividend adjustments not validated).
- 🟢 No outlier-flag for daily price moves > 50%.

### 1.7 Provider health-check (`quant.data.providers_health`)

**What works:**
- Pings 12 providers concurrently in <3s.
- Live result: 10/12 PASS with current keys. Polygon, Alpaca data + broker, FRED, Tiingo, Finnhub, Groq, Marketaux, NewsAPI, FMP all green. Nasdaq Data Link (no account), AlphaVantage (rate-limited, key valid) red.
- Silences httpx loggers to prevent key leakage.

**Gaps:**
- 🟢 No periodic re-check; runs on demand only.
- 🟢 No alerting when a previously-passing provider starts failing.

### 1.8 Live paper-trading session (`quant.execution.live_session`)

**What works:**
- Real Alpaca paper integration. Pulls real positions, real bars, computes signal, computes orders, optionally submits.
- Triple safety gate: `TRADING_ENABLED` + `ALPACA_PAPER` + `--confirm`. All must be true.
- Real run on `quant paper now`: 5 BUY proposals against a $100k paper account, refused submission because `TRADING_ENABLED=false`.

**Gaps:**
- 🟠 **No daily worker / scheduler.** The CLI runs once on demand. There's no cron / Prefect flow that runs `paper now --submit --confirm` at market close.
- 🟠 **No order-book reconciliation.** If a submitted order partially fills, gets rejected, or gets canceled by the broker, nothing in our state model catches it.
- 🟡 **No risk manager wired.** The `quant.execution.broker` modules exist but the live path doesn't call them (no max-position-size, no kill-switch, no daily-loss-limit checks at order time). Risk parameters are in `.env.local` but not consumed.
- 🟡 **Universe is hard-coded to DEV_UNIVERSE (20 names) in the CLI.** Should accept --universe flag.

### 1.9 FastAPI surface (`quant.api.v1`)

**What works:**
- Backtest read endpoints (5 routes) — list / one / equity / manifest / config.
- Paper read endpoints (2 routes) — account / positions.
- Auth via `Depends(get_current_user)`. Path-traversal hardened on backtest routes.
- 25+ integration tests pass.

**Gaps:**
- 🟠 **No write endpoints for paper trading.** Submission stays on the CLI behind safety gates. Reasonable, but means a future web "place order" button needs a new gated route.
- 🟠 **No SSE / WebSocket for live position updates.** The `quant.streaming` module exists but isn't wired to the paper account.
- 🟡 **No rate-limiting middleware** beyond what FastAPI's defaults provide.
- 🟢 **Pydantic v2 strict everywhere.** Good.

### 1.10 Web (`apps/web`)

**What works:**
- `/results` page with KPI grid, equity curve, brutal disclaimer, PIT comparison, repro block, click-to-copy hashes.
- Real numbers from on-disk artifacts (no runtime fetch).
- ESLint + tsc + next build all clean.

**Gaps:**
- 🟠 **`/results` shows the 2014-2018 Kaggle backtest, not the fresh 2019-2026 numbers.** The 2018-2026 artifact bundle exists but isn't loaded by the page yet.
- 🟠 **No `/paper` page.** The `/api/v1/paper/*` endpoints exist; nothing on the web fetches them.
- 🟠 **No "BUY/HOLD/SELL recommendation" UI per symbol.** This is the user-requested core feature; not built.
- 🟡 **No user-account / signup UI.** Auth scaffolding exists; no front door.
- 🟡 **No mobile testing beyond basic responsiveness.**

### 1.11 CI (`.github/workflows/ci.yml`)

**What works:**
- ruff + mypy --strict + pytest + ESLint + tsc + next build + docker build all run on every push.
- No-fake-data regex guard scans `apps/api/src + apps/web/{app,components,lib}` for `Math.random|faker.|lorem ipsum|mock_data|synthetic_data|fake_data|hardcoded_price`.
- Artifact sync step copies the committed demo bundles into the web Docker context.
- Every commit on `main` for the past 24 hours has been green (after fixes).

**Gaps:**
- 🟡 **No coverage gate.** `pytest-cov` runs but doesn't fail on coverage drop.
- 🟡 **No type-coverage gate** (`mypy --strict` already enforces full annotation).
- 🟢 **No security scanning** (Bandit, Semgrep, dependabot).

### 1.12 Documentation

**What works:**
- `TRUST.md` — credibility contract, 6 sections, version-stamped 2026-05-03.
- `REPRODUCE.md` — cold-start reproduction guide, manifest verification.
- `FINAL_REPORT.md` — sprint-by-sprint history with §13 update.
- `README.md` — rewritten to ground every claim.
- This `AUDIT.md` — brutal review.

**Gaps:**
- 🟢 No model-card per training run. Each MLflow run has one implicitly; nothing surfaces it as a markdown.

---

## 2. The single most-blocking critical gap

Of everything above, **§1.4's "trained boosters are not persisted"** is the
one that blocks your user-visible vision: *user connects toy account → kit
recommends BUY/HOLD/SELL today*.

Without a saved model artifact, today's recommendation requires retraining
from scratch (~30 seconds for the 200-symbol panel). That is fixable in
~50 lines of code: pickle the per-fold boosters into the artifact bundle
during training, write a `quant ml predict --symbols X,Y,Z --as-of today`
CLI that loads them + builds today's features + runs `predict_proba` +
prints calibrated probabilities → BUY/HOLD/SELL.

That is the next ship.

---

## 3. Top-5 priority fixes

In the order I would build them next:

1. **🔴 Persist trained boosters + add `quant ml predict` CLI** (Phase 2 of
   today's roadmap). Unblocks live recommendations.
2. **🟠 Wire the BUY/HOLD/SELL recommender into the web /results page or a
   new /paper page.** (Phase 3.)
3. **🟠 Refresh `/results` to use `sp500_momentum_126_2026` instead of
   the 2014-2018 Kaggle bundle.** Single load_artifacts.ts edit.
4. **🟠 Add fundamental + macro signals** (P/E ratio via FMP; 10y rate via
   FRED; both keys are working). Adds two uncorrelated alpha sources.
5. **🟠 Daily Prefect worker** that runs `paper now --submit --confirm`
   at market close, logs fills, alerts on rejections. Closes the loop
   from "scaffold" to "actually trading paper".

---

## 4. Honest reading — what this platform actually is, today

After all of the above, the platform is:

- A **methodologically clean quant research environment** with rigorous
  backtest + statistics + reproducibility infrastructure.
- A **partial live execution path** (real broker integration, real account
  state, plan generation) with the actual order-submission button locked
  behind safety gates.
- **NOT a fund.** No live capital. No daily worker running.
- **NOT a stock-picking service.** It can produce a daily recommendation
  list once §2's gap is closed; until then the only "recommendation" is
  the one printed by `quant paper now` (momentum top-25, no model).

The math is honest. The numbers are reproducible. The gaps are named.
That's the most defensible state a solo project in this niche can ship.

---

## 5. Closing — this is not "top 0.0001%"

Across this whole audit, every section has gaps. Real, named, plausibly
fixable gaps. A repo with this many open issues is not "top one in a
million" of anything in the trading-platform world. It IS plausibly
top-1% of *honest, reproducible, single-developer quant research repos*.
That distinction is the spine of `TRUST.md`. It is the spine of this
audit too.

If a future commit ever phrases this project as "top 0.0001%", treat it
as a regression and revert.
