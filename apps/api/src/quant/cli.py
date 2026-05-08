"""
Quant platform CLI.

    quant universe bootstrap [--enrich]
    quant backfill ohlcv  --start 2016-01-01 [--end YYYY-MM-DD] [--universe SP500|NDX100|SP500_NDX100|DEV]
    quant backfill macro  [--start YYYY-MM-DD]
    quant backfill news   [--hours 24]
    quant flow bootstrap  [--years 10]

Run via `python -m quant.cli ...` or the entrypoint `quant ...`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Coroutine
from datetime import date, timedelta
from typing import Annotated, Any

import typer

from quant.config import settings
from quant.universe.constituents import DEV_UNIVERSE

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
universe_app = typer.Typer(help="Universe operations.")
backfill_app = typer.Typer(help="Data backfills.")
flow_app = typer.Typer(help="Run Prefect flows locally (no orchestrator).")
backtest_app = typer.Typer(help="Walk-forward backtest runner + repro manifest.")
ml_app = typer.Typer(help="ML trainer (LightGBM, triple-barrier, purged K-fold).")
paper_app = typer.Typer(help="Paper-trading planning (no broker submission yet).")
data_app = typer.Typer(help="Data-quality utilities (CSV verification, etc.).")
features_app = typer.Typer(help="Feature pipelines (sentiment, fundamentals, etc.).")
app.add_typer(universe_app, name="universe")
app.add_typer(backfill_app, name="backfill")
app.add_typer(flow_app, name="flow")
app.add_typer(backtest_app, name="backtest")
app.add_typer(ml_app, name="ml")
app.add_typer(paper_app, name="paper")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")


def _setup_logging() -> None:
    logging.basicConfig(
        level=settings.app_log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    _setup_logging()
    return asyncio.run(coro)


# ---------------------------------------------------------------
# universe
# ---------------------------------------------------------------
@universe_app.command("bootstrap")
def universe_bootstrap(
    enrich: Annotated[bool, typer.Option(help="Also enrich metadata via Polygon")] = False,
) -> None:
    """Seed SP500 + NDX100 tickers + membership."""
    from quant.universe import bootstrap_universe

    res = _run(bootstrap_universe(enrich_with_polygon=enrich))
    typer.echo(json.dumps(res, indent=2))


@universe_app.command("list")
def universe_list(
    universe: Annotated[str, typer.Option(help="SP500 | NDX100 | DEV")] = "SP500",
) -> None:
    from quant.universe.loader import active_universe_symbols

    if universe == "DEV":
        typer.echo("\n".join(DEV_UNIVERSE))
        return
    syms = _run(active_universe_symbols(universe))
    typer.echo(f"# {universe}: {len(syms)} symbols")
    typer.echo("\n".join(syms))


@universe_app.command("point-in-time")
def universe_point_in_time(
    as_of: Annotated[str, typer.Option(help="YYYY-MM-DD — date to reconstruct")],
    output: Annotated[str, typer.Option(help="Optional: write symbols to a file")] = "",
) -> None:
    """Reconstruct point-in-time S&P 500 membership via Wikipedia changes table."""
    from quant.universe.constituents import fetch_sp500
    from quant.universe.point_in_time import fetch_sp500_changes, members_as_of

    _setup_logging()
    target = date.fromisoformat(as_of)
    changes = fetch_sp500_changes()
    current = {row["symbol"] for row in _run(fetch_sp500())}
    syms = members_as_of(target, changes, current)
    typer.echo(f"# SP500 as of {target}: {len(syms)} symbols (vs {len(current)} today)")
    if output:
        from pathlib import Path

        Path(output).write_text("\n".join(syms) + "\n", encoding="utf-8")
        typer.echo(f"# wrote {output}")
    else:
        typer.echo("\n".join(syms))


# ---------------------------------------------------------------
# backfill
# ---------------------------------------------------------------
def _resolve_symbols(universe: str) -> list[str]:
    from quant.universe.loader import active_universe_symbols

    if universe == "DEV":
        return list(DEV_UNIVERSE)
    if universe == "SP500_NDX100":
        a = _run(active_universe_symbols("SP500"))
        b = _run(active_universe_symbols("NDX100"))
        return sorted(set(a + b))
    return _run(active_universe_symbols(universe))


@backfill_app.command("ohlcv")
def backfill_ohlcv(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD inclusive")] = "",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD inclusive; default = yesterday")] = "",
    universe: Annotated[str, typer.Option(help="SP500 | NDX100 | SP500_NDX100 | DEV")] = "SP500_NDX100",
    years: Annotated[int, typer.Option(help="Convenience: years back from today if --start omitted")] = 10,
) -> None:
    from quant.ingest.ohlcv import backfill_ohlcv_daily

    start_d = date.fromisoformat(start) if start else (date.today() - timedelta(days=365 * years))
    end_d = date.fromisoformat(end) if end else (date.today() - timedelta(days=1))
    symbols = _resolve_symbols(universe)
    typer.echo(f"OHLCV backfill: {len(symbols)} symbols, {start_d} → {end_d}")
    res = _run(backfill_ohlcv_daily(symbols, start=start_d, end=end_d))
    ok = sum(1 for v in res.values() if v > 0)
    total = sum(res.values())
    typer.echo(f"Done: {ok}/{len(symbols)} symbols, {total} rows")


@backfill_app.command("corp-actions")
def backfill_corp_actions(
    universe: Annotated[str, typer.Option()] = "SP500_NDX100",
) -> None:
    from quant.ingest.corporate_actions import ingest_corporate_actions

    symbols = _resolve_symbols(universe)
    res = _run(ingest_corporate_actions(symbols))
    typer.echo(f"{sum(res.values())} corp-action rows across {len(symbols)} symbols")


@backfill_app.command("macro")
def backfill_macro(
    start: Annotated[str, typer.Option()] = "",
) -> None:
    from quant.ingest.macro import ingest_macro_series

    start_d = date.fromisoformat(start) if start else None
    res = _run(ingest_macro_series(start=start_d))
    typer.echo(json.dumps(res, indent=2))


@backfill_app.command("news")
def backfill_news(
    hours: Annotated[int, typer.Option()] = 24,
    universe: Annotated[str, typer.Option()] = "SP500_NDX100",
) -> None:
    from quant.ingest.news import ingest_news

    symbols = _resolve_symbols(universe)
    res = _run(ingest_news(symbols=symbols, lookback_hours=hours))
    typer.echo(json.dumps(res, indent=2))


# ---------------------------------------------------------------
# flow (local, no Prefect server required)
# ---------------------------------------------------------------
@flow_app.command("bootstrap")
def flow_bootstrap(
    years: Annotated[int, typer.Option()] = 10,
    enrich: Annotated[bool, typer.Option()] = False,
) -> None:
    from quant.workers.flows import bootstrap_flow

    res = _run(bootstrap_flow(years=years, enrich=enrich))
    typer.echo(json.dumps(res, indent=2))


@flow_app.command("daily-close")
def flow_daily_close() -> None:
    from quant.workers.flows import daily_close_flow

    res = _run(daily_close_flow())
    typer.echo(json.dumps(res, indent=2))


@flow_app.command("hourly-news")
def flow_hourly_news() -> None:
    from quant.workers.flows import hourly_news_flow

    res = _run(hourly_news_flow())
    typer.echo(json.dumps(res, indent=2))


# ---------------------------------------------------------------
# backtest
# ---------------------------------------------------------------
@backtest_app.command("run")
def backtest_run(
    config: Annotated[str, typer.Argument(help="Path to a YAML or JSON run config")],
) -> None:
    """Run one walk-forward backtest end-to-end and write the artifact bundle."""
    from quant.backtest.runner import load_config, run_backtest

    _setup_logging()
    cfg = load_config(config)
    typer.echo(f"Running backtest '{cfg.name}' ({cfg.start_date} → {cfg.end_date})")
    report = run_backtest(cfg)
    m = report["metrics"]
    typer.echo(
        f"Done. Sharpe={m['sharpe']:.3f}  DSR P={m['deflated_sharpe_p']:.3f}  "
        f"DD={m['max_drawdown']:.1%}  Turnover={m['turnover']:.1f}x  "
        f"AnnRet={m['annualized_return']:.2%}  AnnVol={m['annualized_vol']:.2%}"
    )
    typer.echo(f"Artifacts: {report['artifacts']['dir']}")


@backtest_app.command("sweep")
def backtest_sweep(
    config: Annotated[str, typer.Argument(help="Path to a YAML or JSON sweep config")],
) -> None:
    """Run a multi-config sweep and emit cross-config PBO + per-run DSR."""
    from quant.backtest.sweep import load_sweep_config, run_sweep

    _setup_logging()
    cfg = load_sweep_config(config)
    typer.echo(
        f"Running sweep '{cfg.name}' ({cfg.start_date} → {cfg.end_date})  "
        f"n_runs={len(cfg.runs)}  cscv_S={cfg.n_slices}"
    )
    report = run_sweep(cfg)
    typer.echo(
        f"PBO={report['pbo']:.3f}  n_obs/config={report['n_observations_per_config']}  "
        f"cscv_trials={report['cscv_n_trials']}"
    )
    for r in report["runs"]:
        typer.echo(
            f"  {r['name']:<28}  Sharpe={r['sharpe']:>6.3f}  "
            f"DSR P={r['deflated_sharpe_p']:>6.3f}  "
            f"DD={r['max_drawdown']:>6.1%}  Turn={r['turnover']:>5.1f}x"
        )
    typer.echo(f"Artifacts: {report['artifacts']['dir']}")


# ---------------------------------------------------------------
# ml
# ---------------------------------------------------------------
@ml_app.command("train")
def ml_train(
    config: Annotated[str, typer.Argument(help="Path to a YAML or JSON train config")],
) -> None:
    """Train one LightGBM model with purged K-fold + MLflow + artifact bundle."""
    from quant.ml import load_config, train

    _setup_logging()
    cfg = load_config(config)
    typer.echo(
        f"Training '{cfg.name}' on {cfg.data.prices_csv} ({cfg.data.start_date} → {cfg.data.end_date})"
    )
    report = train(cfg)
    oof = report["oof_metrics"]
    typer.echo(
        f"Done. logloss={oof['oof_logloss']:.4f}  "
        f"bal_acc={oof['oof_balanced_accuracy']:.4f}  "
        f"macro_auc_ovr={oof['oof_macro_auc_ovr']:.4f}"
    )
    typer.echo(f"MLflow run: {report['mlflow_run_id']}")
    typer.echo(f"Artifacts:  {report['artifacts']['dir']}")


@ml_app.command("tune")
def ml_tune(
    config: Annotated[str, typer.Argument(help="Path to a YAML or JSON train config (defaults inherited)")],
    n_trials: Annotated[int, typer.Option(help="Optuna trials; ~30s each on M2 air")] = 30,
    seed: Annotated[int, typer.Option(help="TPE sampler seed")] = 42,
    json_out: Annotated[str, typer.Option(help="Path to write the best-params JSON")] = "",
) -> None:
    """
    Search LightGBM hyperparameters with Optuna TPE; minimize OOF log-loss.
    Prints best params + history; writes JSON report when --json-out is set.
    """
    from quant.ml import load_config
    from quant.ml.tune import tune

    _setup_logging()
    base = load_config(config)
    typer.echo(f"Tuning '{base.name}' — {n_trials} TPE trials")
    rep = tune(base, n_trials=n_trials, seed=seed)
    typer.echo(f"# best logloss = {rep.best_value:.4f} (over {rep.n_trials} trials)")
    typer.echo("# best params:")
    for k, v in sorted(rep.best_params.items()):
        typer.echo(f"  {k:<20} {v}")
    if json_out:
        from pathlib import Path as _Path

        _Path(json_out).write_text(
            json.dumps(
                {
                    "n_trials": rep.n_trials,
                    "best_value": rep.best_value,
                    "best_params": rep.best_params,
                    "history": rep.history,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        typer.echo(f"# JSON written to {json_out}")


@ml_app.command("predict")
def ml_predict(
    model_dir: Annotated[str, typer.Argument(help="Path to trainer artifact bundle")],
    prices_csv: Annotated[str, typer.Argument(help="Prices CSV (date, symbol, OHLCV)")],
    as_of: Annotated[str, typer.Option(help="YYYY-MM-DD; defaults to most recent date in the CSV")] = "",
    symbols: Annotated[str, typer.Option(help="Comma-separated symbols (default = all in panel)")] = "",
    threshold: Annotated[float, typer.Option(help="|score| threshold for BUY/SELL (else HOLD)")] = 0.10,
    explain: Annotated[bool, typer.Option(help="Print top SHAP drivers per recommendation")] = False,
    drivers_k: Annotated[int, typer.Option(help="Number of top features when --explain")] = 5,
    json_out: Annotated[str, typer.Option(help="Optional JSON output path")] = "",
) -> None:
    """
    Load a trained model + emit BUY/HOLD/SELL recommendations per symbol.

    The model must have been trained with the current `quant.ml.trainer`
    (which persists per-fold boosters + isotonic calibrators + feature meta
    inside the artifact bundle).
    """
    from datetime import date as _date

    import polars as pl

    from quant.ml.predict import load_bundle, recommend

    _setup_logging()
    bundle = load_bundle(model_dir)
    df = pl.read_csv(prices_csv, try_parse_dates=True).with_columns(pl.col("date").cast(pl.Date))
    target_date = _date.fromisoformat(as_of) if as_of else df["date"].max()
    if not isinstance(target_date, _date):
        raise typer.BadParameter(f"could not resolve as_of from CSV: {target_date!r}")

    sym_list: list[str] | None = None
    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    recs = recommend(
        bundle,
        df,
        as_of=target_date,
        symbols=sym_list,
        threshold=threshold,
        explain=explain,
        top_k_drivers=drivers_k,
    )

    typer.echo(f"# {len(recs)} recommendations as of {target_date} (threshold={threshold:.2f}):")
    typer.echo(f"  {'sym':<6} {'action':<6} {'conf':<6} {'P(-)':<6} {'P(0)':<6} {'P(+)':<6} {'score':<7}")
    for r in sorted(recs, key=lambda r: -r.score):
        typer.echo(
            f"  {r.symbol:<6} {r.action:<6} {r.confidence:<6} "
            f"{r.prob_neg1:.3f}  {r.prob_zero:.3f}  {r.prob_pos1:.3f}  {r.score:+.3f}"
        )
        if explain and r.top_drivers:
            for d in r.top_drivers:
                arrow = "↑" if d.contribution > 0 else "↓"
                typer.echo(
                    f"      {arrow} {d.feature:<22} value={d.value:>+9.4f}  contrib={d.contribution:>+8.4f}"
                )
    n_buy = sum(1 for r in recs if r.action == "BUY")
    n_sell = sum(1 for r in recs if r.action == "SELL")
    n_hold = sum(1 for r in recs if r.action == "HOLD")
    typer.echo(f"# {n_buy} BUY, {n_sell} SELL, {n_hold} HOLD")

    if json_out:
        from pathlib import Path as _Path

        def _rec_payload(r: Any) -> dict[str, Any]:
            base: dict[str, Any] = {
                "symbol": r.symbol,
                "as_of": r.as_of.isoformat(),
                "action": r.action,
                "confidence": r.confidence,
                "score": r.score,
                "prob_neg1": r.prob_neg1,
                "prob_zero": r.prob_zero,
                "prob_pos1": r.prob_pos1,
            }
            if r.top_drivers:
                base["top_drivers"] = [
                    {
                        "feature": d.feature,
                        "value": d.value,
                        "contribution": d.contribution,
                    }
                    for d in r.top_drivers
                ]
            return base

        _Path(json_out).write_text(
            json.dumps([_rec_payload(r) for r in recs], indent=2),
            encoding="utf-8",
        )
        typer.echo(f"# JSON written to {json_out}")


# ---------------------------------------------------------------
# paper — plan paper-trading orders from a signal config
# ---------------------------------------------------------------
@paper_app.command("plan")
def paper_plan(
    config: Annotated[str, typer.Argument(help="Backtest YAML/JSON — uses its signal + walk_forward.top_k")],
    as_of: Annotated[str, typer.Option(help="YYYY-MM-DD as-of date for scoring")],
    portfolio_value: Annotated[float, typer.Option(help="Total dollars to allocate")] = 100_000.0,
    positions: Annotated[
        str, typer.Option(help="Path to positions JSON ([{symbol, quantity, last_price}, ...]); empty = flat")
    ] = "",
    output: Annotated[str, typer.Option(help="Optional path to write the JSON plan")] = "",
) -> None:
    """
    Compute the orders that would bring `positions` to the model's target
    allocation as-of `as_of`. No broker submission. Output is a JSON plan
    a human reviews before any trade is placed.
    """
    from datetime import date as _date
    from decimal import Decimal as _Decimal
    from pathlib import Path as _Path

    import polars as pl

    from quant.backtest.runner import build_signal, load_config, load_prices_csv
    from quant.execution.paper_session import (
        Position,
        TargetAllocation,
        compute_target_orders,
    )

    _setup_logging()
    cfg = load_config(config)
    target_date = _date.fromisoformat(as_of)

    prices = load_prices_csv(cfg.prices_csv, cfg.start_date, target_date)
    if prices.is_empty():
        raise typer.BadParameter(f"no prices for {cfg.prices_csv} up to {target_date}")

    producer = build_signal(cfg.signal)
    sigs = producer(target_date, prices)
    if sigs.is_empty():
        raise typer.BadParameter(f"signal returned empty at {target_date}")

    top_k = cfg.walk_forward.top_k
    top = sigs.sort("score", descending=True).head(top_k)
    syms = top["symbol"].to_list()
    weight_each = 1.0 / len(syms)
    target_weights = dict.fromkeys(syms, weight_each)

    # Resolve latest close per top-k symbol from the price panel.
    last_prices_df = (
        prices.filter(pl.col("symbol").is_in(syms))
        .sort(["symbol", "date"])
        .group_by("symbol", maintain_order=True)
        .agg(pl.col("adj_close").last().alias("last"))
    )
    last_prices = {row["symbol"]: _Decimal(str(row["last"])) for row in last_prices_df.iter_rows(named=True)}

    # Read current positions if provided, else flat.
    current_positions: list[Position] = []
    if positions:
        with open(positions, encoding="utf-8") as fh:
            for entry in json.load(fh):
                sym = str(entry["symbol"])
                current_positions.append(
                    Position(
                        symbol=sym,
                        quantity=_Decimal(str(entry["quantity"])),
                        last_price=_Decimal(str(entry.get("last_price", "0"))),
                    )
                )
                # Carry over user-supplied price if our panel doesn't have it.
                if sym not in last_prices and "last_price" in entry:
                    last_prices[sym] = _Decimal(str(entry["last_price"]))

    target = TargetAllocation(
        weights=target_weights,
        portfolio_value=_Decimal(str(portfolio_value)),
    )
    proposals = compute_target_orders(
        current_positions=current_positions,
        target=target,
        latest_prices=last_prices,
    )

    plan = {
        "as_of": target_date.isoformat(),
        "config_name": cfg.name,
        "signal_kind": cfg.signal.kind,
        "top_k": top_k,
        "portfolio_value": float(portfolio_value),
        "target_weights": target_weights,
        "proposed_orders": [
            {
                "symbol": p.symbol,
                "side": p.side,
                "quantity": str(p.quantity),
                "delta_shares": str(p.delta_shares),
                "target_value": str(p.target_value),
                "current_value": str(p.current_value),
            }
            for p in proposals
        ],
    }

    payload = json.dumps(plan, indent=2, default=str)
    if output:
        _Path(output).write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"Plan written to {output} ({len(proposals)} orders)")
    else:
        typer.echo(payload)


# ---------------------------------------------------------------
# data — quality verification
# ---------------------------------------------------------------
@data_app.command("providers-health")
def data_providers_health(
    json_out: Annotated[str, typer.Option(help="Optional path to write the JSON report")] = "",
    fail_on_error: Annotated[bool, typer.Option(help="Exit 1 if any configured provider fails")] = False,
) -> None:
    """Ping every keyed data provider; print PASS/FAIL + latency."""
    from quant.data.providers_health import check_all

    _setup_logging()
    # Silence httpx + adapter loggers — health-check URLs include API keys
    # for some providers (Polygon, FRED, AlphaVantage). Echoing them to a
    # terminal is a leak vector. The user's terminal scrollback / shared
    # screenshots / pasted bug reports must not contain key material.
    for noisy in ("httpx", "httpcore", "quant.adapter"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    results = _run(check_all())
    width = max(len(r.name) for r in results)
    n_pass = n_fail = n_skip = 0
    for r in results:
        if not r.configured:
            n_skip += 1
            mark = "skip"
        elif r.ok:
            n_pass += 1
            mark = "pass"
        else:
            n_fail += 1
            mark = "FAIL"
        latency = f"{r.latency_ms}ms" if r.latency_ms is not None else "—"
        typer.echo(f"  {r.name:<{width}}  {mark:>4}  {latency:>6}  {r.detail}")
    typer.echo(f"# {n_pass} pass, {n_fail} fail, {n_skip} not configured")
    if json_out:
        from pathlib import Path as _Path

        payload = [
            {
                "name": r.name,
                "configured": r.configured,
                "ok": r.ok,
                "detail": r.detail,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]
        _Path(json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if fail_on_error and n_fail > 0:
        raise typer.Exit(code=1)


@data_app.command("verify")
def data_verify(
    path: Annotated[str, typer.Argument(help="Path to a prices CSV (date,symbol,adj_close)")],
    json_out: Annotated[str, typer.Option(help="Optional path to write the report as JSON")] = "",
) -> None:
    """Verify a prices CSV is fit for backtest. Exits 1 on any error."""
    from quant.data import verify_prices_csv

    _setup_logging()
    rep = verify_prices_csv(path)

    typer.echo(
        f"# {rep.path}: rows={rep.rows} symbols={rep.symbols} "
        f"window={rep.date_min}→{rep.date_max} "
        f"errors={rep.n_errors} warnings={rep.n_warnings}"
    )
    for issue in rep.issues:
        typer.echo(f"  [{issue.severity}] {issue.code}: {issue.message}")

    if json_out:
        from pathlib import Path as _Path

        payload = {
            "path": rep.path,
            "rows": rep.rows,
            "symbols": rep.symbols,
            "date_min": rep.date_min.isoformat() if rep.date_min else None,
            "date_max": rep.date_max.isoformat() if rep.date_max else None,
            "ok": rep.ok,
            "n_errors": rep.n_errors,
            "n_warnings": rep.n_warnings,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "detail": i.detail,
                }
                for i in rep.issues
            ],
        }
        _Path(json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        typer.echo(f"# JSON report → {json_out}")

    if not rep.ok:
        raise typer.Exit(code=1)


@paper_app.command("status")
def paper_status(
    json_out: Annotated[str, typer.Option(help="Optional path to write the JSON state")] = "",
    history_csv: Annotated[
        str,
        typer.Option(help="Append today's snapshot to this CSV (creates if missing)"),
    ] = "",
) -> None:
    """
    Read the live Alpaca paper account state — equity, cash, buying power,
    open positions with mark-to-market PnL. Read-only; never sends an order.
    """
    from quant.adapters.alpaca import AlpacaBrokerAdapter
    from quant.execution.live_session import (
        fetch_account_snapshot,
        fetch_current_positions,
    )

    _setup_logging()
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    async def _go() -> None:
        from decimal import Decimal as _Decimal

        adapter = AlpacaBrokerAdapter()
        try:
            snap = await fetch_account_snapshot(adapter)
            positions = await fetch_current_positions(adapter)
            # Re-pull raw positions to extract avg_entry + market_value for PnL.
            raw_positions = await adapter.positions()
        finally:
            await adapter.aclose()

        raw_by_sym = {r.get("symbol", ""): r for r in raw_positions if isinstance(r, dict)}

        typer.echo(
            f"# account: equity=${snap.equity}  cash=${snap.cash}  "
            f"buying_power=${snap.buying_power}  status={snap.status}  "
            f"paper={snap.paper}"
        )
        if not positions:
            typer.echo("# no open positions")
        else:
            typer.echo(f"# {len(positions)} open positions:")
            total_pnl = _Decimal("0")
            total_market = _Decimal("0")
            for p in positions:
                raw = raw_by_sym.get(p.symbol, {})
                avg_entry = _Decimal(str(raw.get("avg_entry_price", "0")))
                market_value = _Decimal(str(raw.get("market_value", "0")))
                upl = _Decimal(str(raw.get("unrealized_pl", "0")))
                upl_pc = _Decimal(str(raw.get("unrealized_plpc", "0")))
                total_pnl += upl
                total_market += market_value
                typer.echo(
                    f"  {p.symbol:<6} qty={p.quantity:<6}  "
                    f"avg=${avg_entry}  last=${p.last_price}  "
                    f"market=${market_value}  uPnL=${upl} ({upl_pc:.2%})"
                )
            typer.echo(f"# total_market=${total_market}  total_unrealized_pnl=${total_pnl}")

        if json_out:
            from pathlib import Path as _Path

            payload = {
                "account": {
                    "equity": str(snap.equity),
                    "cash": str(snap.cash),
                    "buying_power": str(snap.buying_power),
                    "status": snap.status,
                    "paper": snap.paper,
                },
                "positions": [
                    {
                        "symbol": p.symbol,
                        "quantity": str(p.quantity),
                        "last_price": str(p.last_price),
                        "raw": raw_by_sym.get(p.symbol, {}),
                    }
                    for p in positions
                ],
            }
            _Path(json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            typer.echo(f"# JSON written to {json_out}")

        if history_csv:
            import csv as _csv
            from datetime import UTC as _UTC
            from datetime import datetime as _datetime
            from pathlib import Path as _Path

            row = {
                "timestamp": _datetime.now(_UTC).isoformat(),
                "equity": str(snap.equity),
                "cash": str(snap.cash),
                "buying_power": str(snap.buying_power),
                "status": snap.status,
                "n_positions": str(len(positions)),
            }
            history_path = _Path(history_csv)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not history_path.exists()
            with history_path.open("a", newline="", encoding="utf-8") as fh:
                w = _csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)
            typer.echo(f"# history row appended to {history_csv}")

    _run(_go())


@paper_app.command("now")
def paper_now(
    signal_kind: Annotated[
        str,
        typer.Option(help="momentum | low_vol | mean_reversion | ml_bundle | value | sentiment | composite"),
    ] = "momentum",
    lookback_days: Annotated[int, typer.Option(help="Lookback for the signal AND for bar fetch")] = 126,
    top_k: Annotated[int, typer.Option(help="Number of positions to hold")] = 5,
    portfolio_value: Annotated[
        float, typer.Option(help="Override broker equity (default = use account)")
    ] = 0.0,
    universe: Annotated[
        str,
        typer.Option(help="DEV (20 names) | SP500 (full index) | comma-separated symbols"),
    ] = "DEV",
    model_dir: Annotated[
        str,
        typer.Option(
            help="When --signal-kind ml_bundle (or composite primary=ml_bundle), trainer artifact dir"
        ),
    ] = "",
    fundamentals_csv: Annotated[
        str,
        typer.Option(help="When --signal-kind value, path to fundamentals CSV"),
    ] = "",
    sentiment_csv: Annotated[
        str,
        typer.Option(help="When --signal-kind sentiment OR composite, path to sentiment CSV"),
    ] = "",
    sentiment_lookback_days: Annotated[
        int,
        typer.Option(help="Days of sentiment history to average over"),
    ] = 3,
    alpha: Annotated[
        float,
        typer.Option(help="When --signal-kind composite, weight on the ML/primary side (0..1)"),
    ] = 0.7,
    submit: Annotated[bool, typer.Option(help="Actually submit orders (otherwise plan-only)")] = False,
    confirm: Annotated[
        bool, typer.Option(help="Required alongside --submit before any order is sent")
    ] = False,
) -> None:
    """
    Live paper-trading session: pull current Alpaca paper positions + recent
    Alpaca data bars, compute the signal at today's date, propose orders.
    Submission requires --submit AND --confirm AND TRADING_ENABLED=true AND
    ALPACA_PAPER=true (any False => plan-only).
    """
    from quant.adapters.alpaca import AlpacaBrokerAdapter, AlpacaDataAdapter
    from quant.backtest.runner import SignalSpec, build_signal
    from quant.execution.broker import AlpacaBroker
    from quant.execution.live_session import run_live_session
    from quant.execution.risk_gate import RiskLimits
    from quant.universe.constituents import DEV_UNIVERSE

    _setup_logging()
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if signal_kind == "ml_bundle":
        if not model_dir:
            raise typer.BadParameter("--signal-kind ml_bundle requires --model-dir")
        spec = SignalSpec(kind="ml_bundle", params={"model_dir": model_dir})
    elif signal_kind == "value":
        if not fundamentals_csv:
            raise typer.BadParameter("--signal-kind value requires --fundamentals-csv")
        spec = SignalSpec(kind="value", params={"fundamentals_csv": fundamentals_csv})
    elif signal_kind == "sentiment":
        if not sentiment_csv:
            raise typer.BadParameter("--signal-kind sentiment requires --sentiment-csv")
        spec = SignalSpec(
            kind="sentiment",
            params={"sentiment_csv": sentiment_csv, "lookback_days": sentiment_lookback_days},
        )
    elif signal_kind == "composite":
        # Composite blends an ML bundle (primary) with a sentiment score
        # (secondary). Default α=0.7 means the model dominates; sentiment
        # is a tilt, not a takeover.
        if not model_dir:
            raise typer.BadParameter("--signal-kind composite requires --model-dir (primary=ml_bundle)")
        if not sentiment_csv:
            raise typer.BadParameter("--signal-kind composite requires --sentiment-csv")
        if not (0.0 <= alpha <= 1.0):
            raise typer.BadParameter(f"--alpha must be in [0,1], got {alpha}")
        spec = SignalSpec(
            kind="composite",
            params={
                "primary": {"kind": "ml_bundle", "params": {"model_dir": model_dir}},
                "secondary": {
                    "kind": "sentiment",
                    "params": {
                        "sentiment_csv": sentiment_csv,
                        "lookback_days": sentiment_lookback_days,
                    },
                },
                "alpha": alpha,
                "beta": 1.0 - alpha,
                "outer_join": True,  # don't drop names without recent news
            },
        )
    else:
        spec = SignalSpec(kind=signal_kind, params={"lookback_days": lookback_days})
    sig = build_signal(spec)

    if universe == "DEV":
        universe_list = list(DEV_UNIVERSE)
    elif universe == "SP500":
        from quant.universe.constituents import fetch_sp500

        rows = _run(fetch_sp500())
        universe_list = sorted({r["symbol"].strip() for r in rows if r.get("symbol", "").strip()})
    else:
        universe_list = [s.strip().upper() for s in universe.split(",") if s.strip()]
    if not universe_list:
        raise typer.BadParameter(f"empty universe resolved from {universe!r}")

    async def _go() -> None:
        from decimal import Decimal as _Decimal

        broker_adapter = AlpacaBrokerAdapter()
        data_adapter = AlpacaDataAdapter()
        broker = AlpacaBroker()
        try:
            risk_limits = RiskLimits(
                max_position_pct=settings.max_position_pct,
                max_positions=settings.max_positions,
                daily_loss_limit_pct=settings.daily_loss_limit_pct,
                drawdown_kill_pct=settings.drawdown_kill_pct,
            )
            result = await run_live_session(
                signal=sig,
                universe=universe_list,
                broker=broker,
                broker_adapter=broker_adapter,
                data_adapter=data_adapter,
                top_k=top_k,
                lookback_days=max(lookback_days + 30, 200),
                portfolio_value_override=_Decimal(str(portfolio_value)) if portfolio_value > 0 else None,
                trading_enabled=settings.trading_enabled,
                alpaca_paper=settings.alpaca_paper,
                confirm=confirm and submit,
                risk_limits=risk_limits,
            )
        finally:
            await broker_adapter.aclose()
            await data_adapter.aclose()
            await broker.aclose()

        typer.echo(
            f"# session={result.session_id} as_of={result.as_of} "
            f"account_equity=${result.account.equity} status={result.account.status} "
            f"paper={result.account.paper}"
        )
        typer.echo(
            f"# {result.n_symbols_scored} symbols scored, "
            f"top_{top_k} chosen: {', '.join(result.target_weights.keys())}"
        )
        typer.echo(f"# {len(result.proposals)} proposed orders:")
        for p in result.proposals:
            typer.echo(
                f"  {p.side:<4} {p.symbol:<6} qty={p.quantity:<5}  "
                f"target=${p.target_value}  current=${p.current_value}"
            )
        n_blocked = sum(1 for r in result.risk_results if not r.accepted)
        if n_blocked:
            typer.echo(f"# RISK GATE blocked {n_blocked} proposals:")
            for r in result.risk_results:
                if not r.accepted:
                    typer.echo(f"  BLOCK {r.proposal.side} {r.proposal.symbol}: {r.reason}")
        if result.submitted:
            typer.echo(f"# SUBMITTED {len(result.acks)} orders to Alpaca paper broker")
            for ack in result.acks:
                typer.echo(
                    f"  {ack.broker_order_id}  client_id={ack.client_order_id[:16]}  status={ack.status}"
                )
        else:
            typer.echo(
                "# plan-only mode — pass --submit AND --confirm AND set TRADING_ENABLED=true "
                "AND ALPACA_PAPER=true to actually send orders"
            )

    _run(_go())


# ---------------------------------------------------------------
# features — sentiment, fundamentals, etc.
# ---------------------------------------------------------------
@features_app.command("fetch-sentiment")
def features_fetch_sentiment(
    symbols: Annotated[
        str,
        typer.Option(help="Comma-separated symbols, OR 'DEV' for DEV_UNIVERSE"),
    ],
    out: Annotated[str, typer.Argument(help="Output CSV path")],
    days: Annotated[int, typer.Option(help="Lookback window in days (max ~30 for free APIs)")] = 7,
    use_marketaux: Annotated[bool, typer.Option(help="Pull from Marketaux")] = True,
    use_newsapi: Annotated[bool, typer.Option(help="Pull from NewsAPI")] = True,
) -> None:
    """
    Fetch news for symbols, score each via Groq, aggregate per (symbol, date),
    write CSV. Output schema: symbol, date, sentiment_mean, sentiment_count,
    sentiment_max_abs.
    """
    from quant.features.sentiment import fetch_and_score, write_sentiment_csv
    from quant.universe.constituents import DEV_UNIVERSE as _DEV

    _setup_logging()
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if symbols.upper() == "DEV":
        sym_list = list(_DEV)
    else:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        raise typer.BadParameter("--symbols resolved to empty list")

    typer.echo(f"# fetching last {days}d of news for {len(sym_list)} symbols")
    rows = _run(
        fetch_and_score(
            sym_list,
            days=days,
            use_marketaux=use_marketaux,
            use_newsapi=use_newsapi,
        )
    )
    write_sentiment_csv(rows, out)
    typer.echo(f"# {len(rows)} (symbol,date) rows → {out}")
    if rows:
        typer.echo("# sample (top 5 by |sentiment_mean|):")
        sorted_rows = sorted(rows, key=lambda r: -abs(float(r["sentiment_mean"])))[:5]
        for r in sorted_rows:
            typer.echo(
                f"  {r['symbol']:<6} {r['date']}  mean={r['sentiment_mean']:+.3f}  n={r['sentiment_count']}"
            )


if __name__ == "__main__":
    app()
