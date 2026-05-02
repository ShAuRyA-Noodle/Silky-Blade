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
app.add_typer(universe_app, name="universe")
app.add_typer(backfill_app, name="backfill")
app.add_typer(flow_app, name="flow")
app.add_typer(backtest_app, name="backtest")
app.add_typer(ml_app, name="ml")
app.add_typer(paper_app, name="paper")


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


if __name__ == "__main__":
    app()
