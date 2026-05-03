"""
Pull daily bars from Alpaca for a configurable universe + date range and
write a CSV in the schema the backtest runner expects:

    date, symbol, adj_close

Alpaca's free IEX feed covers all S&P 500 names back to 2016. Output goes
to `data/raw/alpaca_<universe>_<start>_<end>.csv`.

Usage:
    python -m scripts.backfill_alpaca_to_csv \
        --universe SP500 --start 2018-01-01 --end 2026-05-02 \
        --out data/raw/alpaca_sp500_2018_2026.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant.adapters.alpaca import AlpacaDataAdapter
from quant.universe.constituents import DEV_UNIVERSE, fetch_sp500

log = logging.getLogger("quant.scripts.backfill_alpaca")


async def _resolve_universe(name: str) -> list[str]:
    if name == "DEV":
        return list(DEV_UNIVERSE)
    if name == "SP500":
        rows = await fetch_sp500()
        # Alpaca won't accept tickers with dots; replace with hyphens (BRK.B → BRK.B
        # actually IS accepted by Alpaca as is). Drop any with whitespace.
        return sorted({r["symbol"].strip() for r in rows if r.get("symbol", "").strip()})
    raise SystemExit(f"unknown universe: {name}")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="SP500", choices=["DEV", "SP500"])
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--batch", type=int, default=100, help="symbols per Alpaca call")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=UTC) + timedelta(days=1)

    symbols = await _resolve_universe(args.universe)
    log.info(
        "backfill: universe=%s symbols=%d window=%s → %s out=%s",
        args.universe,
        len(symbols),
        args.start,
        args.end,
        args.out,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    failed_batches = 0
    async with AlpacaDataAdapter() as a:
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"])

            n_batches = (len(symbols) + args.batch - 1) // args.batch
            for i in range(0, len(symbols), args.batch):
                batch = symbols[i : i + args.batch]
                log.info("batch %d/%d (%d symbols)", i // args.batch + 1, n_batches, len(batch))
                try:
                    bars: dict[str, list[dict[str, Any]]] = await a.bars(
                        batch,
                        timeframe="1Day",
                        start=start_dt,
                        end=end_dt,
                        adjustment="split",
                    )
                except Exception as exc:
                    failed_batches += 1
                    log.warning("batch failed: %s — %s", batch[:3], exc)
                    continue
                for sym, sym_bars in bars.items():
                    if not isinstance(sym_bars, list):
                        continue
                    for bar in sym_bars:
                        ts = bar.get("t")
                        c = bar.get("c")
                        if ts is None or c is None:
                            continue
                        try:
                            d = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
                        except ValueError:
                            continue
                        # Alpaca's `adjustment="split"` already split-adjusts c.
                        # Use the same value for `close` and `adj_close`.
                        w.writerow(
                            [
                                d.isoformat(),
                                sym,
                                bar.get("o", c),
                                bar.get("h", c),
                                bar.get("l", c),
                                c,
                                bar.get("v", 0),
                                c,
                            ]
                        )
                        rows_written += 1

    log.info("done: %d rows, %d failed batches → %s", rows_written, failed_batches, out_path)


if __name__ == "__main__":
    asyncio.run(main())
