"""
Pull a fundamentals snapshot from FMP /stable/quote for a set of symbols
and write a flat CSV the ValueSignal can read.

Schema:
    symbol, price, pe, eps, market_cap, fetched_at_utc

Free-tier rate-limit-friendly: 250 calls/day on /stable/quote, 1 call
per symbol. For 503 names that's two days of free-tier headroom; we
add a 0.25s sleep between calls so a single run uses ~125s of API time
and stays well under per-minute caps.

Usage:
    python -m scripts.fetch_fundamentals --universe SP500 \
        --out data/raw/fundamentals_sp500.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from quant.adapters.fmp import FmpAdapter
from quant.universe.constituents import DEV_UNIVERSE, fetch_sp500

log = logging.getLogger("quant.scripts.fetch_fundamentals")


async def _resolve_universe(name: str) -> list[str]:
    if name == "DEV":
        return list(DEV_UNIVERSE)
    if name == "SP500":
        rows = await fetch_sp500()
        return sorted({r["symbol"].strip() for r in rows if r.get("symbol", "").strip()})
    raise SystemExit(f"unknown universe: {name}")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="SP500", choices=["DEV", "SP500"])
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between symbol calls (rate-limit headroom)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    symbols = await _resolve_universe(args.universe)
    log.info("fetching fundamentals: universe=%s symbols=%d → %s", args.universe, len(symbols), args.out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    n_failed = 0
    fetched_at = datetime.now(UTC).isoformat()

    # Finnhub /stock/metric returns valuation metrics including P/E on the
    # free tier — FMP's /stable/quote gates `pe` as a premium field, so we
    # use Finnhub instead. Both keys are already required by the platform.
    from quant.adapters.finnhub import FinnhubAdapter

    async with FinnhubAdapter() as a, FmpAdapter() as fmp:
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["symbol", "price", "pe", "eps", "market_cap", "fetched_at_utc"])
            for sym in symbols:
                pe_val: float | str = ""
                eps_val: float | str = ""
                price_val: float | str = ""
                mcap_val: float | str = ""
                try:
                    metric = await a.get_json(
                        "/stock/metric", params={"symbol": sym, "metric": "all"}
                    )
                except Exception as exc:
                    n_failed += 1
                    log.warning("  %s metric: %s", sym, exc)
                    await asyncio.sleep(args.sleep)
                    continue
                if isinstance(metric, dict):
                    m = metric.get("metric", {}) if isinstance(metric.get("metric"), dict) else {}
                    pe_val = (
                        m.get("peNormalizedAnnual")
                        or m.get("peExclExtraAnnual")
                        or m.get("peTTM")
                        or ""
                    )
                    eps_val = (
                        m.get("epsNormalizedAnnual") or m.get("epsAnnual") or m.get("epsTTM") or ""
                    )
                    mcap_val = m.get("marketCapitalization") or ""
                # Price comes from the FMP /stable/quote endpoint which IS
                # free. Failure on price doesn't kill the row — value
                # signal only needs P/E.
                try:
                    quote = await fmp.get_json("/stable/quote", params={"symbol": sym})
                except Exception as exc:
                    log.debug("  %s price (FMP): %s", sym, exc)
                else:
                    if isinstance(quote, list) and quote and isinstance(quote[0], dict):
                        price_val = quote[0].get("price", "")
                w.writerow([sym, price_val, pe_val, eps_val, mcap_val, fetched_at])
                rows_written += 1
                await asyncio.sleep(args.sleep)

    log.info("done: %d rows, %d failed → %s", rows_written, n_failed, out_path)


if __name__ == "__main__":
    asyncio.run(main())
