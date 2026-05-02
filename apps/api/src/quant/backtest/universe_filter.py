"""
Universe filters for the walk-forward engine.

A `UniverseFilter` is `Callable[[date], set[str] | None]`. The walk-forward
engine calls it on every rebalance date with the as-of date and intersects
the signal-producer's output with whatever the filter returns. This is how
the runner enforces *point-in-time* membership when the underlying price
data is biased (e.g. the Kaggle S&P 500 5y CSV only contains symbols that
survived to its cutoff).

What a filter CAN fix when given survivors-only price data:
- "Joined-after" forward-looking bias: a symbol IPO'd or was added to the
  S&P 500 after the rebalance date but is in the dataset because it was
  in the index at the cutoff. Without a filter, the strategy can pick it
  using future information about its index inclusion. WITH the filter,
  it's excluded until its actual inclusion date.

What a filter CAN'T fix:
- "Exited-and-removed-from-data" survivorship bias: stocks that left the
  S&P 500 (delisted, acquired) before the cutoff are absent from the
  price data entirely. The filter would have included them, but the
  prices module has no rows. Closing this gap requires a price-data
  vendor with delisted-name coverage (Polygon Stocks, Sharadar, Norgate).

This module provides one factory, `point_in_time_sp500_filter`, that
fetches Wikipedia changes once at construction time and returns a
closure that's cheap to call inside the rebalance loop.
"""

from __future__ import annotations

from datetime import date

from quant.backtest.engine import UniverseFilter
from quant.universe.point_in_time import (
    fetch_sp500_changes,
    members_as_of,
)


def point_in_time_sp500_filter(
    current_members: set[str] | frozenset[str] | None = None,
) -> UniverseFilter:
    """
    Return a `UniverseFilter` that resolves S&P 500 membership at any date.

    `current_members`: the anchor set (today's S&P 500). If omitted, the
    filter calls `quant.universe.constituents.fetch_sp500()` itself —
    blocking, network-bound — so callers that already have the set
    should pass it in for speed.
    """
    if current_members is None:
        # Resolve today's set lazily, async-fetched. Local import keeps
        # this module's static deps minimal.
        import asyncio

        from quant.universe.constituents import fetch_sp500

        current_members = {row["symbol"] for row in asyncio.run(fetch_sp500())}

    changes = fetch_sp500_changes()
    anchor = frozenset(current_members)

    def _filter(rebalance_date: date) -> set[str]:
        return set(members_as_of(rebalance_date, changes, anchor))

    return _filter


__all__ = ["point_in_time_sp500_filter"]
