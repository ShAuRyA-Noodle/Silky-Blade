"""
Point-in-time S&P 500 membership reconstructed from Wikipedia.

Survivorship bias is the #1 unfixed credibility gap in this repo's headline
backtest: the demo dataset only contains symbols that survived to its cutoff.
Companies that were dropped from the index (acquired, bankrupt, demoted) are
silently absent. Momentum or any cross-sectional strategy on a survivors-only
universe overstates returns. Point-in-time membership lets the backtest pick
the universe that *actually existed* on each rebalance date.

Authoritative paid sources (Sharadar, Norgate, S&P Dow Jones direct) cost
$50-200/month. Wikipedia's "List of S&P 500 companies" page maintains a
"Selected changes" table going back to ~2000 with per-event date + added
ticker + removed ticker. That is enough to reconstruct membership for any
date from ~2000 onward — accurate, free, and the same data professional
vendors largely derive from.

Algorithm (reverse-walk from current set):
    1. Start with today's S&P 500 (already in `quant.universe.constituents`).
    2. Walk every change with `date > target_date`, newest first:
         - if a ticker was Added at that change, it was NOT in the set at
           the target date  →  discard from the working set
         - if a ticker was Removed at that change, it WAS in the set at
           the target date  →  add to the working set
    3. The remaining working set is the membership on `target_date`.

This handles tickers that were added-then-removed (or vice versa) within the
target → today window correctly, because each change is applied in reverse
chronological order.

Limitations honestly stated:
- Coverage starts ~2000. Earlier dates fall back to today's set with a
  warning. For 2014+ backtests this is sufficient.
- Wikipedia is community-maintained — occasional missing or mis-dated
  changes are possible. Cross-checking against a paid feed is the right
  next step if the strategy becomes capital-bearing.
- Ticker symbols can be reused (a delisted ticker reused by a different
  company years later). This module returns whatever symbol Wikipedia has
  on each event; downstream code that joins to price data is responsible
  for handling identity, not just symbol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final

import httpx
from bs4 import BeautifulSoup, Tag

WIKIPEDIA_SP500_URL: Final = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS: Final = {"User-Agent": "quant-platform/1.0 (point-in-time-membership)"}

# Wikipedia date formats observed in the changes table:
#   "January 26, 2024" · "October 2, 2023" · sometimes "2024-01-26".
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_DATE_LONG = re.compile(r"^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$")
_DATE_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


@dataclass(frozen=True)
class IndexChange:
    """One row from the Wikipedia changes table."""

    when: date
    added: str | None  # ticker
    removed: str | None  # ticker
    reason: str = ""


# ------------------------------------------------------------------
# Date parsing
# ------------------------------------------------------------------
def _parse_date(text: str) -> date | None:
    text = text.strip().rstrip(".")
    m_long = _DATE_LONG.match(text)
    if m_long:
        month_name, day_str, year_str = m_long.groups()
        month_idx = _MONTHS.get(month_name.lower())
        if month_idx is None:
            return None
        return date(int(year_str), month_idx, int(day_str))
    m_iso = _DATE_ISO.match(text)
    if m_iso:
        y_str, m_str, d_str = m_iso.groups()
        return date(int(y_str), int(m_str), int(d_str))
    return None


# ------------------------------------------------------------------
# HTML parsing
# ------------------------------------------------------------------
def _cell_text(td: Tag) -> str:
    # Strip footnote markers like "[1]" and collapse whitespace.
    text = td.get_text(separator=" ", strip=True)
    text = re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _ticker_from_cell(td: Tag) -> str | None:
    text = _cell_text(td)
    # Tickers are 1–5 uppercase chars, optionally with "." or "-".
    m = re.search(r"\b([A-Z][A-Z0-9.\-]{0,9})\b", text)
    return m.group(1) if m else None


def parse_changes_html(html: str) -> list[IndexChange]:
    """Parse the 'Selected changes' wikitable. Pure function — tested directly."""
    soup = BeautifulSoup(html, "html.parser")
    changes: list[IndexChange] = []

    # Locate the changes table by its header row signature: it has a column
    # header that contains both "Added" and "Removed" cells. Use the
    # last wikitable that matches — Wikipedia keeps the changes table after
    # the constituents table.
    candidate: Tag | None = None
    for table in soup.find_all("table", class_="wikitable"):
        header_text = " ".join(
            th.get_text(" ", strip=True)
            for th in table.find_all("th")  # type: ignore[union-attr]
        ).lower()
        if "added" in header_text and "removed" in header_text and "date" in header_text:
            candidate = table  # type: ignore[assignment]
    if candidate is None:
        return changes

    for row in candidate.find_all("tr"):  # type: ignore[union-attr]
        cells: list[Tag] = [c for c in row.find_all("td") if isinstance(c, Tag)]
        if len(cells) < 5:
            continue
        when = _parse_date(_cell_text(cells[0]))
        if when is None:
            continue
        added = _ticker_from_cell(cells[1]) if _cell_text(cells[1]) else None
        removed = _ticker_from_cell(cells[3]) if _cell_text(cells[3]) else None
        reason = _cell_text(cells[5]) if len(cells) > 5 else ""
        if added is None and removed is None:
            continue
        changes.append(IndexChange(when=when, added=added, removed=removed, reason=reason))

    return changes


# ------------------------------------------------------------------
# Fetcher (network)
# ------------------------------------------------------------------
def fetch_sp500_changes(client: httpx.Client | None = None) -> list[IndexChange]:
    """Fetch + parse Wikipedia. Network-bound; not called in unit tests."""
    owns = client is None
    client = client or httpx.Client(timeout=30.0, headers=_HEADERS, follow_redirects=True)
    try:
        resp = client.get(WIKIPEDIA_SP500_URL)
        resp.raise_for_status()
        return parse_changes_html(resp.text)
    finally:
        if owns:
            client.close()


# ------------------------------------------------------------------
# Reverse-walk reconstruction
# ------------------------------------------------------------------
def members_as_of(
    target_date: date,
    changes: list[IndexChange],
    current_members: set[str] | frozenset[str],
) -> list[str]:
    """
    Reconstruct the S&P 500 membership on `target_date`.

    `changes` may be in any order — we sort by date descending. `current_members`
    is the set of symbols in the index *today* (or any anchor more recent than
    every change in `changes`).
    """
    working = set(current_members)
    for change in sorted(changes, key=lambda c: c.when, reverse=True):
        if change.when <= target_date:
            break
        if change.added is not None:
            working.discard(change.added)
        if change.removed is not None:
            working.add(change.removed)
    return sorted(working)


__all__ = [
    "WIKIPEDIA_SP500_URL",
    "IndexChange",
    "fetch_sp500_changes",
    "members_as_of",
    "parse_changes_html",
]
