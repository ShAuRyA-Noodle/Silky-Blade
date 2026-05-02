"""
Validate a prices CSV before it enters the backtest engine.

Bad data causes inflated Sharpes silently. The walk-forward engine trusts
its inputs — `walk_forward(prices, ...)` does not check for duplicate
(date, symbol) rows or forward-filled gaps; it just consumes them. A
duplicate row with a different price can drop into the wrong test slice
and gift the strategy a free pip.

This module fails loud. Each check is binary; a single failure means the
file is unfit for backtest. Output is a structured `VerifyReport` so the
caller can render it however they like (CLI, JSON, web).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = ("date", "symbol", "adj_close")
MAX_GAP_TRADING_DAYS = 5  # > 1 trading week without a row is suspicious


@dataclass(frozen=True)
class VerifyIssue:
    code: str
    severity: str  # "error" | "warning"
    message: str
    detail: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifyReport:
    path: str
    rows: int
    symbols: int
    date_min: _date | None
    date_max: _date | None
    issues: tuple[VerifyIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


# ------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------
def _check_columns(df: pl.DataFrame) -> list[VerifyIssue]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [
            VerifyIssue(
                code="missing_columns",
                severity="error",
                message=f"missing required columns: {', '.join(missing)}",
                detail={"required": ", ".join(REQUIRED_COLUMNS)},
            )
        ]
    return []


def _check_dtypes(df: pl.DataFrame) -> list[VerifyIssue]:
    issues: list[VerifyIssue] = []
    if df.schema["date"] != pl.Date:
        issues.append(
            VerifyIssue(
                code="bad_date_dtype",
                severity="error",
                message=f"date column must be Date, got {df.schema['date']}",
            )
        )
    if df.schema["adj_close"] not in (pl.Float64, pl.Float32):
        issues.append(
            VerifyIssue(
                code="bad_price_dtype",
                severity="error",
                message=f"adj_close must be Float, got {df.schema['adj_close']}",
            )
        )
    return issues


def _check_nulls(df: pl.DataFrame) -> list[VerifyIssue]:
    issues: list[VerifyIssue] = []
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            n_null = int(df[col].null_count())
            if n_null > 0:
                issues.append(
                    VerifyIssue(
                        code="null_values",
                        severity="error",
                        message=f"{n_null} null values in {col!r}",
                        detail={"column": col, "count": n_null},
                    )
                )
    return issues


def _check_prices(df: pl.DataFrame) -> list[VerifyIssue]:
    issues: list[VerifyIssue] = []
    px = df["adj_close"]
    n_nonpos = int((px <= 0).sum())
    if n_nonpos > 0:
        issues.append(
            VerifyIssue(
                code="non_positive_prices",
                severity="error",
                message=f"{n_nonpos} rows have adj_close <= 0",
                detail={"count": n_nonpos},
            )
        )
    n_nonfinite = int((~px.is_finite()).sum())
    if n_nonfinite > 0:
        issues.append(
            VerifyIssue(
                code="non_finite_prices",
                severity="error",
                message=f"{n_nonfinite} rows have non-finite adj_close (NaN/Inf)",
                detail={"count": n_nonfinite},
            )
        )
    return issues


def _check_duplicates(df: pl.DataFrame) -> list[VerifyIssue]:
    grp = df.group_by(["date", "symbol"]).agg(pl.len().alias("n"))
    dups = grp.filter(pl.col("n") > 1)
    if dups.height == 0:
        return []
    n_total = int(dups["n"].sum() - dups.height)
    sample = dups.sort("n", descending=True).head(3)
    sample_msg = ", ".join(
        f"({row['date']!s}, {row['symbol']}, x{row['n']})" for row in sample.iter_rows(named=True)
    )
    return [
        VerifyIssue(
            code="duplicate_keys",
            severity="error",
            message=f"{dups.height} (date, symbol) keys appear more than once "
            f"({n_total} extra rows). Sample: {sample_msg}",
            detail={"distinct_dup_keys": dups.height, "extra_rows": n_total},
        )
    ]


def _check_sorted(df: pl.DataFrame) -> list[VerifyIssue]:
    issues: list[VerifyIssue] = []
    # Per-symbol date monotonicity. Out-of-order dates can confuse downstream
    # walk-forward windowing if it relies on input order.
    for sym, group in df.group_by("symbol", maintain_order=False):
        dates = group.sort("date")["date"].to_list()
        original = group["date"].to_list()
        if dates != original:
            issues.append(
                VerifyIssue(
                    code="unsorted_dates",
                    severity="warning",
                    message=f"symbol {sym[0]!r}: dates not in order in source CSV",
                    detail={"symbol": str(sym[0])},
                )
            )
            # One warning per file (not per symbol) is enough.
            break
    return issues


def _check_gaps(df: pl.DataFrame) -> list[VerifyIssue]:
    """Flag any per-symbol gap larger than MAX_GAP_TRADING_DAYS calendar days."""
    issues: list[VerifyIssue] = []
    sorted_df = df.sort(["symbol", "date"])
    for sym, group in sorted_df.group_by("symbol", maintain_order=False):
        diffs = group["date"].diff().dt.total_days().drop_nulls()
        if diffs.is_empty():
            continue
        max_gap = int(diffs.max())  # type: ignore[arg-type]
        # 5 trading days ~ 7 calendar days; pad a bit for holidays.
        if max_gap > MAX_GAP_TRADING_DAYS + 4:
            issues.append(
                VerifyIssue(
                    code="large_gap",
                    severity="warning",
                    message=f"symbol {sym[0]!r}: max date gap is {max_gap} calendar days",
                    detail={"symbol": str(sym[0]), "max_gap_days": max_gap},
                )
            )
    return issues


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def verify_prices_csv(path: str | Path) -> VerifyReport:
    """Run every check; return a structured report. Never raises on data issues."""
    p = Path(path)
    if not p.exists():
        return VerifyReport(
            path=str(p),
            rows=0,
            symbols=0,
            date_min=None,
            date_max=None,
            issues=(
                VerifyIssue(
                    code="file_missing",
                    severity="error",
                    message=f"file not found: {p}",
                ),
            ),
        )

    # Read up front; if even reading fails, surface it as a single error.
    try:
        df = pl.read_csv(str(p), try_parse_dates=True)
    except Exception as exc:
        return VerifyReport(
            path=str(p),
            rows=0,
            symbols=0,
            date_min=None,
            date_max=None,
            issues=(
                VerifyIssue(
                    code="read_failed",
                    severity="error",
                    message=f"polars failed to read CSV: {exc.__class__.__name__}: {exc}",
                ),
            ),
        )

    issues: list[VerifyIssue] = []
    issues.extend(_check_columns(df))
    if any(i.code == "missing_columns" for i in issues):
        # Other checks assume the columns exist.
        return VerifyReport(
            path=str(p),
            rows=int(df.height),
            symbols=0,
            date_min=None,
            date_max=None,
            issues=tuple(issues),
        )
    df = df.with_columns(pl.col("date").cast(pl.Date))
    issues.extend(_check_dtypes(df))
    issues.extend(_check_nulls(df))
    issues.extend(_check_prices(df))
    issues.extend(_check_duplicates(df))
    issues.extend(_check_sorted(df))
    issues.extend(_check_gaps(df))

    return VerifyReport(
        path=str(p),
        rows=int(df.height),
        symbols=int(df["symbol"].n_unique()),
        date_min=df["date"].min(),  # type: ignore[arg-type]
        date_max=df["date"].max(),  # type: ignore[arg-type]
        issues=tuple(issues),
    )


def write_csv_repro(report: VerifyReport, path: str | Path) -> None:
    """Optional helper: dump the issue list to a CSV for triage."""
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["severity", "code", "message"])
        for issue in report.issues:
            w.writerow([issue.severity, issue.code, issue.message])


__all__ = [
    "MAX_GAP_TRADING_DAYS",
    "REQUIRED_COLUMNS",
    "VerifyIssue",
    "VerifyReport",
    "verify_prices_csv",
    "write_csv_repro",
]
