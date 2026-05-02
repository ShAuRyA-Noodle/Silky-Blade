"""Tests for the prices-CSV verifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant.data.verify import verify_prices_csv


def _write(tmp_path: Path, content: str, name: str = "prices.csv") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------
def test_clean_csv_passes(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "date,symbol,adj_close\n"
        "2020-01-02,AAA,100.0\n"
        "2020-01-03,AAA,101.0\n"
        "2020-01-06,AAA,102.5\n"
        "2020-01-02,BBB,50.0\n"
        "2020-01-03,BBB,49.5\n"
        "2020-01-06,BBB,51.0\n",
    )
    rep = verify_prices_csv(p)
    assert rep.ok
    assert rep.n_errors == 0
    assert rep.rows == 6
    assert rep.symbols == 2


# ------------------------------------------------------------------
# Each error case in isolation
# ------------------------------------------------------------------
def test_missing_file_reports_error(tmp_path: Path) -> None:
    rep = verify_prices_csv(tmp_path / "does-not-exist.csv")
    assert not rep.ok
    assert any(i.code == "file_missing" for i in rep.issues)


def test_missing_columns_reports_error(tmp_path: Path) -> None:
    p = _write(tmp_path, "date,symbol,close\n2020-01-02,AAA,100.0\n")
    rep = verify_prices_csv(p)
    assert not rep.ok
    assert any(i.code == "missing_columns" for i in rep.issues)


def test_null_prices_reports_error(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "date,symbol,adj_close\n2020-01-02,AAA,100.0\n2020-01-03,AAA,\n",
    )
    rep = verify_prices_csv(p)
    assert not rep.ok
    assert any(i.code == "null_values" for i in rep.issues)


def test_zero_price_reports_error(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "date,symbol,adj_close\n2020-01-02,AAA,100.0\n2020-01-03,AAA,0.0\n",
    )
    rep = verify_prices_csv(p)
    assert not rep.ok
    assert any(i.code == "non_positive_prices" for i in rep.issues)


def test_negative_price_reports_error(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "date,symbol,adj_close\n2020-01-02,AAA,100.0\n2020-01-03,AAA,-1.0\n",
    )
    rep = verify_prices_csv(p)
    assert not rep.ok
    assert any(i.code == "non_positive_prices" for i in rep.issues)


def test_duplicate_keys_reports_error(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "date,symbol,adj_close\n2020-01-02,AAA,100.0\n2020-01-02,AAA,101.0\n2020-01-03,AAA,102.0\n",
    )
    rep = verify_prices_csv(p)
    assert not rep.ok
    issue = next(i for i in rep.issues if i.code == "duplicate_keys")
    assert issue.detail["distinct_dup_keys"] == 1
    assert issue.detail["extra_rows"] == 1


# ------------------------------------------------------------------
# Warning-level checks (don't fail the file)
# ------------------------------------------------------------------
def test_large_gap_emits_warning_not_error(tmp_path: Path) -> None:
    # 30-day gap between two rows for AAA
    p = _write(
        tmp_path,
        "date,symbol,adj_close\n2020-01-02,AAA,100.0\n2020-02-15,AAA,105.0\n2020-02-18,AAA,106.0\n",
    )
    rep = verify_prices_csv(p)
    assert rep.ok  # warnings only, no errors
    assert rep.n_warnings >= 1
    assert any(i.code == "large_gap" and i.severity == "warning" for i in rep.issues)


def test_unsorted_dates_emits_warning(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "date,symbol,adj_close\n2020-01-03,AAA,101.0\n2020-01-02,AAA,100.0\n2020-01-06,AAA,102.0\n",
    )
    rep = verify_prices_csv(p)
    assert rep.ok
    assert any(i.code == "unsorted_dates" for i in rep.issues)


# ------------------------------------------------------------------
# End-to-end on the real demo CSV — must pass cleanly
# ------------------------------------------------------------------
@pytest.mark.skipif(
    not Path("examples/backtest/sp500_5yr_adjusted.csv").exists(),
    reason="adapted demo CSV not present (regenerable via prepare_sp500_5yr.py)",
)
def test_real_demo_csv_passes() -> None:
    rep = verify_prices_csv("examples/backtest/sp500_5yr_adjusted.csv")
    # Real CSV should have NO errors. Warnings (gaps over weekends/holidays)
    # are tolerated.
    assert rep.ok, [f"{i.severity}: {i.code}: {i.message}" for i in rep.issues if i.severity == "error"]
    assert rep.rows > 100_000
    assert rep.symbols > 400
