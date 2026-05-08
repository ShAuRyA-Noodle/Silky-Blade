"""Unit tests for sentiment feature module + SentimentSignal + CompositeSignal."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant.backtest.runner import SignalSpec, build_signal
from quant.backtest.signals import (
    CompositeSignal,
    MomentumSignal,
    SentimentSignal,
)
from quant.features.sentiment import (
    _aggregate_per_symbol_day,
    _ScoredArticle,
    write_sentiment_csv,
)


# ------------------------------------------------------------------
# _aggregate_per_symbol_day
# ------------------------------------------------------------------
def test_aggregate_groups_by_symbol_and_date() -> None:
    scored = [
        _ScoredArticle("AAPL", date(2026, 5, 1), 0.6, "bullish", "h1"),
        _ScoredArticle("AAPL", date(2026, 5, 1), 0.4, "neutral", "h2"),
        _ScoredArticle("AAPL", date(2026, 5, 2), -0.3, "bearish", "h3"),
        _ScoredArticle("MSFT", date(2026, 5, 1), 0.1, "neutral", "h4"),
    ]
    rows = _aggregate_per_symbol_day(scored)
    by_key = {(r["symbol"], r["date"]): r for r in rows}
    aapl_d1 = by_key[("AAPL", "2026-05-01")]
    assert aapl_d1["sentiment_count"] == 2
    assert aapl_d1["sentiment_mean"] == pytest.approx(0.5)
    assert aapl_d1["sentiment_max_abs"] == pytest.approx(0.6)
    aapl_d2 = by_key[("AAPL", "2026-05-02")]
    assert aapl_d2["sentiment_count"] == 1
    assert aapl_d2["sentiment_mean"] == pytest.approx(-0.3)
    assert aapl_d2["sentiment_max_abs"] == pytest.approx(-0.3)
    msft_d1 = by_key[("MSFT", "2026-05-01")]
    assert msft_d1["sentiment_count"] == 1


def test_aggregate_empty() -> None:
    assert _aggregate_per_symbol_day([]) == []


# ------------------------------------------------------------------
# write_sentiment_csv roundtrip
# ------------------------------------------------------------------
def test_write_csv_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "AAPL",
            "date": "2026-05-01",
            "sentiment_mean": 0.5,
            "sentiment_count": 2,
            "sentiment_max_abs": 0.6,
        }
    ]
    p = tmp_path / "sent.csv"
    write_sentiment_csv(rows, p)
    text = p.read_text(encoding="utf-8")
    assert "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs" in text
    assert "AAPL,2026-05-01,0.5,2,0.6" in text


# ------------------------------------------------------------------
# SentimentSignal
# ------------------------------------------------------------------
def _empty_history() -> pl.DataFrame:
    return pl.DataFrame({"date": [], "symbol": [], "adj_close": []})


def test_sentiment_signal_averages_lookback(tmp_path: Path) -> None:
    p = tmp_path / "sent.csv"
    p.write_text(
        "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs\n"
        "AAA,2026-04-29,0.2,1,0.2\n"
        "AAA,2026-04-30,0.6,1,0.6\n"
        "AAA,2026-05-01,1.0,1,1.0\n"
        "BBB,2026-05-01,-0.4,1,-0.4\n",
        encoding="utf-8",
    )
    sig = SentimentSignal(sentiment_csv=str(p), lookback_days=3)
    scores = sig(date(2026, 5, 1), _empty_history())
    got = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True)}
    # AAA mean over (2026-04-29, 04-30, 05-01) = (0.2 + 0.6 + 1.0) / 3 = 0.6
    assert got["AAA"] == pytest.approx(0.6)
    assert got["BBB"] == pytest.approx(-0.4)


def test_sentiment_signal_drops_outside_window(tmp_path: Path) -> None:
    p = tmp_path / "sent.csv"
    p.write_text(
        "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs\n"
        "OLD,2026-01-01,0.9,1,0.9\n"  # too old
        "NEW,2026-05-01,0.3,1,0.3\n",
        encoding="utf-8",
    )
    sig = SentimentSignal(sentiment_csv=str(p), lookback_days=3)
    scores = sig(date(2026, 5, 1), _empty_history())
    syms = scores["symbol"].to_list()
    assert "OLD" not in syms
    assert "NEW" in syms


def test_sentiment_signal_empty_when_no_window_overlap(tmp_path: Path) -> None:
    p = tmp_path / "sent.csv"
    p.write_text(
        "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs\nX,2020-01-01,0.5,1,0.5\n",
        encoding="utf-8",
    )
    sig = SentimentSignal(sentiment_csv=str(p), lookback_days=3)
    scores = sig(date(2026, 5, 1), _empty_history())
    assert scores.height == 0


def test_build_signal_sentiment(tmp_path: Path) -> None:
    p = tmp_path / "sent.csv"
    p.write_text(
        "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs\nX,2026-05-01,0.5,1,0.5\n",
        encoding="utf-8",
    )
    s = build_signal(
        SignalSpec(
            kind="sentiment",
            params={"sentiment_csv": str(p), "lookback_days": 5},
        )
    )
    assert isinstance(s, SentimentSignal)
    assert s.lookback_days == 5


def test_build_signal_sentiment_requires_csv() -> None:
    with pytest.raises(ValueError, match="sentiment_csv"):
        build_signal(SignalSpec(kind="sentiment", params={}))


# ------------------------------------------------------------------
# CompositeSignal
# ------------------------------------------------------------------
def _gbm_history(n_days: int = 50) -> pl.DataFrame:
    """Two-symbol synthetic GBM panel (UP trends up, DOWN trends down)."""
    from datetime import timedelta

    rows: list[dict[str, object]] = []
    start = date(2026, 3, 1)
    for i in range(n_days):
        d = start + timedelta(days=i)
        rows.append({"date": d, "symbol": "UP", "adj_close": 100.0 * (1.01**i)})
        rows.append({"date": d, "symbol": "DOWN", "adj_close": 100.0 * (0.99**i)})
    return pl.DataFrame(rows)


def test_composite_blends_two_signals(tmp_path: Path) -> None:
    """alpha*momentum + beta*sentiment, inner-joined on symbol."""
    p = tmp_path / "sent.csv"
    p.write_text(
        "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs\n"
        "UP,2026-04-19,0.5,1,0.5\n"
        "DOWN,2026-04-19,-0.5,1,-0.5\n",
        encoding="utf-8",
    )
    momentum = MomentumSignal(lookback_days=20)
    sentiment = SentimentSignal(sentiment_csv=str(p), lookback_days=3)
    comp = CompositeSignal(primary=momentum, secondary=sentiment, alpha=0.5, beta=0.5)
    scores = comp(date(2026, 4, 19), _gbm_history(n_days=50))
    by = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True)}
    # UP momentum is positive, sentiment +0.5 → positive score
    # DOWN momentum is negative, sentiment -0.5 → negative score
    assert by["UP"] > 0
    assert by["DOWN"] < 0
    assert by["UP"] > by["DOWN"]


def test_composite_outer_join_imputes_zero(tmp_path: Path) -> None:
    """When outer_join=True, missing-side gets 0; symbol stays in result."""
    p = tmp_path / "sent.csv"
    # only UP has sentiment; DOWN has none
    p.write_text(
        "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs\nUP,2026-04-19,0.5,1,0.5\n",
        encoding="utf-8",
    )
    momentum = MomentumSignal(lookback_days=20)
    sentiment = SentimentSignal(sentiment_csv=str(p), lookback_days=3)
    comp_inner = CompositeSignal(primary=momentum, secondary=sentiment, alpha=0.5, beta=0.5, outer_join=False)
    comp_outer = CompositeSignal(primary=momentum, secondary=sentiment, alpha=0.5, beta=0.5, outer_join=True)
    inner = comp_inner(date(2026, 4, 19), _gbm_history(n_days=50))
    outer = comp_outer(date(2026, 4, 19), _gbm_history(n_days=50))
    assert inner["symbol"].to_list() == ["UP"]  # DOWN dropped
    assert set(outer["symbol"].to_list()) == {"UP", "DOWN"}  # DOWN kept, sentiment=0


def test_composite_rejects_bad_weights() -> None:
    with pytest.raises(ValueError, match="alpha\\+beta"):
        CompositeSignal(
            primary=MomentumSignal(20),
            secondary=MomentumSignal(20),
            alpha=0.6,
            beta=0.5,
        )(date(2026, 1, 1), _empty_history())


def test_build_signal_composite(tmp_path: Path) -> None:
    p = tmp_path / "sent.csv"
    p.write_text(
        "symbol,date,sentiment_mean,sentiment_count,sentiment_max_abs\nX,2026-05-01,0.5,1,0.5\n",
        encoding="utf-8",
    )
    s = build_signal(
        SignalSpec(
            kind="composite",
            params={
                "primary": {"kind": "momentum", "params": {"lookback_days": 20}},
                "secondary": {
                    "kind": "sentiment",
                    "params": {"sentiment_csv": str(p), "lookback_days": 3},
                },
                "alpha": 0.7,
            },
        )
    )
    assert isinstance(s, CompositeSignal)
    assert s.alpha == pytest.approx(0.7)
    assert s.beta == pytest.approx(0.3)


def test_build_signal_composite_requires_children() -> None:
    with pytest.raises(ValueError, match="composite signal"):
        build_signal(SignalSpec(kind="composite", params={}))
