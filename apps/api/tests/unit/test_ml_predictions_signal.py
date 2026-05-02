"""Tests for MLPredictionsSignal — replays trainer OOF predictions as signal."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from quant.backtest.runner import SignalSpec, build_signal
from quant.backtest.signals import MLPredictionsSignal


def _write_oof(path: Path, rows: list[dict[str, object]]) -> Path:
    fields = [
        "date",
        "symbol",
        "label",
        "touch_date",
        "prob_neg1",
        "prob_zero",
        "prob_pos1",
        "prob_neg1_calibrated",
        "prob_zero_calibrated",
        "prob_pos1_calibrated",
        "in_oof",
        "pred_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _row(
    d: date,
    sym: str,
    *,
    p_neg: float,
    p_zero: float,
    p_pos: float,
    in_oof: bool = True,
) -> dict[str, object]:
    return {
        "date": d.isoformat(),
        "symbol": sym,
        "label": 0,
        "touch_date": d.isoformat(),
        "prob_neg1": p_neg,
        "prob_zero": p_zero,
        "prob_pos1": p_pos,
        # Use the same probabilities for raw + calibrated in tests; real-world
        # they differ but the ranking math is identical.
        "prob_neg1_calibrated": p_neg,
        "prob_zero_calibrated": p_zero,
        "prob_pos1_calibrated": p_pos,
        "in_oof": str(in_oof).lower(),
        "pred_class": 0,
    }


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
def test_build_signal_ml_predictions(tmp_path: Path) -> None:
    p = _write_oof(tmp_path / "oof.csv", [_row(date(2020, 1, 2), "A", p_neg=0.2, p_zero=0.5, p_pos=0.3)])
    s = build_signal(SignalSpec(kind="ml_predictions", params={"predictions_csv": str(p)}))
    assert isinstance(s, MLPredictionsSignal)
    assert s.predictions_csv == str(p)
    assert s.use_calibrated is True


def test_build_signal_ml_predictions_requires_csv() -> None:
    with pytest.raises(ValueError, match="predictions_csv"):
        build_signal(SignalSpec(kind="ml_predictions", params={}))


# ------------------------------------------------------------------
# Score = P(+1) - P(-1)
# ------------------------------------------------------------------
def test_ml_predictions_signal_ranks_bull_over_bear(tmp_path: Path) -> None:
    rows = [
        _row(date(2020, 1, 2), "BULL", p_neg=0.1, p_zero=0.2, p_pos=0.7),
        _row(date(2020, 1, 2), "BEAR", p_neg=0.7, p_zero=0.2, p_pos=0.1),
        _row(date(2020, 1, 2), "FLAT", p_neg=0.3, p_zero=0.4, p_pos=0.3),
    ]
    p = _write_oof(tmp_path / "oof.csv", rows)
    sig = MLPredictionsSignal(predictions_csv=str(p))
    scores = sig(date(2020, 1, 5), pl_empty_history())
    got = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True)}
    assert got["BULL"] == pytest.approx(0.6)
    assert got["BEAR"] == pytest.approx(-0.6)
    assert got["FLAT"] == pytest.approx(0.0)


def test_ml_predictions_signal_uses_latest_in_oof_per_symbol(tmp_path: Path) -> None:
    rows = [
        # Two predictions for the same symbol on different dates; only the
        # most recent in-OOF one should drive the score.
        _row(date(2020, 1, 2), "X", p_neg=0.5, p_zero=0.4, p_pos=0.1),
        _row(date(2020, 2, 5), "X", p_neg=0.1, p_zero=0.4, p_pos=0.5),
    ]
    p = _write_oof(tmp_path / "oof.csv", rows)
    sig = MLPredictionsSignal(predictions_csv=str(p))
    scores = sig(date(2020, 3, 1), pl_empty_history())
    got = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True)}
    # Latest (2020-02-05): score = 0.5 - 0.1 = 0.4
    assert got["X"] == pytest.approx(0.4)


def test_ml_predictions_signal_excludes_non_oof_rows(tmp_path: Path) -> None:
    rows = [
        _row(date(2020, 1, 2), "A", p_neg=0.1, p_zero=0.2, p_pos=0.7, in_oof=False),
        _row(date(2020, 1, 2), "B", p_neg=0.7, p_zero=0.2, p_pos=0.1, in_oof=True),
    ]
    p = _write_oof(tmp_path / "oof.csv", rows)
    sig = MLPredictionsSignal(predictions_csv=str(p))
    scores = sig(date(2020, 3, 1), pl_empty_history())
    syms = scores["symbol"].to_list()
    assert "A" not in syms
    assert "B" in syms


def test_ml_predictions_signal_returns_empty_before_first_prediction(tmp_path: Path) -> None:
    rows = [_row(date(2024, 1, 2), "A", p_neg=0.1, p_zero=0.2, p_pos=0.7)]
    p = _write_oof(tmp_path / "oof.csv", rows)
    sig = MLPredictionsSignal(predictions_csv=str(p))
    scores = sig(date(2020, 1, 1), pl_empty_history())
    assert scores.height == 0


def test_ml_predictions_signal_falls_back_to_raw_when_calibrated_off(tmp_path: Path) -> None:
    rows = [_row(date(2020, 1, 2), "A", p_neg=0.2, p_zero=0.3, p_pos=0.5)]
    p = _write_oof(tmp_path / "oof.csv", rows)
    sig = MLPredictionsSignal(predictions_csv=str(p), use_calibrated=False)
    scores = sig(date(2020, 3, 1), pl_empty_history())
    assert scores["score"][0] == pytest.approx(0.3)  # 0.5 - 0.2


def pl_empty_history():  # pragma: no cover - tiny helper
    import polars as pl

    return pl.DataFrame({"date": [], "symbol": [], "adj_close": []})
