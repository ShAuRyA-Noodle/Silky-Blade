"""
Integration tests for the backtest artifact read endpoints.

The artifact bundle under test is built by calling the real `run_backtest()`
on synthetic-but-real GBM prices — there are no hand-written JSON files and
no mocks of the runner. The CSV writer mirrors `_write_prices_csv` from
`tests/unit/test_backtest_runner.py`.

Auth is bypassed via FastAPI's `dependency_overrides` because the artifact
endpoints don't actually consume any user data — the override returns a
plausible inactive `User` shape so the surrounding type contract still holds.
"""

from __future__ import annotations

import csv
import json
import math
import uuid
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from quant.api.v1.backtest import EquityCurveOut, RunListOut
from quant.backtest.engine import WalkForwardConfig
from quant.backtest.runner import RunConfig, SignalSpec, StatsSpec, run_backtest
from quant.config import get_settings, settings
from quant.core.dependencies import get_current_user
from quant.db.models import User, UserRole, UserTier
from quant.main import app

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------
# Synthetic prices — same shape as test_backtest_runner._write_prices_csv
# ------------------------------------------------------------------
def _write_prices_csv(path: Path, n_days: int = 600, n_symbols: int = 8, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days) if (start + timedelta(days=i)).weekday() < 5]
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    drifts = np.linspace(-0.0002, 0.0008, n_symbols)
    vol = 0.012

    rows: list[tuple[str, str, float]] = []
    for s_idx, sym in enumerate(symbols):
        price = 100.0
        for d in dates:
            rets = float(rng.normal(drifts[s_idx], vol))
            price *= math.exp(rets)
            rows.append((d.isoformat(), sym, round(price, 4)))

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "symbol", "adj_close"])
        w.writerows(rows)
    return path


def _stub_user() -> User:
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        hashed_password="!",
        full_name="Test User",
        role=UserRole.viewer,
        tier=UserTier.free,
        is_active=True,
        is_verified=True,
    )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture()
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture()
def real_bundle(tmp_path: Path, artifact_root: Path) -> dict[str, Path]:
    """Build one real artifact bundle by calling `run_backtest()` end-to-end."""
    prices = _write_prices_csv(tmp_path / "prices.csv")
    cfg = RunConfig(
        name="route_e2e",
        prices_csv=str(prices),
        start_date=date(2020, 1, 1),
        end_date=date(2022, 6, 30),
        output_dir=str(artifact_root),
        walk_forward=WalkForwardConfig(train_days=60, test_days=5, top_k=3, cost_bps=2.0),
        signal=SignalSpec(kind="momentum", params={"lookback_days": 40}),
        stats=StatsSpec(n_trials=3, sharpes_std=0.3),
    )
    report = run_backtest(cfg)
    out_dir = Path(report["artifacts"]["dir"])
    return {
        "dir": out_dir,
        "report": out_dir / "report.json",
        "equity": out_dir / "equity_curve.csv",
        "manifest": out_dir / "manifest.json",
        "config": out_dir / "config.snapshot.json",
    }


@pytest.fixture()
def client(artifact_root: Path) -> Iterator[TestClient]:
    def _override_settings() -> object:
        settings.backtest_artifact_root = artifact_root
        return settings

    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_settings] = _override_settings
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_settings, None)


# ------------------------------------------------------------------
# List endpoint
# ------------------------------------------------------------------
def test_list_empty_when_root_missing(client: TestClient, artifact_root: Path) -> None:
    assert not artifact_root.exists()
    r = client.get("/api/v1/backtests")
    assert r.status_code == 200, r.text
    payload = RunListOut.model_validate(r.json())
    assert payload.runs == []


def test_list_empty_when_root_has_no_runs(client: TestClient, artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True)
    r = client.get("/api/v1/backtests")
    assert r.status_code == 200, r.text
    payload = RunListOut.model_validate(r.json())
    assert payload.runs == []


def test_list_returns_real_run(client: TestClient, real_bundle: dict[str, Path]) -> None:
    r = client.get("/api/v1/backtests")
    assert r.status_code == 200, r.text
    payload = RunListOut.model_validate(r.json())
    assert len(payload.runs) == 1
    run = payload.runs[0]
    assert run.name == "route_e2e"

    on_disk = json.loads(real_bundle["report"].read_text())
    assert run.metrics_summary.sharpe == pytest.approx(float(on_disk["metrics"]["sharpe"]))
    assert run.metrics_summary.max_drawdown == pytest.approx(float(on_disk["metrics"]["max_drawdown"]))
    assert run.metrics_summary.deflated_sharpe_p == pytest.approx(
        float(on_disk["metrics"]["deflated_sharpe_p"]),
        nan_ok=True,
    )


def test_list_sort_order_newest_first(
    client: TestClient,
    tmp_path: Path,
    artifact_root: Path,
    real_bundle: dict[str, Path],
) -> None:
    prices = _write_prices_csv(tmp_path / "prices2.csv", seed=11)
    cfg = RunConfig(
        name="route_e2e_second",
        prices_csv=str(prices),
        start_date=date(2020, 1, 1),
        end_date=date(2022, 6, 30),
        output_dir=str(artifact_root),
        walk_forward=WalkForwardConfig(train_days=60, test_days=5, top_k=3, cost_bps=2.0),
        signal=SignalSpec(kind="momentum", params={"lookback_days": 40}),
        stats=StatsSpec(n_trials=3, sharpes_std=0.3),
    )
    run_backtest(cfg)

    r = client.get("/api/v1/backtests")
    assert r.status_code == 200, r.text
    payload = RunListOut.model_validate(r.json())
    assert [run.name for run in payload.runs] == ["route_e2e_second", "route_e2e"]
    assert payload.runs[0].created_at >= payload.runs[1].created_at


# ------------------------------------------------------------------
# Get one report
# ------------------------------------------------------------------
def test_get_report_matches_disk(client: TestClient, real_bundle: dict[str, Path]) -> None:
    r = client.get("/api/v1/backtests/route_e2e")
    assert r.status_code == 200, r.text
    body = r.json()

    on_disk = json.loads(real_bundle["report"].read_text())
    assert body["name"] == on_disk["name"]
    assert body["window"]["start"] == on_disk["window"]["start"]
    assert body["window"]["end"] == on_disk["window"]["end"]
    assert body["metrics"]["sharpe"] == pytest.approx(float(on_disk["metrics"]["sharpe"]))
    assert body["metrics"]["max_drawdown"] == pytest.approx(float(on_disk["metrics"]["max_drawdown"]))

    manifest = json.loads(real_bundle["manifest"].read_text())
    assert body["manifest_summary"]["code_sha"] == manifest["code_sha"]
    assert body["manifest_summary"]["config_hash"] == manifest["config_hash"]
    assert body["manifest_summary"]["data_fingerprint"] == manifest["data_fingerprint"]


def test_get_report_404_when_missing(client: TestClient, artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True)
    r = client.get("/api/v1/backtests/does_not_exist")
    assert r.status_code == 404


# ------------------------------------------------------------------
# Equity curve
# ------------------------------------------------------------------
def test_get_equity_matches_csv_row_count(client: TestClient, real_bundle: dict[str, Path]) -> None:
    with real_bundle["equity"].open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        on_disk = list(reader)

    r = client.get("/api/v1/backtests/route_e2e/equity")
    assert r.status_code == 200, r.text
    payload = EquityCurveOut.model_validate(r.json())
    assert len(payload.points) == len(on_disk)
    assert payload.points[0].date.isoformat() == on_disk[0]["date"]
    assert payload.points[0].equity == pytest.approx(float(on_disk[0]["equity"]))
    assert payload.points[-1].date.isoformat() == on_disk[-1]["date"]
    assert payload.points[-1].equity == pytest.approx(float(on_disk[-1]["equity"]))


def test_get_equity_404_when_missing(client: TestClient, artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True)
    r = client.get("/api/v1/backtests/does_not_exist/equity")
    assert r.status_code == 404


# ------------------------------------------------------------------
# Manifest + config
# ------------------------------------------------------------------
def test_get_manifest_matches_disk(client: TestClient, real_bundle: dict[str, Path]) -> None:
    r = client.get("/api/v1/backtests/route_e2e/manifest")
    assert r.status_code == 200, r.text
    on_disk = json.loads(real_bundle["manifest"].read_text())
    body = r.json()
    assert body["code_sha"] == on_disk["code_sha"]
    assert body["config_hash"] == on_disk["config_hash"]
    assert body["data_fingerprint"] == on_disk["data_fingerprint"]
    assert body["python_version"] == on_disk["python_version"]
    assert body["package_versions"] == on_disk["package_versions"]


def test_get_config_matches_disk(client: TestClient, real_bundle: dict[str, Path]) -> None:
    r = client.get("/api/v1/backtests/route_e2e/config")
    assert r.status_code == 200, r.text
    on_disk = json.loads(real_bundle["config"].read_text())
    body = r.json()
    assert body["name"] == on_disk["name"]
    assert body["walk_forward"] == on_disk["walk_forward"]
    assert body["signal"]["kind"] == on_disk["signal"]["kind"]
    assert body["stats"]["n_trials"] == on_disk["stats"]["n_trials"]


# ------------------------------------------------------------------
# Path traversal — every variant must 4xx, never 200
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_name",
    [
        "..",
        "../etc",
        "../../etc/passwd",
        "%2e%2e%2fetc%2fpasswd",
        "foo/bar",
        ".hidden",
        "name with space",
        "abs",
        "name;rm",
    ],
)
def test_path_traversal_rejected(
    client: TestClient,
    real_bundle: dict[str, Path],
    bad_name: str,
) -> None:
    for suffix in ("", "/equity", "/manifest", "/config"):
        r = client.get(f"/api/v1/backtests/{bad_name}{suffix}")
        assert r.status_code in {400, 404}, (bad_name, suffix, r.status_code, r.text)
        assert r.status_code != 200
