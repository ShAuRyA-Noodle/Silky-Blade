"""
Backtest artifact read endpoints.

Surfaces the on-disk artifact bundles written by
`quant.backtest.runner.run_backtest` over HTTP so the web UI can render real
results. Layout the runner produces:

    <backtest_artifact_root>/<run_name>/
        report.json
        equity_curve.csv
        manifest.json
        config.snapshot.json

Routes are read-only — they never mutate the filesystem.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from quant.config import Settings, get_settings
from quant.core.dependencies import get_current_user
from quant.db.models import User

router = APIRouter(prefix="/backtests", tags=["backtests"])


# regex anchors enforce no path components — defense against traversal
# (`..`, `/`, leading `.`, NUL all fail the match)
_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


# ---------------------------------------------------------------
# Schemas — strict, no `Any` in response models
# ---------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricsSummary(_Strict):
    sharpe: float
    max_drawdown: float
    deflated_sharpe_p: float


class RunSummary(_Strict):
    name: str
    created_at: datetime
    metrics_summary: MetricsSummary


class RunListOut(_Strict):
    runs: list[RunSummary]


class ReportWindow(_Strict):
    start: date
    end: date
    n_rebalances: int


class ReportMetrics(_Strict):
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float
    deflated_sharpe_p: float
    dsr_n_trials: int
    dsr_sharpes_std: float
    return_skew: float
    return_kurtosis: float


class ReportWalkForward(_Strict):
    train_days: int
    test_days: int
    top_k: int
    cost_bps: float
    initial_capital: float


class ReportSignal(_Strict):
    kind: str
    params: dict[str, float | int | str | bool]


class ReportArtifacts(_Strict):
    dir: str
    report: str
    equity_curve: str
    manifest: str
    config_snapshot: str


class ManifestSummary(_Strict):
    code_sha: str
    config_hash: str
    data_fingerprint: str


class ReportOut(_Strict):
    name: str
    window: ReportWindow
    metrics: ReportMetrics
    walk_forward: ReportWalkForward
    signal: ReportSignal
    artifacts: ReportArtifacts | None = None
    manifest_summary: ManifestSummary


class EquityPoint(_Strict):
    date: date
    equity: float


class EquityCurveOut(_Strict):
    points: list[EquityPoint]


class ManifestOut(_Strict):
    code_sha: str
    config_hash: str
    created_at: datetime
    data_fingerprint: str
    package_versions: dict[str, str]
    python_version: str


class ConfigStats(_Strict):
    n_trials: int
    sharpes_std: float


class ConfigSnapshotOut(_Strict):
    name: str
    prices_csv: str
    start_date: date
    end_date: date
    output_dir: str
    walk_forward: ReportWalkForward
    signal: ReportSignal
    stats: ConfigStats


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]


def _validated_run_dir(name: str, settings: Settings) -> Path:
    if not _NAME_RE.match(name):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid run name")
    root = settings.backtest_artifact_root.resolve()
    run_dir = (root / name).resolve()
    # belt-and-braces: refuse anything that escapes the configured root,
    # even after symlink resolution
    if root != run_dir and root not in run_dir.parents:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid run name")
    if not run_dir.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {name!r} not found")
    return run_dir


def _read_json_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{path.name} not found")
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"{path.name} is malformed")
    return loaded


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@router.get(
    "",
    response_model=RunListOut,
    summary="List backtest runs",
    description=(
        "Lists every artifact bundle under the configured "
        "`backtest_artifact_root`, newest first by `report.json` mtime."
    ),
)
async def list_backtests(
    settings: SettingsDep,
    _user: CurrentUserDep,
) -> RunListOut:
    root = settings.backtest_artifact_root
    if not root.is_dir():
        return RunListOut(runs=[])

    rows: list[tuple[float, RunSummary]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not _NAME_RE.match(child.name):
            continue
        report_path = child / "report.json"
        if not report_path.is_file():
            continue
        try:
            with report_path.open("r", encoding="utf-8") as fh:
                report = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            continue
        try:
            summary = MetricsSummary(
                sharpe=float(metrics["sharpe"]),
                max_drawdown=float(metrics["max_drawdown"]),
                deflated_sharpe_p=float(metrics["deflated_sharpe_p"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        mtime = report_path.stat().st_mtime
        rows.append(
            (
                mtime,
                RunSummary(
                    name=str(report.get("name", child.name)),
                    created_at=datetime.fromtimestamp(mtime).astimezone(),
                    metrics_summary=summary,
                ),
            )
        )

    rows.sort(key=lambda kv: kv[0], reverse=True)
    return RunListOut(runs=[r for _, r in rows])


@router.get(
    "/{name}",
    response_model=ReportOut,
    summary="Get a backtest report",
    description="Returns the parsed `report.json` plus a derived `manifest_summary`.",
)
async def get_backtest(
    name: str,
    settings: SettingsDep,
    _user: CurrentUserDep,
) -> ReportOut:
    run_dir = _validated_run_dir(name, settings)
    report = _read_json_file(run_dir / "report.json")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "manifest.json not found")
    manifest = _read_json_file(manifest_path)
    payload = dict(report)
    payload["manifest_summary"] = {
        "code_sha": str(manifest.get("code_sha", "")),
        "config_hash": str(manifest.get("config_hash", "")),
        "data_fingerprint": str(manifest.get("data_fingerprint", "")),
    }
    return ReportOut.model_validate(payload)


@router.get(
    "/{name}/equity",
    response_model=EquityCurveOut,
    summary="Get a backtest equity curve",
    description="Streams the on-disk `equity_curve.csv` row-by-row into JSON.",
)
async def get_backtest_equity(
    name: str,
    settings: SettingsDep,
    _user: CurrentUserDep,
) -> EquityCurveOut:
    run_dir = _validated_run_dir(name, settings)
    csv_path = run_dir / "equity_curve.csv"
    if not csv_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "equity_curve.csv not found")

    points: list[EquityPoint] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "date" not in reader.fieldnames or "equity" not in reader.fieldnames:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "equity_curve.csv is malformed (missing date/equity columns)",
            )
        for row in reader:
            try:
                points.append(
                    EquityPoint(
                        date=date.fromisoformat(row["date"]),
                        equity=float(row["equity"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "equity_curve.csv has malformed rows",
                ) from e
    return EquityCurveOut(points=points)


@router.get(
    "/{name}/manifest",
    response_model=ManifestOut,
    summary="Get a backtest reproducibility manifest",
    description="Returns the parsed `manifest.json` (code_sha, config_hash, data_fingerprint, env).",
)
async def get_backtest_manifest(
    name: str,
    settings: SettingsDep,
    _user: CurrentUserDep,
) -> ManifestOut:
    run_dir = _validated_run_dir(name, settings)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "manifest.json not found")
    manifest = _read_json_file(manifest_path)
    return ManifestOut.model_validate(manifest)


@router.get(
    "/{name}/config",
    response_model=ConfigSnapshotOut,
    summary="Get the snapshotted backtest config",
    description="Returns the parsed `config.snapshot.json` exactly as written by the runner.",
)
async def get_backtest_config(
    name: str,
    settings: SettingsDep,
    _user: CurrentUserDep,
) -> ConfigSnapshotOut:
    run_dir = _validated_run_dir(name, settings)
    config_path = run_dir / "config.snapshot.json"
    if not config_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "config.snapshot.json not found")
    snapshot = _read_json_file(config_path)
    return ConfigSnapshotOut.model_validate(snapshot)


__all__ = ["router"]
