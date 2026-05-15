"""
Signal (model output) read endpoints.

The frontend hits these for the daily signal dashboard — what the current
active models think about the universe.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from quant.core.dependencies import get_current_user, get_db
from quant.db.models import ModelRun, Signal, SignalDirection, User

router = APIRouter(tags=["signals"])


class SignalOut(BaseModel):
    date: date
    symbol: str
    model_run_id: uuid.UUID
    direction: SignalDirection
    confidence: Decimal
    score: Decimal
    rank_in_universe: int | None
    entry_price: Decimal | None
    target_price: Decimal | None
    stop_price: Decimal | None
    horizon_days: int
    shap_values: dict[str, Any] | None
    risk_level: str | None
    explanation: str | None
    computed_at: datetime


class ModelRunOut(BaseModel):
    id: uuid.UUID
    name: str
    family: str
    version: int
    feature_set_version: str
    train_start: date
    train_end: date
    cv_scheme: str
    metrics: dict[str, Any]
    is_active: bool


@router.get("/signals", response_model=list[SignalOut])
async def list_signals(
    db: AsyncSession = Depends(get_db),
    on_date: date | None = Query(None, alias="date"),
    direction: SignalDirection | None = Query(None),
    symbol: str | None = Query(None),
    active_models_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
) -> list[SignalOut]:
    # Default to latest available date.
    if on_date is None:
        latest_stmt = select(Signal.date).order_by(Signal.date.desc()).limit(1)
        latest = (await db.execute(latest_stmt)).scalar_one_or_none()
        if latest is None:
            return []
        on_date = latest

    stmt = select(Signal).where(Signal.date == on_date)
    if direction is not None:
        stmt = stmt.where(Signal.direction == direction)
    if symbol:
        stmt = stmt.where(Signal.symbol == symbol.upper())
    if active_models_only:
        stmt = stmt.join(ModelRun, ModelRun.id == Signal.model_run_id).where(ModelRun.is_active.is_(True))
    stmt = stmt.order_by(Signal.rank_in_universe.asc().nulls_last(), Signal.confidence.desc()).limit(limit)

    rows = (await db.execute(stmt)).scalars().all()
    return [
        SignalOut(
            date=r.date,
            symbol=r.symbol,
            model_run_id=r.model_run_id,
            direction=r.direction,
            confidence=r.confidence,
            score=r.score,
            rank_in_universe=r.rank_in_universe,
            entry_price=r.entry_price,
            target_price=r.target_price,
            stop_price=r.stop_price,
            horizon_days=r.horizon_days,
            shap_values=r.shap_values,
            risk_level=r.risk_level,
            explanation=r.explanation,
            computed_at=r.computed_at,
        )
        for r in rows
    ]


@router.get("/signals/{symbol}/history", response_model=list[SignalOut])
async def signal_history(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
    limit: int = Query(90, ge=1, le=500),
) -> list[SignalOut]:
    sym = symbol.upper()
    stmt = (
        select(Signal)
        .join(ModelRun, ModelRun.id == Signal.model_run_id)
        .where(Signal.symbol == sym, ModelRun.is_active.is_(True))
        .order_by(Signal.date.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        SignalOut(
            date=r.date,
            symbol=r.symbol,
            model_run_id=r.model_run_id,
            direction=r.direction,
            confidence=r.confidence,
            score=r.score,
            rank_in_universe=r.rank_in_universe,
            entry_price=r.entry_price,
            target_price=r.target_price,
            stop_price=r.stop_price,
            horizon_days=r.horizon_days,
            shap_values=r.shap_values,
            risk_level=r.risk_level,
            explanation=r.explanation,
            computed_at=r.computed_at,
        )
        for r in rows
    ]


@router.get("/models", response_model=list[ModelRunOut])
async def list_models(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
    active_only: bool = Query(False),
) -> list[ModelRunOut]:
    stmt = select(ModelRun).order_by(ModelRun.created_at.desc())
    if active_only:
        stmt = stmt.where(ModelRun.is_active.is_(True))
    rows = (await db.execute(stmt)).scalars().all()
    return [
        ModelRunOut(
            id=r.id,
            name=r.name,
            family=r.family,
            version=r.version,
            feature_set_version=r.feature_set_version,
            train_start=r.train_start,
            train_end=r.train_end,
            cv_scheme=r.cv_scheme,
            metrics=r.metrics or {},
            is_active=r.is_active,
        )
        for r in rows
    ]


@router.get("/models/{model_id}", response_model=ModelRunOut)
async def get_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ModelRunOut:
    r = await db.get(ModelRun, model_id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model not found")
    return ModelRunOut(
        id=r.id,
        name=r.name,
        family=r.family,
        version=r.version,
        feature_set_version=r.feature_set_version,
        train_start=r.train_start,
        train_end=r.train_end,
        cv_scheme=r.cv_scheme,
        metrics=r.metrics or {},
        is_active=r.is_active,
    )
