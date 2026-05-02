"""v1 API router aggregator."""

from fastapi import APIRouter

from quant.api.v1.admin import router as admin_router
from quant.api.v1.auth import router as auth_router
from quant.api.v1.backtest import router as backtest_router
from quant.api.v1.market import router as market_router
from quant.api.v1.orders import router as orders_router
from quant.api.v1.signals import router as signals_router
from quant.api.v1.stream import router as stream_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth_router)
api_router.include_router(orders_router)
api_router.include_router(stream_router)
api_router.include_router(market_router)
api_router.include_router(signals_router)
api_router.include_router(admin_router)
api_router.include_router(backtest_router)

__all__ = ["api_router"]
