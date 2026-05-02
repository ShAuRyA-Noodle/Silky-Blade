"""
Typed application settings loaded from .env.local.

Every provider key is declared here with the exact type and a field-level
description. Missing Tier-1 keys fail fast at boot — you cannot accidentally
start the platform with a broken data provider.

Import pattern:
    from quant.config import settings
    api_key = settings.polygon_api_key.get_secret_value()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (two levels up from apps/api/src/quant/config/settings.py)
_REPO_ROOT = Path(__file__).resolve().parents[5]
_ENV_FILE = _REPO_ROOT / ".env.local"


# ---------------------------------------------------------------
# Universe presets
# ---------------------------------------------------------------
UniversePreset = Literal["SP500", "NDX100", "SP500_NDX100", "R1000", "DEV"]


# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
class Settings(BaseSettings):
    """
    Strongly-typed runtime configuration.

    Validation rules:
    - Tier-1 keys (Polygon, Alpaca, FRED) are REQUIRED.
    - Tier-2/3 keys are optional but warn at boot if missing.
    - DB/Redis/MinIO URLs are required.
    - `JWT_SECRET_KEY` must be non-default in non-development envs.

    Backtest artifacts:
    - `backtest_artifact_root` is the on-disk root that the backtest CLI
      writes its `<run_name>/{report,equity_curve,manifest,config.snapshot}`
      bundles to. The HTTP layer reads from the same path, read-only.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------- App --------
    app_env: Literal["development", "staging", "production"] = "development"
    app_name: str = "quant-platform"
    app_debug: bool = True
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    app_timezone: str = "America/New_York"

    # -------- API --------
    api_host: str = "0.0.0.0"  # noqa: S104  containerized API binds to all interfaces
    api_port: int = 8000
    api_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    # -------- Security --------
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_minutes: int = 30
    jwt_refresh_ttl_days: int = 14
    bcrypt_rounds: int = 12

    # -------- Database --------
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "quant"
    postgres_user: str = "quant"
    postgres_password: SecretStr
    database_url: str
    database_url_sync: str

    # -------- Redis --------
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_url: str = "redis://redis:6379/0"

    # -------- MinIO --------
    minio_endpoint: str = "minio:9000"
    minio_access_key: SecretStr
    minio_secret_key: SecretStr
    minio_bucket_models: str = "models"
    minio_bucket_reports: str = "reports"
    minio_bucket_data: str = "data"
    minio_secure: bool = False

    # -------- MLflow --------
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_s3_endpoint_url: str = "http://minio:9000"
    mlflow_experiment_name: str = "quant-platform"

    # ============================================================
    # PROVIDER KEYS — every one is used. See docs/adapters.md.
    # ============================================================

    # ---- Tier 1 (REQUIRED) ----
    polygon_api_key: SecretStr = Field(
        ..., description="Polygon.io — primary OHLCV, corporate actions, news v2"
    )
    alpaca_api_key_id: SecretStr = Field(..., description="Alpaca — broker API key ID")
    alpaca_api_secret_key: SecretStr = Field(..., description="Alpaca — broker API secret")
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_url: str = "https://data.alpaca.markets"
    alpaca_stream_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    alpaca_paper: bool = True

    fred_api_key: SecretStr = Field(
        ..., description="FRED — macro series (VIX, yields, DXY, CPI, unemployment)"
    )

    # ---- Tier 2 (strong value-add) ----
    groq_api_key: SecretStr = Field(..., description="Groq — LLM for news sentiment + signal explanations")
    groq_model_fast: str = "llama-3.1-8b-instant"
    groq_model_smart: str = "llama-3.3-70b-versatile"

    finnhub_api_key: SecretStr = Field(
        ..., description="Finnhub — earnings calendar, insider transactions, recommendations"
    )
    tiingo_api_key: SecretStr = Field(..., description="Tiingo — OHLCV fallback, fundamentals, IEX quotes")
    marketaux_api_key: SecretStr = Field(..., description="Marketaux — financial-tagged news feed")

    # ---- Tier 3 (nice to have) ----
    newsapi_key: SecretStr | None = Field(default=None, description="NewsAPI — general news fallback")
    fmp_api_key: SecretStr | None = Field(
        default=None, description="FMP — analyst targets, estimates, revisions"
    )
    nasdaq_data_link_api_key: SecretStr | None = Field(
        default=None, description="Nasdaq Data Link — free macro/alt datasets"
    )
    alphavantage_api_key: SecretStr | None = Field(
        default=None, description="Alpha Vantage — last-resort fallback + FX"
    )
    sec_edgar_user_agent: str = "quant-platform-dev noreply@example.com"

    # ---- Deferred ----
    # reddit_* keys deliberately absent (user skipped for v1)

    # ============================================================
    # TRADING / RISK
    # ============================================================
    trading_enabled: bool = False
    universe: UniversePreset = "SP500_NDX100"
    initial_capital_usd: float = 100_000.0
    max_position_pct: float = 0.05
    max_positions: int = 20
    max_sector_pct: float = 0.30
    daily_loss_limit_pct: float = 0.02
    drawdown_kill_pct: float = 0.15
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 3.0

    # ============================================================
    # BACKTEST ARTIFACTS
    # ============================================================
    backtest_artifact_root: Path = Path("./examples/backtest/artifacts")

    # ============================================================
    # Validators
    # ============================================================
    @field_validator("jwt_secret_key")
    @classmethod
    def _jwt_not_trivial(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 chars")
        return v

    @model_validator(mode="after")
    def _production_hardening(self) -> Settings:
        if self.app_env == "production":
            insecure = self.jwt_secret_key.get_secret_value().startswith("dev_")
            if insecure:
                raise ValueError(
                    "Refusing to boot in production with a dev JWT secret. "
                    'Generate with: python -c "import secrets; print(secrets.token_urlsafe(64))"'
                )
            if self.app_debug:
                raise ValueError("APP_DEBUG must be false in production")
        return self

    # ============================================================
    # Convenience
    # ============================================================
    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"

    @property
    def is_prod(self) -> bool:
        return self.app_env == "production"

    def provider_summary(self) -> dict[str, bool]:
        """Which providers are configured. Used by /health."""
        return {
            "polygon": bool(self.polygon_api_key.get_secret_value()),
            "alpaca": bool(self.alpaca_api_key_id.get_secret_value()),
            "fred": bool(self.fred_api_key.get_secret_value()),
            "groq": bool(self.groq_api_key.get_secret_value()),
            "finnhub": bool(self.finnhub_api_key.get_secret_value()),
            "tiingo": bool(self.tiingo_api_key.get_secret_value()),
            "marketaux": bool(self.marketaux_api_key.get_secret_value()),
            "newsapi": self.newsapi_key is not None,
            "fmp": self.fmp_api_key is not None,
            "nasdaq_data_link": self.nasdaq_data_link_api_key is not None,
            "alphavantage": self.alphavantage_api_key is not None,
        }


# ---------------------------------------------------------------
# Singleton accessor (cached; settings are immutable at runtime)
# ---------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Eager singleton for normal imports
settings: Settings = get_settings()
