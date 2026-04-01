"""Central configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "stocktradingbot/1.0"

    twitter_bearer_token: str = ""

    # ── Trading Parameters ────────────────────────────────────────────────────
    bankroll_usd: float = Field(default=1000.0, ge=1.0)
    kelly_fraction: float = Field(default=0.25, ge=0.01, le=1.0)
    min_edge: float = Field(default=0.05, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.60, ge=0.5, le=1.0)
    max_bet_fraction: float = Field(default=0.10, ge=0.001, le=0.5)

    # ── Stock Filters ─────────────────────────────────────────────────────────
    min_volume: int = Field(default=1_000_000, ge=0)
    min_market_cap: float = Field(default=1_000_000_000.0, ge=0.0)
    min_price: float = Field(default=5.0, ge=0.0)
    earnings_buffer_days: int = Field(default=3, ge=0)

    # ── Position Management ───────────────────────────────────────────────────
    holding_days: int = Field(default=5, ge=1, le=30)
    stop_loss_pct: float = Field(default=0.03, ge=0.005, le=0.20)
    take_profit_pct: float = Field(default=0.05, ge=0.01, le=0.50)
    max_open_positions: int = Field(default=5, ge=1, le=20)

    # ── Infrastructure ────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./stocktradingbot.db"
    model_path: str = "./ml/xgb_calibrator.joblib"

    # ── LLM Model ─────────────────────────────────────────────────────────────
    llm_model: str = "claude-sonnet-4-6"

    # ── Scan settings ─────────────────────────────────────────────────────────
    scan_interval_seconds: int = 300
    volume_ratio_threshold: float = 2.0    # flag if volume > 2x average
    rsi_oversold: float = 30.0             # flag if RSI < 30
    rsi_overbought: float = 70.0           # flag if RSI > 70
    price_change_threshold: float = 0.03   # flag if 1d price change > 3%

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
