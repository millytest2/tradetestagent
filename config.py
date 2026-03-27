"""Central configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "prediction-market-bot/1.0"

    twitter_bearer_token: str = ""

    # ── Trading Parameters ────────────────────────────────────────────────────
    bankroll_usdc: float = Field(default=1000.0, ge=1.0)
    kelly_fraction: float = Field(default=0.25, ge=0.01, le=1.0)
    min_edge: float = Field(default=0.05, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.65, ge=0.5, le=1.0)
    max_bet_fraction: float = Field(default=0.10, ge=0.001, le=0.5)
    min_liquidity_usdc: float = Field(default=1000.0, ge=0.0)
    min_volume_usdc: float = Field(default=500.0, ge=0.0)
    max_time_to_resolution_days: int = Field(default=30, ge=1)
    min_time_to_resolution_days: int = Field(default=1, ge=0)

    # ── Infrastructure ────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./tradetestagent.db"
    model_path: str = "./ml/xgb_calibrator.joblib"

    # ── LLM Model ─────────────────────────────────────────────────────────────
    llm_model: str = "claude-opus-4-6"

    # ── Polymarket endpoints ──────────────────────────────────────────────────
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"

    # ── Scan settings ─────────────────────────────────────────────────────────
    scan_limit: int = 300
    scan_interval_seconds: int = 300
    weird_price_move_threshold: float = 0.05  # 5% price move in 24h
    weird_spread_threshold: float = 0.10      # 10% spread

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
