"""Central configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # Polymarket (for data scanning + live trading outside US)
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    polymarket_signature_type: int = 1   # 1=Privy/email embedded wallet, 0=MetaMask EOA

    # Polymarket US (CFTC-regulated, US-legal API — api.polymarket.us)
    # Generate at polymarket.us/developer after KYC. Ed25519 auth.
    polymarket_key_id: str = ""          # Key ID (UUID)
    polymarket_secret_key: str = ""      # Base64-encoded Ed25519 private key

    # Kalshi (US-legal, CFTC-regulated — for live trading from US)
    # Auth uses RSA-PSS: generate key pair at kalshi.com → Settings → API
    kalshi_api_key: str = ""              # Key ID (UUID from Kalshi dashboard)
    kalshi_private_key_path: str = "./kalshi_private.pem"  # path to PEM file
    kalshi_demo: bool = False             # True = use demo/paper environment

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "prediction-market-bot/1.0"

    twitter_bearer_token: str = ""

    # ── Exchange routing ──────────────────────────────────────────────────────
    # "polymarket" = use Polymarket CLOB for live trades (requires non-US IP)
    # "kalshi"     = use Kalshi for live trades (US-legal, recommended for US users)
    # "both"       = scan Polymarket + Kalshi, route trades to best available
    live_exchange: str = "kalshi"

    # ── Trading Parameters ────────────────────────────────────────────────────
    bankroll_usdc: float = Field(default=1000.0, ge=1.0)
    kelly_fraction: float = Field(default=0.35, ge=0.01, le=1.0)   # bumped 0.25→0.35 for slightly more conviction
    min_edge: float = Field(default=0.04, ge=0.0, le=1.0)    # QUALITY: require a genuine edge
    max_open_positions: int = Field(default=8, ge=1, le=100) # SAFETY STOP: cap total open positions across cycles
    fee_buffer: float = Field(default=0.02, ge=0.0, le=0.5)  # QUALITY: cushion for exchange fees

    # ── Position management (SELL / exit rules on open positions) ──────────────
    stop_loss_pct: float = Field(default=0.40, ge=0.0, le=1.0)    # exit if position value falls 40% from entry
    take_profit_pct: float = Field(default=0.60, ge=0.0, le=5.0)  # lock in if position value rises 60% from entry
    min_confidence: float = Field(default=0.50, ge=0.30, le=1.0)  # QUALITY: require real conviction
    max_bet_fraction: float = Field(default=0.15, ge=0.001, le=0.5)   # bumped 0.10→0.15 cap per bet
    min_liquidity_usdc: float = Field(default=1000.0, ge=0.0)
    min_volume_usdc: float = Field(default=500.0, ge=0.0)
    max_time_to_resolution_days: int = Field(default=120, ge=1)  # US futures run months out
    min_time_to_resolution_days: int = Field(default=7, ge=0)    # no near-term/next-day resolutions

    # ── Infrastructure ────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./tradetestagent.db"
    model_path: str = "./ml/xgb_calibrator.joblib"

    # ── LLM Model ─────────────────────────────────────────────────────────────
    llm_model: str = "claude-sonnet-4-6"

    # ── Polymarket endpoints ──────────────────────────────────────────────────
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"

    # ── Scan settings ─────────────────────────────────────────────────────────
    scan_limit: int = 300
    scan_interval_seconds: int = 300
    weird_price_move_threshold: float = 0.05  # 5% price move in 24h
    weird_spread_threshold: float = 0.10      # 10% spread

    # ── A/B testing ───────────────────────────────────────────────────────────
    ab_testing_enabled: bool = True

    # ── Email notifications ───────────────────────────────────────────────────
    # Gmail: myaccount.google.com → Security → App Passwords → generate one
    notify_email: str = ""              # destination address (your inbox)
    notify_from_email: str = ""         # sending address (gmail account)
    notify_smtp_password: str = ""      # Gmail App Password (16-char)
    notify_smtp_host: str = "smtp.gmail.com"
    notify_smtp_port: int = 587

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
