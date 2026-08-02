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
    # "polymarket_us" = Polymarket US (CFTC-regulated, US-legal) — PRODUCTION
    # "polymarket"    = international Polymarket CLOB (requires non-US IP)
    # "kalshi"        = Kalshi (US-legal)
    # "both"          = scan Polymarket + Kalshi, route to best available
    live_exchange: str = "kalshi"

    # ── Trading Parameters ────────────────────────────────────────────────────
    bankroll_usdc: float = Field(default=100.0, ge=1.0)  # conservative fallback — live cash is fetched each cycle; this only applies if that fails
    kelly_fraction: float = Field(default=0.45, ge=0.01, le=1.0)   # bigger bets that scale with edge/bankroll
    min_edge: float = Field(default=0.02, ge=0.0, le=1.0)    # LEARNING BOOTSTRAP: looser so fast-settling trades place
    max_open_positions: int = Field(default=12, ge=1, le=100) # headroom for ~4 more near-term positions
    fee_buffer: float = Field(default=0.02, ge=0.0, le=0.5)  # cushion for exchange fees
    pause_new_trades: bool = Field(default=False)  # KILL SWITCH: hold all open positions, open nothing new
    favorite_price_floor: float = Field(default=0.70, ge=0.5, le=0.95)  # "obvious win" strong-favorite threshold
    min_entry_price: float = Field(default=0.0, ge=0.0, le=0.95)  # FAVORITES MODE: only bet a side priced >= this (targets high win rate)
    max_positions_per_event: int = Field(default=1, ge=1, le=10)  # never stack bets on the same event (5 Wimbledon picks compete with each other)
    min_bet_usdc: float = Field(default=2.0, ge=1.0)  # meaningful minimum: a signal that passes every gate bets at least this (small bankrolls compute Kelly bets in cents)

    # ── Favorite staking (backed favorites are sized off bankroll, not Kelly) ───
    # SUPERSEDED by confident_stake_pct / ordinary_stake_pct below: stakes are now
    # a continuous PERCENTAGE of the live wallet, so there is no longer a plateau
    # where a $37 and a $99 account bet the same $5. The flat_bet_* values are
    # retained because scan_agent derives its liquidity floor from the largest
    # stake we would ever place, and for fallback/reference.
    flat_bet_base: float = Field(default=2.0, ge=1.0)        # legacy ordinary flat
    flat_bet_confident: float = Field(default=5.0, ge=1.0)   # legacy confident flat
    flat_bet_scaled: float = Field(default=10.0, ge=1.0)     # legacy scaled flat (also sizes the liquidity floor)
    favorite_confident_conf: float = Field(default=0.80, ge=0.5, le=1.0)  # conviction to earn the confident stake %
    scale_up_bankroll: float = Field(default=100.0, ge=1.0)  # goal marker: progress is reported against this
    max_positions_per_event_confident: int = Field(default=2, ge=1, le=10)  # allow a 2nd position on one event only for confident favorites
    max_event_exposure_fraction: float = Field(default=0.25, ge=0.01, le=1.0)  # cap COMBINED stake on one event to this fraction of the wallet, so "2 confident positions" can't over-concentrate a correlated outcome (e.g. two YES bets on the same GDP event)
    strategy_epoch: str = "2026-07-03"  # circuit breaker judges only trades placed on/after this date — trades from before the big bug-fix wave (wrong-side fills, longshots, stacking) don't indict the current strategy
    allow_short_side: bool = Field(default=False)  # PM-US BUY_SHORT execution is unverified — block NO-side live orders until proven

    # ── Position management (SELL / exit rules on open positions) ──────────────
    stop_loss_pct: float = Field(default=0.40, ge=0.0, le=1.0)    # exit if position value falls 40% from entry
    take_profit_pct: float = Field(default=0.60, ge=0.0, le=5.0)  # lock in if position value rises 60% from entry
    min_confidence: float = Field(default=0.45, ge=0.30, le=1.0)  # LEARNING BOOTSTRAP: looser so fast-settling trades place
    max_bet_fraction: float = Field(default=0.25, ge=0.001, le=0.5)   # bigger cap; conviction scaling keeps weak bets small
    min_liquidity_usdc: float = Field(default=1000.0, ge=0.0)  # ceiling for the scaled floor below
    # Liquidity is required PROPORTIONATE to our stake, not as a flat $1k: we bet
    # $2-$10, so demanding $1,000 of book depth rejected ~99% of the venue for no
    # execution benefit. Require stake x multiple, never below the absolute floor.
    liquidity_stake_multiple: float = Field(default=25.0, ge=1.0, le=500.0)
    # Continuous stake sizing (replaces the flat $2/$5/$10 ladder's cliff at $100):
    # every dollar gained/deposited immediately nudges the next bet up, and every
    # dollar lost nudges it down. Bounded by min_bet_usdc and max_bet_fraction.
    confident_stake_pct: float = Field(default=0.08, ge=0.005, le=0.30)  # high-conviction favorite: 8% of bankroll
    ordinary_stake_pct: float = Field(default=0.035, ge=0.005, le=0.20)  # ordinary backed favorite: 3.5%
    max_trades_per_cycle: int = Field(default=8, ge=1, le=50)  # upper bound; actual budget scales with bankroll
    # Dry powder: stop OPENING positions once cash falls to this share of total
    # equity. The old floor was a flat $1, i.e. "deploy everything" — which left
    # no capital for better opportunities and made the account fully dependent on
    # already-open bets. Managing/exiting existing positions is never blocked.
    min_cash_reserve_fraction: float = Field(default=0.25, ge=0.0, le=0.9)
    # ── Degraded (FREE) mode: running without LLM credits ─────────────────────
    # With no AI second opinion the trap-veto loses its main input, so the bot
    # trades on the market price plus a thinly-trained model. Two brakes:
    # refuse favorites the RAW model rates far below the price, and take fewer
    # positions per invocation. Neither applies when the LLM is available.
    free_mode_max_model_gap: float = Field(default=0.45, ge=0.05, le=1.0)
    free_mode_trade_budget_divisor: int = Field(default=2, ge=1, le=10)
    # ── Binary-contract exits (favorites mode) ────────────────────────────────
    # Percent-change bands are the wrong tool for contracts that settle at 0 or
    # 1. A +60% take-profit needs price 1.31 on a 0.82 entry — unreachable, so
    # it never fired; a -40% stop fires at 0.49, which is ordinary noise in a
    # thin market. The exit logic could therefore only ever realize LOSSES.
    # Use PRICE levels instead, matching the actual thesis: the position is a
    # claim on the outcome, so hold it to resolution unless the market has
    # genuinely repriced.
    favorite_take_profit_price: float = Field(default=0.97, ge=0.5, le=1.0)  # near-certain → bank it, free the capital
    favorite_stop_price: float = Field(default=0.25, ge=0.0, le=0.6)         # collapse this far = real news, not noise
    min_liquidity_floor_abs: float = Field(default=150.0, ge=0.0)  # hard floor: exclude dead books
    min_volume_usdc: float = Field(default=500.0, ge=0.0)
    max_time_to_resolution_days: int = Field(default=120, ge=1)  # US futures run months out
    min_time_to_resolution_days: int = Field(default=1, ge=0)    # LEARNING BOOTSTRAP: allow next-few-days markets so they settle fast

    # ── Infrastructure ────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./tradetestagent.db"
    model_path: str = "./ml/xgb_calibrator.joblib"

    # ── LLM Model ─────────────────────────────────────────────────────────────
    llm_model: str = "claude-sonnet-4-6"
    llm_enabled: bool = True   # FREE MODE: false = zero Anthropic API calls; bot
                               # trades timidly (favorites-only, capped confidence)
                               # and defers postmortems until re-enabled

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
