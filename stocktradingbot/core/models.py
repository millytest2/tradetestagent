"""Pydantic data models shared across agents."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class StockSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeOutcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"


class TradeStatus(str, Enum):
    PROPOSED = "PROPOSED"
    BLOCKED = "BLOCKED"
    PLACED = "PLACED"
    SETTLED = "SETTLED"
    FAILED = "FAILED"


# ── Stock Models ──────────────────────────────────────────────────────────────

class Stock(BaseModel):
    """A US equity as returned by the scanner."""
    ticker: str
    company_name: str = ""
    current_price: float
    volume_today: float = 0.0
    avg_volume_20d: float = 0.0
    volume_ratio: float = 1.0           # today / 20d avg
    rsi_14: float = 50.0                # RSI(14)
    macd_signal: float = 0.0            # MACD histogram (positive=bullish)
    bb_position: float = 0.5            # Bollinger Band position (0=lower, 1=upper)
    price_change_1d: float = 0.0        # 1-day % change
    price_change_5d: float = 0.0        # 5-day % change
    price_change_20d: float = 0.0       # 20-day % change
    distance_from_52w_high: float = 0.0 # (price / 52w_high) - 1  (negative = below high)
    distance_from_52w_low: float = 0.0  # (price / 52w_low) - 1   (positive = above low)
    short_interest_ratio: float = 0.0   # short interest / float shares
    days_to_earnings: Optional[int] = None  # None if unknown
    market_cap: float = 0.0
    sector: str = ""
    is_flagged: bool = False
    flag_reason: str = ""
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class FlaggedStock(BaseModel):
    """Stock that passed the scan filter."""
    stock: Stock
    flag_reason: str
    priority_score: float = 0.0   # higher = more interesting


# ── Research Models ───────────────────────────────────────────────────────────

class SocialPost(BaseModel):
    source: str          # "twitter" | "reddit" | "rss"
    author: str = ""
    text: str
    url: str = ""
    published_at: Optional[datetime] = None
    likes: int = 0
    retweets: int = 0
    score: int = 0       # reddit score


class SentimentResult(BaseModel):
    compound: float      # VADER compound (-1 to +1)
    positive: float
    negative: float
    neutral: float
    post_count: int
    avg_engagement: float


class ResearchReport(BaseModel):
    ticker: str
    company_name: str
    posts: list[SocialPost] = Field(default_factory=list)
    sentiment: SentimentResult
    narrative_summary: str = ""
    key_claims: list[str] = Field(default_factory=list)
    researched_at: datetime = Field(default_factory=datetime.utcnow)
    trend_score: float = 50.0           # Google Trends interest for ticker/company
    whale_bid_imbalance: float = 0.0    # Alpaca order book bid/ask imbalance


# ── Prediction Models ─────────────────────────────────────────────────────────

class PredictionFeatures(BaseModel):
    """Features fed to both XGBoost and the LLM."""
    # Technical indicators
    rsi_14: float
    macd_signal: float
    bb_position: float
    volume_ratio: float
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    distance_from_52w_high: float
    short_interest_ratio: float
    # Sentiment / social
    compound_sentiment: float
    post_count: int
    avg_engagement: float
    trend_score: float
    whale_bid_imbalance: float


class Prediction(BaseModel):
    ticker: str
    company_name: str
    xgb_up_probability: float           # P(price up in 5 days) from XGBoost
    llm_up_probability: float           # P(price up in 5 days) from LLM
    calibrated_up_probability: float    # weighted ensemble
    edge: float                         # calibrated_prob - 0.5 (base rate)
    confidence: float                   # model confidence (0–1)
    side: StockSide                     # LONG or SHORT
    reasoning: str = ""
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    should_trade: bool = False


# ── Risk / Trade Models ───────────────────────────────────────────────────────

class BetSizing(BaseModel):
    kelly_fraction_full: float
    kelly_fraction_used: float    # fractional Kelly
    bet_usd: float
    max_allowed_usd: float
    bankroll_usd: float
    edge: float
    odds: float                   # risk/reward ratio (take_profit / stop_loss)


class TradeDecision(BaseModel):
    approved: bool
    rejection_reason: str = ""
    prediction: Prediction
    sizing: Optional[BetSizing] = None


class Trade(BaseModel):
    id: Optional[int] = None
    ticker: str
    company_name: str = ""
    side: StockSide
    entry_price: float
    bet_usd: float
    shares: float
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    status: TradeStatus = TradeStatus.PLACED
    outcome: TradeOutcome = TradeOutcome.PENDING
    pnl_usd: float = 0.0
    alpaca_order_id: str = ""
    entry_date: datetime = Field(default_factory=datetime.utcnow)
    exit_date: Optional[datetime] = None
    notes: str = ""


# ── Postmortem Models ─────────────────────────────────────────────────────────

class PostmortemFinding(BaseModel):
    trade_id: int
    agent_name: str
    finding: str
    root_cause: str
    recommendation: str
    severity: str = "medium"    # low | medium | high | critical
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PostmortemReport(BaseModel):
    trade_id: int
    ticker: str
    findings: list[PostmortemFinding] = Field(default_factory=list)
    system_updates: list[str] = Field(default_factory=list)
    lessons_learned: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
