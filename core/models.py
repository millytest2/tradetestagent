"""Pydantic data models shared across agents."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class MarketSide(str, Enum):
    YES = "YES"
    NO = "NO"


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


# ── Market Models ─────────────────────────────────────────────────────────────

class Market(BaseModel):
    """A prediction market as returned by the scanner."""
    condition_id: str
    question: str
    description: str = ""
    end_date_iso: str
    liquidity_usdc: float
    volume_24h_usdc: float
    yes_price: float              # 0.0 – 1.0
    no_price: float               # 0.0 – 1.0
    spread: float                 # no_price - yes_price spread
    price_change_24h: float       # abs % price move in last 24h
    time_to_resolution_days: float
    is_flagged: bool = False      # flagged by scanner for weird activity
    flag_reason: str = ""
    tags: list[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    yes_token_id: str = ""        # CLOB token ID for order book whale detection
    no_token_id: str = ""


class FlaggedMarket(BaseModel):
    """Market that passed the scan filter."""
    market: Market
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
    market_id: str
    question: str
    posts: list[SocialPost] = Field(default_factory=list)
    sentiment: SentimentResult
    narrative_summary: str = ""
    key_claims: list[str] = Field(default_factory=list)
    researched_at: datetime = Field(default_factory=datetime.utcnow)
    trend_score: float = 50.0          # Google Trends interest for this market's topic
    whale_bid_imbalance: float = 0.0   # Net whale order book pressure on YES


# ── Prediction Models ─────────────────────────────────────────────────────────

class PredictionFeatures(BaseModel):
    """Features fed to both XGBoost and the LLM."""
    compound_sentiment: float
    positive_sentiment: float
    negative_sentiment: float
    post_count: int
    avg_engagement: float
    price_change_24h: float
    spread: float
    liquidity_usdc: float
    volume_24h_usdc: float
    time_to_resolution_days: float
    current_yes_price: float
    whale_bid_imbalance: float = 0.0   # CLOB order book: -1=heavy sells, +1=heavy buys
    trend_score: float = 50.0          # Google Trends interest 0-100 (50=average)


class Prediction(BaseModel):
    market_id: str
    question: str
    xgb_yes_probability: float
    llm_yes_probability: float
    calibrated_yes_probability: float   # weighted ensemble
    market_yes_price: float
    edge: float                          # calibrated_prob - market_price
    confidence: float                    # model confidence (0–1)
    side: MarketSide                     # which side has edge
    reasoning: str = ""                  # LLM reasoning
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    should_trade: bool = False


# ── Risk / Trade Models ───────────────────────────────────────────────────────

class BetSizing(BaseModel):
    kelly_fraction_full: float
    kelly_fraction_used: float    # fractional Kelly
    bet_usdc: float
    max_allowed_usdc: float
    bankroll_usdc: float
    edge: float
    odds: float                   # decimal odds


class TradeDecision(BaseModel):
    approved: bool
    rejection_reason: str = ""
    prediction: Prediction
    sizing: Optional[BetSizing] = None


class Trade(BaseModel):
    id: Optional[int] = None
    market_id: str
    question: str
    side: MarketSide
    entry_price: float
    bet_usdc: float
    shares: float
    status: TradeStatus = TradeStatus.PLACED
    outcome: TradeOutcome = TradeOutcome.PENDING
    pnl_usdc: float = 0.0
    tx_hash: str = ""
    placed_at: datetime = Field(default_factory=datetime.utcnow)
    settled_at: Optional[datetime] = None
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
    question: str
    findings: list[PostmortemFinding] = Field(default_factory=list)
    system_updates: list[str] = Field(default_factory=list)
    lessons_learned: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
