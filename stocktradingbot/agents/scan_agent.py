"""
Step 1 — Scan Agent
────────────────────
Scans the S&P 500 universe, computes technical indicators, and flags
stocks with unusual momentum, volume, or RSI signals for deeper research.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from config import settings
from core.models import FlaggedStock, Stock
from integrations.alpaca import SP500_UNIVERSE, build_stock

logger = logging.getLogger(__name__)

# Concurrent fetch limit — avoid rate-limiting Alpaca data API
_SEMAPHORE = asyncio.Semaphore(10)


def _passes_base_filter(stock: Stock) -> bool:
    if stock.current_price < settings.min_price:
        return False
    if stock.avg_volume_20d < settings.min_volume:
        return False
    # Skip earnings — too binary, not our edge
    if stock.days_to_earnings is not None and stock.days_to_earnings <= settings.earnings_buffer_days:
        return False
    return True


def _detect_signal(stock: Stock) -> tuple[bool, str]:
    """Detect actionable technical signals. Returns (flagged, reason)."""
    reasons: list[str] = []

    # RSI extremes
    if stock.rsi_14 < settings.rsi_oversold:
        reasons.append(f"RSI oversold ({stock.rsi_14:.1f})")
    elif stock.rsi_14 > settings.rsi_overbought:
        reasons.append(f"RSI overbought ({stock.rsi_14:.1f})")

    # Unusual volume
    if stock.volume_ratio >= settings.volume_ratio_threshold:
        reasons.append(f"volume surge {stock.volume_ratio:.1f}x avg")

    # Price momentum
    if abs(stock.price_change_1d) >= settings.price_change_threshold:
        direction = "up" if stock.price_change_1d > 0 else "down"
        reasons.append(f"1d move {stock.price_change_1d:+.1%} ({direction})")

    # MACD cross (meaningful signal)
    if abs(stock.macd_signal) > 0.5:
        direction = "bullish" if stock.macd_signal > 0 else "bearish"
        reasons.append(f"MACD histogram {stock.macd_signal:+.2f} ({direction})")

    # Bollinger Band extremes
    if stock.bb_position <= 0.05:
        reasons.append(f"at lower Bollinger Band (bb={stock.bb_position:.2f})")
    elif stock.bb_position >= 0.95:
        reasons.append(f"at upper Bollinger Band (bb={stock.bb_position:.2f})")

    return bool(reasons), "; ".join(reasons)


def _priority_score(stock: Stock, flag_reason: str) -> float:
    score = 0.0

    # RSI extremes → mean reversion opportunity
    rsi_distance = max(0, stock.rsi_14 - 70) + max(0, 30 - stock.rsi_14)
    score += rsi_distance * 0.1   # 0–3 pts

    # Volume surge → informed activity
    score += min((stock.volume_ratio - 1) * 0.5, 2.0)

    # Price move → catalyst present
    score += min(abs(stock.price_change_1d) * 10, 2.0)

    # MACD strength
    score += min(abs(stock.macd_signal) * 0.5, 1.0)

    # Bollinger Band extremes → reversion likely
    bb_extreme = max(stock.bb_position - 0.85, 0) + max(0.15 - stock.bb_position, 0)
    score += bb_extreme * 5.0   # 0–0.75 pts

    # Flag bonus
    if flag_reason:
        score += 2.0

    return round(score, 3)


async def _fetch_with_semaphore(ticker: str) -> Optional[Stock]:
    async with _SEMAPHORE:
        return await build_stock(ticker)


async def scan_stocks(universe: Optional[list[str]] = None) -> list[FlaggedStock]:
    """
    Scan stocks in universe and return flagged opportunities.

    Pipeline:
      1. Fetch bar history + compute indicators concurrently
      2. Apply base filters
      3. Detect technical signals
      4. Score and sort by priority
    """
    tickers = universe or SP500_UNIVERSE
    logger.info("Scan agent: fetching %d stocks...", len(tickers))

    tasks = [_fetch_with_semaphore(t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    stocks: list[Stock] = []
    for t, r in zip(tickers, results):
        if isinstance(r, Stock):
            stocks.append(r)
        elif isinstance(r, Exception):
            logger.debug("Failed to fetch %s: %s", t, r)

    logger.info("%d / %d stocks fetched successfully", len(stocks), len(tickers))

    passing: list[Stock] = [s for s in stocks if _passes_base_filter(s)]
    logger.info("%d passed base filter (price≥$%.0f, vol≥%s)",
                len(passing), settings.min_price, f"{settings.min_volume:,}")

    flagged: list[FlaggedStock] = []
    for stock in passing:
        is_flagged, reason = _detect_signal(stock)
        if not is_flagged:
            continue  # only trade stocks with an actual signal
        score = _priority_score(stock, reason)
        flagged.append(FlaggedStock(
            stock=stock,
            flag_reason=reason,
            priority_score=score,
        ))

    flagged.sort(key=lambda x: x.priority_score, reverse=True)

    logger.info(
        "Scan complete — %d flagged stocks (top: %s at %.2f)",
        len(flagged),
        flagged[0].stock.ticker if flagged else "none",
        flagged[0].priority_score if flagged else 0.0,
    )
    return flagged
