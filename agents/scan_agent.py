"""
Step 1 — Scan Agent
────────────────────
Fetches up to 300 active Polymarket markets, filters by liquidity /
volume / time-to-resolution, and flags any with unusual price action
or wide spreads for deeper research.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import settings
from core.models import FlaggedMarket, Market
from integrations.polymarket import get_active_markets

logger = logging.getLogger(__name__)


def _passes_base_filter(market: Market) -> bool:
    """Return True if the market meets basic quality thresholds."""
    if market.liquidity_usdc < settings.min_liquidity_usdc:
        return False
    if market.volume_24h_usdc < settings.min_volume_usdc:
        return False
    if market.time_to_resolution_days > settings.max_time_to_resolution_days:
        return False
    # HARD RULE (from PatternAgent postmortem): never trade already-expired markets.
    # time_to_resolution_days < 0 means the deadline has passed.
    if market.time_to_resolution_days < 0:
        return False
    if market.time_to_resolution_days < settings.min_time_to_resolution_days:
        return False
    # Skip markets that are already at near-certainty prices (>97% or <3%)
    if market.yes_price >= 0.97 or market.yes_price <= 0.03:
        return False
    return True


def _detect_anomaly(market: Market) -> tuple[bool, str]:
    """
    Detect unusual price moves or spreads that suggest new information.

    Returns (is_flagged, reason).
    """
    reasons: list[str] = []

    if market.price_change_24h >= settings.weird_price_move_threshold:
        reasons.append(
            f"price moved {market.price_change_24h:.1%} in 24h"
        )

    if market.spread >= settings.weird_spread_threshold:
        reasons.append(
            f"wide spread {market.spread:.1%}"
        )

    # High volume relative to liquidity suggests rush of informed trading
    if market.liquidity_usdc > 0:
        vol_to_liq = market.volume_24h_usdc / market.liquidity_usdc
        if vol_to_liq >= 2.0:
            reasons.append(
                f"volume/liquidity ratio {vol_to_liq:.1f}x (unusual activity)"
            )

    flagged = len(reasons) > 0
    return flagged, "; ".join(reasons)


def _priority_score(market: Market, flag_reason: str) -> float:
    """
    Score a market by trading opportunity priority.
    Higher is more interesting.
    """
    score = 0.0

    # Edge potential: prices near 0.5 → most uncertainty → most potential edge
    distance_from_fair = abs(market.yes_price - 0.5)
    score += (0.5 - distance_from_fair) * 4.0   # 0–2 pts

    # Anomaly signal — fresh information = opportunity
    if flag_reason:
        score += 2.0

    # Liquidity — more liquid = safer execution, tighter slippage
    score += min(market.liquidity_usdc / 10_000, 2.0)

    # Volume/liquidity ratio — high ratio = informed trading activity
    if market.liquidity_usdc > 0:
        vol_ratio = market.volume_24h_usdc / market.liquidity_usdc
        score += min(vol_ratio * 0.5, 1.0)
    else:
        score += min(market.volume_24h_usdc / 5_000, 1.0)

    # Prefer markets resolving sooner (but not too soon) — faster feedback
    days = market.time_to_resolution_days
    if 1 <= days <= 3:
        score += 2.0   # imminent — highest priority
    elif 4 <= days <= 10:
        score += 1.5
    elif days <= 30:
        score += 0.5

    # Penalise extreme prices — harder to find edge near certainty
    if market.yes_price >= 0.88 or market.yes_price <= 0.12:
        score -= 1.0

    return score


async def scan_markets(limit: int = 300) -> list[FlaggedMarket]:
    """
    Scan prediction markets and return flagged opportunities.

    Pipeline:
      1. Fetch up to `limit` active markets from Polymarket
      2. Apply base filters (liquidity, volume, time window)
      3. Detect anomalies (price moves, spreads)
      4. Score and sort by opportunity priority
    """
    logger.info("Scan agent starting — fetching up to %d markets...", limit)

    markets = await get_active_markets(limit=limit)
    if not markets:
        logger.warning("No markets returned from Polymarket API")
        return []

    passing: list[Market] = []
    for m in markets:
        if _passes_base_filter(m):
            passing.append(m)

    logger.info(
        "%d / %d markets passed base filter (liq≥$%.0f, vol≥$%.0f, %d–%dd)",
        len(passing), len(markets),
        settings.min_liquidity_usdc, settings.min_volume_usdc,
        settings.min_time_to_resolution_days, settings.max_time_to_resolution_days,
    )

    flagged: list[FlaggedMarket] = []
    for m in passing:
        is_flagged, reason = _detect_anomaly(m)
        m.is_flagged = is_flagged
        m.flag_reason = reason
        score = _priority_score(m, reason)

        flagged.append(FlaggedMarket(
            market=m,
            flag_reason=reason or "base filter pass",
            priority_score=score,
        ))

    # Sort highest priority first
    flagged.sort(key=lambda x: x.priority_score, reverse=True)

    anomaly_count = sum(1 for f in flagged if f.market.is_flagged)
    logger.info(
        "Scan complete — %d markets queued (%d with anomalies)",
        len(flagged), anomaly_count,
    )
    return flagged
