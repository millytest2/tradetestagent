"""
Step 4 — Risk Agent
────────────────────
Before any trade executes:
  1. Calculate fractional Kelly bet size
  2. Check bankroll limits
  3. Block if trade is too risky
  4. Place trade on-chain (or dry-run) if approved
  5. Monitor until settlement
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from config import settings
from core.database import (
    get_trade_stats,
    save_trade,
    update_trade_outcome,
)
from core.models import (
    BetSizing,
    FlaggedMarket,
    MarketSide,
    Prediction,
    Trade,
    TradeDecision,
    TradeOutcome,
    TradeStatus,
)
from integrations.polymarket import check_settlement, place_trade
from utils.kelly import compute_bet_sizing

logger = logging.getLogger(__name__)


# ── Dynamic Kelly ─────────────────────────────────────────────────────────────

CIRCUIT_BREAKER_FILE = "TRADING_PAUSED.txt"
CIRCUIT_BREAKER_WINDOW = 30     # look at last N settled trades
CIRCUIT_BREAKER_THRESHOLD = 0.52  # pause if win rate drops below this


def _check_rolling_circuit_breaker() -> None:
    """
    Examine the last 30 settled trades. If win rate < 52%, write a pause
    file and raise an exception that blocks all further trades.

    To resume trading manually: delete TRADING_PAUSED.txt
    """
    # If already paused, keep blocking
    import os
    if os.path.exists(CIRCUIT_BREAKER_FILE):
        with open(CIRCUIT_BREAKER_FILE) as f:
            msg = f.read().strip()
        raise RuntimeError(f"TRADING PAUSED — {msg}. Delete {CIRCUIT_BREAKER_FILE} to resume.")

    try:
        from core.database import SessionLocal, TradeRow
        with SessionLocal() as session:
            recent = (
                session.query(TradeRow)
                .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
                .order_by(TradeRow.settled_at.desc())
                .limit(CIRCUIT_BREAKER_WINDOW)
                .all()
            )
        if len(recent) < CIRCUIT_BREAKER_WINDOW:
            return   # not enough data yet

        wins = sum(1 for t in recent if t.outcome == "WIN")
        rate = wins / len(recent)

        if rate < CIRCUIT_BREAKER_THRESHOLD:
            msg = (
                f"Win rate {rate:.1%} on last {CIRCUIT_BREAKER_WINDOW} trades "
                f"fell below {CIRCUIT_BREAKER_THRESHOLD:.0%} threshold. "
                f"Strategy may be breaking down. Review before resuming."
            )
            with open(CIRCUIT_BREAKER_FILE, "w") as f:
                f.write(msg)
            logger.critical("CIRCUIT BREAKER TRIGGERED: %s", msg)
            raise RuntimeError(f"TRADING PAUSED — {msg}")

    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("Circuit breaker check failed (non-blocking): %s", e)


def _dynamic_kelly_multiplier() -> float:
    """
    Scale bet size based on recent performance over the last 15 settled trades.

      ≥ 70% win rate → 1.25x  (hot streak — press the edge)
      ≤ 45% win rate → 0.65x  (cold streak — reduce exposure)
      Otherwise      → 1.00x  (normal sizing)

    Requires at least 5 settled trades to activate (avoids noise on tiny samples).
    """
    try:
        from core.database import SessionLocal, TradeRow
        with SessionLocal() as session:
            recent = (
                session.query(TradeRow)
                .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
                .order_by(TradeRow.settled_at.desc())
                .limit(15)
                .all()
            )
        if len(recent) < 5:
            return 1.0
        wins = sum(1 for t in recent if t.outcome == "WIN")
        recent_rate = wins / len(recent)
        if recent_rate >= 0.70:
            logger.info(
                "Dynamic Kelly: 1.25x (hot streak — recent win rate %.0f%%)",
                recent_rate * 100,
            )
            return 1.25
        if recent_rate <= 0.45:
            logger.info(
                "Dynamic Kelly: 0.65x (cold streak — recent win rate %.0f%%)",
                recent_rate * 100,
            )
            return 0.65
        return 1.0
    except Exception:
        return 1.0


# ── Risk checks ───────────────────────────────────────────────────────────────

def _check_risk(
    prediction: Prediction,
    sizing: BetSizing,
    bankroll: float,
) -> tuple[bool, str]:
    """
    Run a battery of risk checks.
    Returns (approved: bool, reason: str).
    """
    # 1. Edge too small
    if prediction.edge < settings.min_edge:
        return False, f"Edge {prediction.edge:.3f} below minimum {settings.min_edge}"

    # 2. Confidence too low
    if prediction.confidence < settings.min_confidence:
        return False, (
            f"Confidence {prediction.confidence:.2f} below "
            f"minimum {settings.min_confidence}"
        )

    # 3. Bet size is zero (Kelly said don't bet)
    if sizing.bet_usdc <= 0:
        return False, "Kelly criterion returned zero or negative bet size"

    # 4. Bet exceeds max fraction of bankroll
    max_allowed = bankroll * settings.max_bet_fraction
    if sizing.bet_usdc > max_allowed:
        return False, (
            f"Bet ${sizing.bet_usdc:.2f} exceeds max allowed "
            f"${max_allowed:.2f} ({settings.max_bet_fraction:.0%} of bankroll)"
        )

    # 5. Bankroll too small for minimum bet
    if sizing.bet_usdc < 1.0:
        return False, f"Computed bet ${sizing.bet_usdc:.2f} below $1.00 dust threshold"

    # 6. Guard against betting more than 50% of bankroll on a single trade
    if sizing.bet_usdc > bankroll * 0.50:
        return False, "Single trade would exceed 50% of bankroll — hard cap"

    # 7. Win rate sanity check (don't trade if recent history is very bad)
    stats = get_trade_stats()
    if stats["total"] >= 10 and stats["win_rate"] < 0.30:
        return False, (
            f"Recent win rate {stats['win_rate']:.1%} is very low — "
            "circuit breaker triggered"
        )

    # 8. Rolling 30-trade circuit breaker — pause if strategy is breaking down
    _check_rolling_circuit_breaker()

    return True, ""


# ── Main Risk Agent ───────────────────────────────────────────────────────────

async def evaluate_and_trade(
    flagged: FlaggedMarket,
    prediction: Prediction,
    bankroll_usdc: Optional[float] = None,
    dry_run: bool = True,
) -> TradeDecision:
    """
    Evaluate risk, size the bet, and place the trade if approved.

    Args:
        flagged:      The flagged market from the scan agent.
        prediction:   Calibrated prediction from the prediction agent.
        bankroll_usdc: Current bankroll override (uses settings default if None).
        dry_run:      If True, simulate the trade without spending real money.

    Returns:
        TradeDecision with approval status and optional trade details.
    """
    bankroll = bankroll_usdc or settings.bankroll_usdc
    market = flagged.market

    # Skip if we already have an open position on this market
    from core.database import SessionLocal, TradeRow
    with SessionLocal() as _s:
        existing = (
            _s.query(TradeRow)
            .filter(
                TradeRow.market_id == market.condition_id,
                TradeRow.outcome == "PENDING",
            )
            .first()
        )
    if existing:
        return TradeDecision(
            approved=False,
            rejection_reason=f"Already have open position on this market (trade #{existing.id})",
            prediction=prediction,
            sizing=compute_bet_sizing(
                win_prob=prediction.calibrated_yes_probability,
                market_price=market.yes_price,
                bankroll_usdc=bankroll,
            ),
        )

    # Determine which price we're trading against
    if prediction.side == MarketSide.YES:
        market_price = market.yes_price
        win_prob = prediction.calibrated_yes_probability
    else:
        market_price = market.no_price
        # calibrated_yes_probability is already flipped to P(NO wins) in prediction_agent
        win_prob = prediction.calibrated_yes_probability

    # ── Bet sizing ────────────────────────────────────────────────────────────
    sizing = compute_bet_sizing(
        win_prob=win_prob,
        market_price=market_price,
        bankroll_usdc=bankroll,
    )

    # Apply dynamic Kelly multiplier (hot/cold streak adjustment)
    kelly_mult = _dynamic_kelly_multiplier()
    if kelly_mult != 1.0:
        adjusted_bet = min(
            sizing.bet_usdc * kelly_mult,
            bankroll * settings.max_bet_fraction,
        )
        sizing = sizing.model_copy(update={"bet_usdc": max(0.0, adjusted_bet)})

    logger.info(
        "Risk check — side=%s, win_prob=%.3f, market_p=%.3f, "
        "kelly=%.3f, bet=$%.2f, bankroll=$%.2f",
        prediction.side.value, win_prob, market_price,
        sizing.kelly_fraction_full, sizing.bet_usdc, bankroll,
    )

    # ── Risk gate ─────────────────────────────────────────────────────────────
    approved, rejection_reason = _check_risk(prediction, sizing, bankroll)

    if not approved:
        logger.warning("Trade BLOCKED: %s", rejection_reason)
        return TradeDecision(
            approved=False,
            rejection_reason=rejection_reason,
            prediction=prediction,
            sizing=sizing,
        )

    # ── Execute trade ─────────────────────────────────────────────────────────
    logger.info(
        "Trade APPROVED — placing %s on '%s' for $%.2f at %.3f [dry_run=%s]",
        prediction.side.value, market.question[:60],
        sizing.bet_usdc, market_price, dry_run,
    )

    try:
        trade = await place_trade(
            condition_id=market.condition_id,
            side=prediction.side,
            bet_usdc=sizing.bet_usdc,
            price=market_price,
            dry_run=dry_run,
        )
        trade.question = market.question

        # Persist to DB with feature snapshot for later ML training.
        # Sentiment fields default to 0.0 here; main.py may pass them via
        # extra_features to get the full picture.
        import json
        features_snapshot = {
            "compound_sentiment":   getattr(prediction, "_sentiment_compound", 0.0),
            "positive_sentiment":   getattr(prediction, "_sentiment_positive", 0.0),
            "negative_sentiment":   getattr(prediction, "_sentiment_negative", 0.0),
            "post_count":           getattr(prediction, "_post_count", 0),
            "avg_engagement":       getattr(prediction, "_avg_engagement", 0.0),
            "price_change_24h":     market.price_change_24h,
            "spread":               market.spread,
            "liquidity_usdc":       market.liquidity_usdc,
            "volume_24h_usdc":      market.volume_24h_usdc,
            "time_to_resolution_days": market.time_to_resolution_days,
            "current_yes_price":    market.yes_price,
            "whale_bid_imbalance":  getattr(prediction, "_whale_bid_imbalance", 0.0),
            "trend_score":          getattr(prediction, "_trend_score", 50.0),
        }
        trade.notes = json.dumps({"features": features_snapshot})

        trade_id = save_trade(trade)
        trade.id = trade_id

        logger.info(
            "Trade placed — id=%d, tx=%s", trade_id, trade.tx_hash
        )

        return TradeDecision(
            approved=True,
            prediction=prediction,
            sizing=sizing,
        )

    except Exception as e:
        logger.error("Trade execution failed: %s", e)
        return TradeDecision(
            approved=False,
            rejection_reason=f"Execution error: {e}",
            prediction=prediction,
            sizing=sizing,
        )


# ── Settlement monitor ────────────────────────────────────────────────────────

async def monitor_and_settle(
    trade: Trade,
    poll_interval_seconds: int = 3600,
    max_polls: int = 720,   # ~30 days at 1h intervals
) -> Optional[Trade]:
    """
    Poll until the market resolves and record the final outcome.

    Returns the updated Trade with outcome set, or None if monitoring
    was abandoned (e.g. max polls reached).
    """
    if trade.id is None:
        logger.warning("Cannot monitor trade without a DB id")
        return None

    logger.info(
        "Monitoring trade %d ('%s') — polling every %ds",
        trade.id, trade.question[:60], poll_interval_seconds,
    )

    for poll in range(max_polls):
        await asyncio.sleep(poll_interval_seconds)

        try:
            result = await check_settlement(trade)
            if not result or not result.get("resolved"):
                continue

            market = result.get("market")
            if not market:
                continue

            # Determine outcome from final market price
            if trade.side == MarketSide.YES:
                won = market.yes_price >= 0.95   # effectively resolved YES
            else:
                won = market.no_price >= 0.95    # effectively resolved NO

            if won:
                outcome = TradeOutcome.WIN
                # Return = shares * $1 (binary market pays $1 per winning share)
                pnl = trade.shares - trade.bet_usdc
            else:
                outcome = TradeOutcome.LOSS
                pnl = -trade.bet_usdc

            update_trade_outcome(
                trade_id=trade.id,
                outcome=outcome,
                pnl_usdc=pnl,
            )
            trade.outcome = outcome
            trade.pnl_usdc = pnl
            trade.settled_at = datetime.utcnow()
            trade.status = TradeStatus.SETTLED

            logger.info(
                "Trade %d settled — outcome=%s, pnl=$%.2f",
                trade.id, outcome.value, pnl,
            )
            return trade

        except Exception as e:
            logger.error("Settlement poll %d failed for trade %d: %s", poll, trade.id, e)

    logger.warning("Trade %d monitoring abandoned after %d polls", trade.id, max_polls)
    return None
