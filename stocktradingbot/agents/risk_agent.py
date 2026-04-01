"""
Step 4 — Risk Agent
────────────────────
Computes Kelly bet sizing, runs risk checks, places bracket orders via Alpaca.
Includes dynamic Kelly multiplier and rolling circuit breaker.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from config import settings
from core.database import (
    count_open_positions,
    get_trade_stats,
    save_trade,
    update_trade_outcome,
)
from core.models import (
    BetSizing,
    FlaggedStock,
    Prediction,
    StockSide,
    Trade,
    TradeDecision,
    TradeOutcome,
    TradeStatus,
)
from integrations.alpaca import get_bankroll, place_bracket_order

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_FILE = "TRADING_PAUSED.txt"
CIRCUIT_BREAKER_WINDOW = 30
CIRCUIT_BREAKER_THRESHOLD = 0.52


def _check_circuit_breaker() -> None:
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
                .order_by(TradeRow.exit_date.desc())
                .limit(CIRCUIT_BREAKER_WINDOW)
                .all()
            )
        if len(recent) < CIRCUIT_BREAKER_WINDOW:
            return

        wins = sum(1 for t in recent if t.outcome == "WIN")
        rate = wins / len(recent)

        if rate < CIRCUIT_BREAKER_THRESHOLD:
            msg = (
                f"Win rate {rate:.1%} on last {CIRCUIT_BREAKER_WINDOW} trades "
                f"fell below {CIRCUIT_BREAKER_THRESHOLD:.0%}. "
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
    try:
        from core.database import SessionLocal, TradeRow
        with SessionLocal() as session:
            recent = (
                session.query(TradeRow)
                .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
                .order_by(TradeRow.exit_date.desc())
                .limit(15)
                .all()
            )
        if len(recent) < 5:
            return 1.0
        wins = sum(1 for t in recent if t.outcome == "WIN")
        rate = wins / len(recent)
        if rate >= 0.70:
            logger.info("Dynamic Kelly: 1.25x (hot streak %.0f%%)", rate * 100)
            return 1.25
        if rate <= 0.45:
            logger.info("Dynamic Kelly: 0.65x (cold streak %.0f%%)", rate * 100)
            return 0.65
        return 1.0
    except Exception:
        return 1.0


def _compute_kelly(prediction: Prediction, bankroll: float) -> BetSizing:
    """Kelly sizing for a stock trade with stop-loss / take-profit."""
    # Odds ratio = take_profit / stop_loss (risk/reward)
    tp_pct = settings.take_profit_pct
    sl_pct = settings.stop_loss_pct
    odds = tp_pct / sl_pct  # e.g. 5% TP / 3% SL = 1.67

    p = prediction.calibrated_up_probability if prediction.side == StockSide.LONG else (
        1 - prediction.calibrated_up_probability
    )
    q = 1 - p
    b = odds - 1.0
    kf_full = max(0.0, (b * p - q) / b) if b > 0 else 0.0
    kf_used = kf_full * settings.kelly_fraction

    max_allowed = bankroll * settings.max_bet_fraction
    raw_bet = kf_used * bankroll
    bet_usd = min(raw_bet, max_allowed)
    if bet_usd < 1.0:
        bet_usd = 0.0

    return BetSizing(
        kelly_fraction_full=kf_full,
        kelly_fraction_used=kf_used,
        bet_usd=bet_usd,
        max_allowed_usd=max_allowed,
        bankroll_usd=bankroll,
        edge=prediction.edge,
        odds=odds,
    )


def _check_risk(prediction: Prediction, sizing: BetSizing, bankroll: float) -> tuple[bool, str]:
    if prediction.edge < settings.min_edge:
        return False, f"Edge {prediction.edge:.3f} below minimum {settings.min_edge}"

    if prediction.confidence < settings.min_confidence:
        return False, f"Confidence {prediction.confidence:.2f} below minimum {settings.min_confidence}"

    if sizing.bet_usd <= 0:
        return False, "Kelly criterion returned zero bet"

    if sizing.bet_usd > bankroll * settings.max_bet_fraction:
        return False, f"Bet ${sizing.bet_usd:.2f} exceeds max fraction"

    if sizing.bet_usd < 1.0:
        return False, f"Bet ${sizing.bet_usd:.2f} below $1 dust threshold"

    if sizing.bet_usd > bankroll * 0.50:
        return False, "Single trade would exceed 50% of bankroll — hard cap"

    open_count = count_open_positions()
    if open_count >= settings.max_open_positions:
        return False, f"Max open positions reached ({open_count}/{settings.max_open_positions})"

    stats = get_trade_stats()
    if stats["total"] >= 10 and stats["win_rate"] < 0.30:
        return False, f"Win rate {stats['win_rate']:.1%} is very low — circuit breaker"

    _check_circuit_breaker()
    return True, ""


async def evaluate_and_trade(
    flagged: FlaggedStock,
    prediction: Prediction,
    bankroll_usd: Optional[float] = None,
    dry_run: bool = True,
) -> TradeDecision:
    """Evaluate risk, size bet, and place bracket order if approved."""
    bankroll = bankroll_usd
    if bankroll is None:
        try:
            bankroll = await get_bankroll()
        except Exception:
            bankroll = settings.bankroll_usd

    stock = flagged.stock
    sizing = _compute_kelly(prediction, bankroll)

    # Dynamic Kelly multiplier
    kelly_mult = _dynamic_kelly_multiplier()
    if kelly_mult != 1.0:
        adjusted = min(sizing.bet_usd * kelly_mult, bankroll * settings.max_bet_fraction)
        sizing = sizing.model_copy(update={"bet_usd": max(0.0, adjusted)})

    logger.info(
        "Risk check — $%s %s | edge=%.3f conf=%.2f bet=$%.2f bankroll=$%.2f",
        stock.ticker, prediction.side.value,
        prediction.edge, prediction.confidence, sizing.bet_usd, bankroll,
    )

    approved, rejection_reason = _check_risk(prediction, sizing, bankroll)

    if not approved:
        logger.warning("Trade BLOCKED ($%s): %s", stock.ticker, rejection_reason)
        return TradeDecision(
            approved=False,
            rejection_reason=rejection_reason,
            prediction=prediction,
            sizing=sizing,
        )

    # Calculate shares and bracket prices
    shares = sizing.bet_usd / stock.current_price
    if prediction.side == StockSide.LONG:
        stop_loss_price = stock.current_price * (1 - settings.stop_loss_pct)
        take_profit_price = stock.current_price * (1 + settings.take_profit_pct)
    else:
        stop_loss_price = stock.current_price * (1 + settings.stop_loss_pct)
        take_profit_price = stock.current_price * (1 - settings.take_profit_pct)

    logger.info(
        "Trade APPROVED — $%s %s %.4f shares @ $%.2f | SL=%.2f TP=%.2f [dry_run=%s]",
        stock.ticker, prediction.side.value, shares,
        stock.current_price, stop_loss_price, take_profit_price, dry_run,
    )

    try:
        order = await place_bracket_order(
            ticker=stock.ticker,
            side=prediction.side,
            shares=shares,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            dry_run=dry_run,
        )

        trade = Trade(
            ticker=stock.ticker,
            company_name=stock.company_name,
            side=prediction.side,
            entry_price=stock.current_price,
            bet_usd=sizing.bet_usd,
            shares=shares,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            status=TradeStatus.PLACED,
            alpaca_order_id=order.get("id", ""),
        )

        import json
        trade.notes = json.dumps({
            "features": {
                "rsi_14": stock.rsi_14,
                "macd_signal": stock.macd_signal,
                "bb_position": stock.bb_position,
                "volume_ratio": stock.volume_ratio,
                "price_change_1d": stock.price_change_1d,
                "price_change_5d": stock.price_change_5d,
                "price_change_20d": stock.price_change_20d,
                "distance_from_52w_high": stock.distance_from_52w_high,
                "short_interest_ratio": stock.short_interest_ratio,
                "compound_sentiment": prediction._sentiment_compound if hasattr(prediction, "_sentiment_compound") else 0.0,
                "post_count": 0,
                "avg_engagement": 0.0,
                "trend_score": 50.0,
                "whale_bid_imbalance": 0.0,
            }
        })

        trade_id = save_trade(trade)
        trade.id = trade_id
        logger.info("Trade saved — id=%d, order=%s", trade_id, order.get("id", ""))

        return TradeDecision(approved=True, prediction=prediction, sizing=sizing)

    except Exception as e:
        logger.error("Trade execution failed for $%s: %s", stock.ticker, e)
        return TradeDecision(
            approved=False,
            rejection_reason=f"Execution error: {e}",
            prediction=prediction,
            sizing=sizing,
        )
