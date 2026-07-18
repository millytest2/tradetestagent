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
            try:
                from utils.notifications import notify_circuit_breaker
                notify_circuit_breaker(win_rate=rate, window=CIRCUIT_BREAKER_WINDOW)
            except Exception:
                pass
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


def _drawdown_governor() -> float:
    """
    Second sizing lever: shrink bets while the account is in a REALIZED
    drawdown, to protect capital exactly when the strategy is losing. Returns a
    multiplier in [0.4, 1.0]. At break-even or better it's 1.0 (no effect); a
    10% realized loss on the configured bankroll → ~0.9, a 30% loss → ~0.7,
    floored at 0.4 so it never fully stops.
    """
    try:
        stats = get_trade_stats()
        realized = stats.get("total_pnl_usdc", 0.0)
        if realized >= 0:
            return 1.0
        # Drawdown measured against INITIAL capital (current bankroll + what
        # was lost) — measuring against the current/fallback bankroll would
        # over-punish after a withdrawal shrinks the configured number.
        start = max(1.0, settings.bankroll_usdc + abs(realized))
        return max(0.4, min(1.0, 1.0 + realized / start))
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
    # 1. Edge too small — final gate uses the FEE-AWARE floor so a trade whose
    #    edge is eaten by fees can't slip through even if built directly.
    edge_floor = settings.min_edge + settings.fee_buffer
    if prediction.edge < edge_floor:
        return False, f"Edge {prediction.edge:.3f} below fee-aware floor {edge_floor:.3f}"

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

    # 5. Below the $1 dust threshold (matches utils/kelly.py) — sub-$1 bets are
    #    meaningless, so block them rather than place a token position.
    if sizing.bet_usdc < 1.0:
        return False, f"Computed bet ${sizing.bet_usdc:.2f} below $1.00 dust threshold"

    # 6. Guard against betting more than 50% of bankroll on a single trade
    if sizing.bet_usdc > bankroll * 0.50:
        return False, "Single trade would exceed 50% of bankroll — hard cap"

    # 7. Win rate sanity check — judged ONLY on trades placed under the
    #    CURRENT ruleset (on/after strategy_epoch). The pre-fix trades
    #    (wrong-side fills, no-LLM longshots, event stacking) settled last, so
    #    even a recent-N window reads as their failures; those code paths are
    #    structurally blocked and must not indict the fixed strategy. Until 10
    #    new-regime trades have settled there's nothing to judge — trade on.
    try:
        from datetime import datetime as _dt
        from core.database import SessionLocal, TradeRow
        epoch = _dt.fromisoformat(settings.strategy_epoch)
        with SessionLocal() as _s:
            recent = (
                _s.query(TradeRow)
                .filter(
                    TradeRow.outcome.in_(["WIN", "LOSS"]),
                    TradeRow.placed_at >= epoch,
                )
                .order_by(TradeRow.settled_at.desc())
                .limit(10)
                .all()
            )
        if len(recent) >= 10:
            recent_rate = sum(1 for t in recent if t.outcome == "WIN") / len(recent)
            if recent_rate < 0.30:
                return False, (
                    f"Win rate {recent_rate:.1%} over last {len(recent)} settled "
                    f"(placed since {settings.strategy_epoch}) — circuit breaker triggered"
                )
    except Exception as e:
        logger.debug("Recent win-rate check failed (non-blocking): %s", e)

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
    # Use the passed bankroll when provided (including a real 0.0 — do NOT let
    # `or` fall through to the configured default, which would size a huge bet
    # off an empty wallet). Fall back to config only when None.
    bankroll = bankroll_usdc if bankroll_usdc is not None else settings.bankroll_usdc
    market = flagged.market
    if bankroll <= 0:
        return TradeDecision(
            approved=False,
            rejection_reason="No spendable bankroll",
            prediction=prediction,
            sizing=compute_bet_sizing(
                win_prob=prediction.calibrated_yes_probability,
                market_price=flagged.market.yes_price,
                bankroll_usdc=0.0,
            ),
        )

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

    # A backed favorite (favorites fast-path) is a flat-bet win-rate play — Kelly
    # returns ~0 on a fair-value favorite, so seed the base bet at the meaningful
    # minimum; the multipliers + cap below still apply.
    if getattr(prediction, "_favorite_flat", False) and sizing.bet_usdc < settings.min_bet_usdc:
        sizing = sizing.model_copy(update={"bet_usdc": min(
            settings.min_bet_usdc, bankroll * settings.max_bet_fraction,
        )})

    # ── Sizing multipliers ────────────────────────────────────────────────────
    # (a) Streak multiplier — press on hot streaks, pull back on cold ones.
    # (b) Conviction scaling — Kelly assumes the win probability is known
    #     exactly, but ours is an estimate; scale by prediction confidence
    #     (floored at 0.5x so low-conviction trades still place small and keep
    #     feeding the learning loop).
    kelly_mult = _dynamic_kelly_multiplier()
    conf_scale = 0.5 + 0.5 * min(1.0, max(0.0, prediction.confidence))
    conviction_mult = kelly_mult * conf_scale
    if conviction_mult != 1.0:
        adjusted_bet = min(
            sizing.bet_usdc * conviction_mult,
            bankroll * settings.max_bet_fraction,
        )
        sizing = sizing.model_copy(update={"bet_usdc": max(0.0, adjusted_bet)})
    # Meaningful-minimum floor: on a small bankroll, thin-edge favorites compute
    # Kelly bets in cents and the $1 dust rule would block everything. A signal
    # that earned a positive Kelly bet places at least min_bet_usdc (still
    # capped by max_bet_fraction).
    if sizing.bet_usdc > 0:
        floored = min(max(sizing.bet_usdc, settings.min_bet_usdc),
                      bankroll * settings.max_bet_fraction)
        sizing = sizing.model_copy(update={"bet_usdc": floored})
    if conviction_mult != 1.0:
        logger.info(
            "Sizing: streak=%.2fx × conviction=%.2fx (conf=%.2f) = %.2fx → $%.2f",
            kelly_mult, conf_scale, prediction.confidence, conviction_mult,
            sizing.bet_usdc,
        )

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

    # ── A/B variant assignment ─────────────────────────────────────────────────
    ab_variant = "A"
    use_governor = True   # default when A/B is disabled
    if settings.ab_testing_enabled:
        from core.ab_testing import get_variant_for_trade
        variant = get_variant_for_trade()
        ab_variant = variant.name
        use_governor = getattr(variant, "use_drawdown_governor", True)
        # Apply variant sizing as a SCALE on the configured kelly_fraction, so
        # the KELLY_FRACTION env knob stays meaningful (a fixed per-variant
        # kelly used to silently override it).
        if sizing.bet_usdc > 0:
            from utils.kelly import compute_bet_sizing as _cbs
            variant_sizing = _cbs(
                win_prob=win_prob,
                market_price=market_price,
                bankroll_usdc=bankroll,
                kelly_override=settings.kelly_fraction * variant.kelly_scale,
            )
            # Preserve the streak + conviction multipliers through the variant
            # recompute — otherwise A/B sizing would silently discard them.
            variant_bet = min(
                variant_sizing.bet_usdc * conviction_mult,
                bankroll * settings.max_bet_fraction,
            )
            sizing = sizing.model_copy(update={"bet_usdc": max(0.0, variant_bet)})

    # ── Second sizing lever: drawdown governor (A/B-tested) ────────────────────
    # Variant A applies it (shrinks bets when the account is in a realized
    # drawdown); Variant B skips it (stays aggressive). Real A/B data then shows
    # whether the governor improves results.
    if use_governor:
        governor = _drawdown_governor()
        if governor < 1.0 and sizing.bet_usdc > 0:
            sizing = sizing.model_copy(
                update={"bet_usdc": max(0.0, sizing.bet_usdc * governor)}
            )
            logger.info("Drawdown governor: ×%.2f → bet=$%.2f", governor, sizing.bet_usdc)

    # Re-apply the meaningful-minimum floor after the variant/governor resizes so
    # an approved signal doesn't get shrunk below the $1 dust re-check.
    if sizing.bet_usdc > 0:
        sizing = sizing.model_copy(update={"bet_usdc": min(
            max(sizing.bet_usdc, settings.min_bet_usdc),
            bankroll * settings.max_bet_fraction,
        )})

    # PM-US SHORT-side safety: BUY_SHORT execution semantics are unverified on
    # the US API (evidence of positions filling on the wrong side at the
    # complementary price). Until a live order is verified, block NO-side
    # live trades rather than risk buying the opposite of what we sized.
    if (prediction.side == MarketSide.NO and not dry_run
            and not settings.allow_short_side
            and settings.live_exchange.lower() in ("polymarket_us", "polymarketus", "pmus")):
        return TradeDecision(
            approved=False,
            rejection_reason="NO-side (BUY_SHORT) execution unverified on PM-US — blocked",
            prediction=prediction,
            sizing=sizing,
        )

    # Re-validate the dust floor AFTER the variant/governor resize — those run
    # past _check_risk and could have shrunk an approved bet below $1.
    if sizing.bet_usdc < 1.0:
        logger.warning("Post-sizing bet ${:.2f} fell below $1 — skipping".format(sizing.bet_usdc))
        return TradeDecision(
            approved=False,
            rejection_reason=f"Final bet ${sizing.bet_usdc:.2f} below $1 after sizing",
            prediction=prediction,
            sizing=sizing,
        )

    # ── Execute trade ─────────────────────────────────────────────────────────
    logger.info(
        "Trade APPROVED [Variant %s] — placing %s on '%s' for $%.2f at %.3f [dry_run=%s exchange=%s]",
        ab_variant, prediction.side.value, market.question[:60],
        sizing.bet_usdc, market_price, dry_run, settings.live_exchange,
    )

    try:
        # ── Exchange routing ───────────────────────────────────────────────────
        # Markets carry their source. Kalshi tickers are uppercase alnum
        # (e.g. "KXBTCD-25"); Polymarket condition_ids are 0x-prefixed hashes.
        exchange = settings.live_exchange.lower()
        market_is_kalshi = not market.condition_id.startswith("0x")

        if exchange == "kalshi" or (exchange == "both" and market_is_kalshi):
            # Route to Kalshi (US-legal)
            from integrations.kalshi import place_trade as kalshi_place
            trade = await kalshi_place(
                condition_id=market.condition_id,
                side=prediction.side,
                bet_usdc=sizing.bet_usdc,
                price=market_price,
                dry_run=dry_run,
            )
        elif exchange in ("polymarket_us", "polymarketus", "pmus"):
            # Route to Polymarket US (CFTC-regulated, US-legal — api.polymarket.us)
            from integrations.polymarket_us import place_trade as pmus_place
            trade = await pmus_place(
                condition_id=market.condition_id,
                side=prediction.side,
                bet_usdc=sizing.bet_usdc,
                price=market_price,
                dry_run=dry_run,
                slug=market.slug,
            )
        else:
            from integrations.polymarket import place_trade as poly_place
            trade = await poly_place(
                condition_id=market.condition_id,
                side=prediction.side,
                bet_usdc=sizing.bet_usdc,
                price=market_price,
                dry_run=dry_run,
            )
        trade.question = market.question

        import json
        features_snapshot = {
            "compound_sentiment":      getattr(prediction, "_sentiment_compound", 0.0),
            "positive_sentiment":      getattr(prediction, "_sentiment_positive", 0.0),
            "negative_sentiment":      getattr(prediction, "_sentiment_negative", 0.0),
            "post_count":              getattr(prediction, "_post_count", 0),
            "avg_engagement":          getattr(prediction, "_avg_engagement", 0.0),
            "price_change_24h":        market.price_change_24h,
            "spread":                  market.spread,
            "liquidity_usdc":          market.liquidity_usdc,
            "volume_24h_usdc":         market.volume_24h_usdc,
            "time_to_resolution_days": market.time_to_resolution_days,
            "current_yes_price":       market.yes_price,
            "whale_bid_imbalance":     getattr(prediction, "_whale_bid_imbalance", 0.0),
            "trend_score":             getattr(prediction, "_trend_score", 50.0),
        }
        trade.notes = json.dumps({
            "features": features_snapshot,
            "ab_variant": ab_variant,
            "exchange": exchange,
        })

        trade_id = save_trade(trade)
        trade.id = trade_id
        logger.info("Trade placed — id=%d variant=%s exchange=%s tx=%s",
                    trade_id, ab_variant, exchange, trade.tx_hash)

        # Email alert for the placed trade
        try:
            from utils.notifications import notify_trade_placed
            notify_trade_placed(
                question=market.question,
                side=prediction.side.value,
                bet_usdc=sizing.bet_usdc,
                price=market_price,
                edge=prediction.edge,
                bankroll=bankroll,
            )
        except Exception:
            pass

        return TradeDecision(approved=True, prediction=prediction, sizing=sizing)

    except Exception as e:
        logger.error("Trade execution failed: %s", e)
        return TradeDecision(
            approved=False,
            rejection_reason=f"Execution error: {e}",
            prediction=prediction,
            sizing=sizing,
        )

# (monitor_and_settle removed: dead code — it referenced an unimported
#  check_settlement and a dict return shape no exchange module provides.
#  Settlement runs through main._settle_pending_trades each cycle instead.)
