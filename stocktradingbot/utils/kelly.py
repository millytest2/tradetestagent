"""Kelly Criterion bet sizing utilities."""

from __future__ import annotations

from config import settings
from core.models import BetSizing


def kelly_fraction(win_prob: float, odds: float) -> float:
    """
    Standard Kelly fraction: f* = (b*p - q) / b
    where b = decimal odds - 1, p = win prob, q = 1 - p.
    Returns 0 if no edge.
    """
    if odds <= 1.0 or win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    b = odds - 1.0
    p = win_prob
    q = 1.0 - p
    return max(0.0, (b * p - q) / b)


def compute_bet_sizing(
    win_prob: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    bankroll_usd: float,
) -> BetSizing:
    """
    Compute fractional Kelly bet size for a stock trade.

    Args:
        win_prob:        Probability the trade wins (reaches TP before SL)
        stop_loss_pct:   Stop loss as fraction (e.g. 0.03 = 3%)
        take_profit_pct: Take profit as fraction (e.g. 0.05 = 5%)
        bankroll_usd:    Current bankroll
    """
    if stop_loss_pct <= 0 or take_profit_pct <= 0:
        return BetSizing(
            kelly_fraction_full=0.0, kelly_fraction_used=0.0,
            bet_usd=0.0,
            max_allowed_usd=bankroll_usd * settings.max_bet_fraction,
            bankroll_usd=bankroll_usd,
            edge=win_prob - 0.5,
            odds=take_profit_pct / stop_loss_pct,
        )

    odds = 1.0 + take_profit_pct / stop_loss_pct
    kf_full = kelly_fraction(win_prob, odds)
    kf_used = kf_full * settings.kelly_fraction

    max_allowed = bankroll_usd * settings.max_bet_fraction
    raw_bet = kf_used * bankroll_usd
    bet_usd = min(raw_bet, max_allowed)
    if bet_usd < 1.0:
        bet_usd = 0.0

    return BetSizing(
        kelly_fraction_full=kf_full,
        kelly_fraction_used=kf_used,
        bet_usd=bet_usd,
        max_allowed_usd=max_allowed,
        bankroll_usd=bankroll_usd,
        edge=win_prob - 0.5,
        odds=odds,
    )
