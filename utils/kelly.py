"""Kelly Criterion bet sizing utilities."""

from __future__ import annotations

from core.models import BetSizing
from config import settings


def kelly_fraction(win_prob: float, odds: float) -> float:
    """
    Standard Kelly fraction formula.

    f* = (b*p - q) / b
    where:
      b = decimal odds - 1  (net odds on a $1 bet)
      p = win probability
      q = 1 - p (loss probability)

    Returns a value in [-1, 1]. Negative means don't bet.
    """
    if odds <= 1.0 or win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    b = odds - 1.0
    p = win_prob
    q = 1.0 - p
    return (b * p - q) / b


def compute_bet_sizing(
    win_prob: float,
    market_price: float,
    bankroll_usdc: float,
) -> BetSizing:
    """
    Compute bet size using fractional Kelly.

    For a binary prediction market:
      - If you bet YES at price p, payout is 1/p per share (decimal odds = 1/p)
      - Edge = win_prob - market_price

    Returns a BetSizing with recommended bet amount.
    """
    if market_price <= 0 or market_price >= 1:
        return BetSizing(
            kelly_fraction_full=0.0,
            kelly_fraction_used=0.0,
            bet_usdc=0.0,
            max_allowed_usdc=bankroll_usdc * settings.max_bet_fraction,
            bankroll_usdc=bankroll_usdc,
            edge=win_prob - market_price,
            odds=1.0 / max(market_price, 1e-6),
        )

    odds = 1.0 / market_price          # decimal odds
    kf_full = kelly_fraction(win_prob, odds)
    kf_used = kf_full * settings.kelly_fraction  # fractional Kelly

    # Cap at max_bet_fraction of bankroll
    max_allowed = bankroll_usdc * settings.max_bet_fraction
    raw_bet = max(0.0, kf_used * bankroll_usdc)
    bet_usdc = min(raw_bet, max_allowed)

    # Minimum $1 bet (dust threshold)
    if bet_usdc < 1.0:
        bet_usdc = 0.0

    return BetSizing(
        kelly_fraction_full=kf_full,
        kelly_fraction_used=kf_used,
        bet_usdc=bet_usdc,
        max_allowed_usdc=max_allowed,
        bankroll_usdc=bankroll_usdc,
        edge=win_prob - market_price,
        odds=odds,
    )
