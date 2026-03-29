"""Performance analytics computed from the trade history."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

from config import settings
from core.database import SessionLocal, TradeRow


def _settled_trades() -> list[TradeRow]:
    with SessionLocal() as s:
        return s.query(TradeRow).filter(TradeRow.outcome.in_(["WIN", "LOSS"])).order_by(TradeRow.settled_at).all()


def _all_trades() -> list[TradeRow]:
    with SessionLocal() as s:
        return s.query(TradeRow).order_by(TradeRow.placed_at).all()


# ── Core metrics ──────────────────────────────────────────────────────────────

def win_rate() -> float:
    rows = _settled_trades()
    if not rows:
        return 0.0
    wins = sum(1 for r in rows if r.outcome == "WIN")
    return wins / len(rows)


def total_pnl() -> float:
    return sum(r.pnl_usdc for r in _settled_trades())


def pnl_series() -> list[tuple[datetime, float]]:
    """Cumulative PnL over time (one point per settled trade)."""
    rows = _settled_trades()
    cumulative, series = 0.0, []
    for r in rows:
        cumulative += r.pnl_usdc
        series.append((r.settled_at or r.placed_at, cumulative))
    return series


def bankroll_series(initial: float = 1000.0) -> list[tuple[datetime, float]]:
    series = pnl_series()
    return [(dt, initial + pnl) for dt, pnl in series]


def sharpe_ratio(risk_free: float = 0.0) -> float:
    """Daily Sharpe ratio from settled trade PnL."""
    rows = _settled_trades()
    if len(rows) < 2:
        return 0.0
    pnls = [r.pnl_usdc for r in rows]
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (mean - risk_free) / std * math.sqrt(365)   # annualised


def max_drawdown() -> float:
    """Maximum peak-to-trough drawdown as a fraction of initial bankroll."""
    series = [pnl for _, pnl in pnl_series()]
    if not series:
        return 0.0
    # Anchor denominator to initial bankroll — avoids divide-by-near-zero
    # when peak PnL is tiny (which caused 1000%+ readings).
    denominator = max(settings.bankroll_usdc, 1.0)
    peak_pnl = 0.0   # start at 0: no gain yet at trade 0
    max_dd = 0.0
    for pnl in series:
        if pnl > peak_pnl:
            peak_pnl = pnl
        dd = (peak_pnl - pnl) / denominator
        if dd > max_dd:
            max_dd = dd
    return max_dd


def avg_edge() -> float:
    rows = _settled_trades()
    if not rows:
        return 0.0
    # Edge approximated from entry price distance to 0.5
    edges = [abs(r.entry_price - 0.5) for r in rows]
    return sum(edges) / len(edges)


def win_rate_by_side() -> dict[str, float]:
    with SessionLocal() as s:
        rows = s.query(TradeRow).filter(TradeRow.outcome.in_(["WIN", "LOSS"])).all()
    result = {}
    for side in ["YES", "NO"]:
        side_rows = [r for r in rows if r.side == side]
        if side_rows:
            result[side] = sum(1 for r in side_rows if r.outcome == "WIN") / len(side_rows)
        else:
            result[side] = 0.0
    return result


def exposure_usdc() -> float:
    """Sum of open (PLACED, PENDING) bet sizes."""
    with SessionLocal() as s:
        rows = s.query(TradeRow).filter(TradeRow.outcome == "PENDING").all()
    return sum(r.bet_usdc for r in rows)


def recent_trades(n: int = 10) -> list[dict]:
    with SessionLocal() as s:
        rows = (
            s.query(TradeRow)
            .order_by(TradeRow.placed_at.desc())
            .limit(n)
            .all()
        )
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "question": (r.question or "")[:60],
            "side": r.side,
            "entry": r.entry_price,
            "bet": r.bet_usdc,
            "outcome": r.outcome,
            "pnl": r.pnl_usdc,
            "placed": r.placed_at.strftime("%m-%d %H:%M") if r.placed_at else "",
        })
    return out


def summary() -> dict:
    rows_all = _all_trades()
    rows_settled = _settled_trades()
    wins = sum(1 for r in rows_settled if r.outcome == "WIN")
    losses = sum(1 for r in rows_settled if r.outcome == "LOSS")
    return {
        "total_trades": len(rows_all),
        "settled": len(rows_settled),
        "pending": sum(1 for r in rows_all if r.outcome == "PENDING"),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(rows_settled) if rows_settled else 0.0,
        "total_pnl": total_pnl(),
        "sharpe": sharpe_ratio(),
        "max_drawdown": max_drawdown(),
        "exposure_usdc": exposure_usdc(),
        "avg_edge": avg_edge(),
    }
