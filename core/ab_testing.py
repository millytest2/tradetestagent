"""
A/B Split Testing Framework
────────────────────────────
Continuously tests two strategy variants against each other.
The system automatically promotes the winning variant and
retires the loser after a statistically significant sample.

Variant A = conservative baseline
Variant B = aggressive / experimental settings

Stored in SQLite alongside trade records via the `ab_variant` field in notes JSON.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Variant definitions ───────────────────────────────────────────────────────

@dataclass
class StrategyVariant:
    name: str             # "A" or "B"
    label: str            # human readable
    min_edge: float       # minimum edge to trade
    min_confidence: float
    kelly_fraction: float
    sentiment_weight: float   # 0-1 weight on sentiment vs technical
    whale_weight: float       # 0-1 weight on whale signal
    llm_weight: float         # LLM weight in ensemble (1 - this = XGBoost weight)
    contra_indicator: bool    # fade extreme sentiment against market price


VARIANT_A = StrategyVariant(
    name="A",
    label="Conservative Baseline",
    min_edge=0.06,
    min_confidence=0.62,
    kelly_fraction=0.20,
    sentiment_weight=0.50,
    whale_weight=0.30,
    llm_weight=0.60,
    contra_indicator=True,
)

VARIANT_B = StrategyVariant(
    name="B",
    label="Aggressive Edge-Seeker",
    min_edge=0.04,
    min_confidence=0.58,
    kelly_fraction=0.30,
    sentiment_weight=0.40,
    whale_weight=0.50,
    llm_weight=0.70,
    contra_indicator=False,
)

VARIANTS = {"A": VARIANT_A, "B": VARIANT_B}
MIN_SAMPLE_SIZE = 15       # minimum trades before statistical comparison
CONFIDENCE_LEVEL = 0.90    # 90% confidence to declare a winner


# ── Assignment ────────────────────────────────────────────────────────────────

_current_promoted: Optional[str] = None   # None = random split, "A" or "B" = promoted


def get_variant_for_trade() -> StrategyVariant:
    """
    Assign a variant for the next trade.

    - If a variant has been statistically promoted, always use it.
    - Otherwise, randomly split 50/50 (controlled A/B test).
    """
    global _current_promoted

    if _current_promoted:
        return VARIANTS[_current_promoted]

    # Check DB for a promoted variant
    promoted = _get_promoted_from_db()
    if promoted:
        _current_promoted = promoted
        logger.info("A/B: Using promoted variant %s", promoted)
        return VARIANTS[promoted]

    # Random 50/50 split
    chosen = random.choice(["A", "B"])
    return VARIANTS[chosen]


def _get_promoted_from_db() -> Optional[str]:
    """Check if analysis has promoted a variant."""
    try:
        from core.database import SessionLocal, SystemUpdateRow
        with SessionLocal() as s:
            row = (
                s.query(SystemUpdateRow)
                .filter(SystemUpdateRow.update_type == "ab_promotion")
                .order_by(SystemUpdateRow.created_at.desc())
                .first()
            )
        if row:
            payload = json.loads(row.payload_json or "{}")
            return payload.get("variant")
    except Exception:
        pass
    return None


# ── Analysis ──────────────────────────────────────────────────────────────────

def _wilson_lower_bound(wins: int, n: int, z: float = 1.645) -> float:
    """Wilson score lower bound for proportion — 90% CI."""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / denom


def analyze_variants() -> dict:
    """
    Compare A and B variants from the trade history.
    Returns stats dict and optionally declares a winner.
    """
    try:
        from core.database import SessionLocal, TradeRow
        with SessionLocal() as s:
            settled = (
                s.query(TradeRow)
                .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
                .all()
            )
    except Exception as e:
        logger.warning("A/B analysis failed: %s", e)
        return {}

    buckets: dict[str, dict] = {"A": {"wins": 0, "losses": 0, "pnl": 0.0},
                                  "B": {"wins": 0, "losses": 0, "pnl": 0.0}}

    for row in settled:
        try:
            notes = json.loads(row.notes or "{}")
            variant = notes.get("ab_variant", "A")
            if variant not in buckets:
                continue
            b = buckets[variant]
            if row.outcome == "WIN":
                b["wins"] += 1
            else:
                b["losses"] += 1
            b["pnl"] += row.pnl_usd or 0.0
        except Exception:
            continue

    results = {}
    for v, b in buckets.items():
        n = b["wins"] + b["losses"]
        wr = b["wins"] / n if n > 0 else 0.0
        wilson = _wilson_lower_bound(b["wins"], n)
        results[v] = {
            "variant": v,
            "label": VARIANTS[v].label,
            "trades": n,
            "wins": b["wins"],
            "losses": b["losses"],
            "win_rate": wr,
            "wilson_lb": wilson,
            "pnl": b["pnl"],
        }

    # Declare winner if both have enough samples
    n_a = results.get("A", {}).get("trades", 0)
    n_b = results.get("B", {}).get("trades", 0)

    winner = None
    if n_a >= MIN_SAMPLE_SIZE and n_b >= MIN_SAMPLE_SIZE:
        wilson_a = results["A"]["wilson_lb"]
        wilson_b = results["B"]["wilson_lb"]

        if wilson_a > wilson_b + 0.05:    # A clearly better
            winner = "A"
        elif wilson_b > wilson_a + 0.05:  # B clearly better
            winner = "B"

        if winner:
            logger.info(
                "A/B WINNER: Variant %s (Wilson LB A=%.3f B=%.3f)",
                winner, wilson_a, wilson_b,
            )
            _promote_variant(winner, results)

    return {
        "results": results,
        "winner": winner,
        "enough_data": n_a >= MIN_SAMPLE_SIZE and n_b >= MIN_SAMPLE_SIZE,
    }


def _promote_variant(variant: str, results: dict) -> None:
    """Save promotion decision to DB."""
    global _current_promoted
    _current_promoted = variant
    try:
        from core.database import save_system_update
        save_system_update(
            update_type="ab_promotion",
            description=f"Variant {variant} promoted as winner based on statistical analysis",
            payload={
                "variant": variant,
                "results": results,
            },
        )
        logger.info("Variant %s promoted permanently.", variant)
    except Exception as e:
        logger.warning("Could not save promotion: %s", e)


def get_stats() -> dict:
    """Public summary of A/B test results."""
    return analyze_variants()
