"""
Empirical learning from settled trades — works WITHOUT the LLM.

The qualitative loop (5-agent postmortems → written lessons) is dead whenever
Anthropic credits are exhausted: run_postmortem() returns immediately and saves
nothing, and the lessons it would have written are only ever injected into an
LLM prompt, so they could not steer a free-mode decision anyway.

This module closes that gap with statistics instead of prose. It reads settled
trades, groups them by properties known BEFORE entry (entry-price band, market
family), and reports which buckets have actually lost money. The prediction
agent consults it directly, so the bot stops repeating its own losing patterns
even with no API access at all.

Deliberately conservative: a bucket must clear a minimum sample and lose clearly
before it is avoided. With ~30 settled trades the risk is overfitting noise, so
the bar to veto is high and every verdict states its own sample size.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Entry-price bands. A favorite bought at 0.70 and one at 0.89 are different
# propositions: different hit rates and very different payouts.
_PRICE_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.68, "sub-favorite"),
    (0.68, 0.78, "favorite-low"),
    (0.78, 0.88, "favorite-mid"),
    (0.88, 1.01, "favorite-high"),
)

# Market family from the PM-US slug prefix, which is a stable venue convention.
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("aachc-", "sports"), ("aqc-", "sports"), ("tec-", "sports"),
    ("ewc-", "politics-general"), ("enwc-", "politics-primary"),
    ("gdpc-", "macro"), ("nfpc-", "macro"),
)


def price_band(entry_price: float | None) -> str:
    p = entry_price or 0.0
    for lo, hi, name in _PRICE_BANDS:
        if lo <= p < hi:
            return name
    return "unknown"


def market_family(slug: str | None) -> str:
    s = (slug or "").lower()
    for prefix, name in _FAMILY_PREFIXES:
        if s.startswith(prefix):
            return name
    m = re.match(r"^([a-z]+)-", s)
    return m.group(1) if m else "unknown"


@dataclass
class BucketStats:
    key: str
    kind: str            # "price_band" | "family"
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    samples: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return (self.wins / self.n) if self.n else 0.0

    def __str__(self) -> str:
        return (f"{self.kind}:{self.key} n={self.n} "
                f"win={self.win_rate:.0%} pnl=${self.pnl:+.2f}")


def compute_bucket_stats() -> dict[str, BucketStats]:
    """Group every settled trade into price-band and market-family buckets."""
    from core.database import SessionLocal, TradeRow

    out: dict[str, BucketStats] = {}

    def bump(kind: str, key: str, won: bool, pnl: float, tid: int) -> None:
        k = f"{kind}:{key}"
        b = out.setdefault(k, BucketStats(key=key, kind=kind))
        if won:
            b.wins += 1
        else:
            b.losses += 1
        b.pnl += pnl or 0.0
        b.samples.append(tid)

    try:
        with SessionLocal() as s:
            rows = (
                s.query(TradeRow)
                .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
                .all()
            )
            settled = [
                (r.id, r.market_id, r.entry_price, r.outcome, r.pnl_usdc)
                for r in rows
            ]
    except Exception as e:
        logger.debug("Empirical stats query failed: %s", e)
        return {}

    # Families we no longer trade at all must not poison the cross-family
    # price-band stats: sports losses would otherwise drag a price band below
    # the veto threshold and block politics trades in that same band, even
    # though no further sports trade can ever be placed. Their family bucket is
    # still recorded, so the ban remains visible and auditable.
    from config import settings
    excluded = {"sports"} if settings.exclude_sports_markets else set()

    for tid, slug, entry, outcome, pnl in settled:
        won = outcome == "WIN"
        fam = market_family(slug)
        bump("family", fam, won, pnl, tid)
        if fam not in excluded:
            bump("price_band", price_band(entry), won, pnl, tid)
    return out


def losing_buckets(min_n: int = 5, max_win_rate: float = 0.34) -> dict[str, BucketStats]:
    """Buckets with enough history to judge that are demonstrably losing.

    Both conditions must hold: at least `min_n` settled trades AND a win rate at
    or below `max_win_rate` AND negative total P&L. Requiring all three keeps a
    short unlucky streak from banning a whole category.
    """
    bad = {}
    for k, b in compute_bucket_stats().items():
        if b.n >= min_n and b.win_rate <= max_win_rate and b.pnl < 0:
            bad[k] = b
    return bad


def entry_verdict(entry_price: float, slug: str,
                  min_n: int = 5, max_win_rate: float = 0.34) -> tuple[bool, str]:
    """(allowed, reason) for a proposed entry, judged on our own settled record.

    Returns allowed=False only when the exact bucket this trade falls into has
    a real, sizeable losing history. The reason always carries the sample size
    so a veto can be audited rather than taken on faith.
    """
    bad = losing_buckets(min_n=min_n, max_win_rate=max_win_rate)
    if not bad:
        return True, ""
    for kind, key in (("price_band", price_band(entry_price)),
                      ("family", market_family(slug))):
        b = bad.get(f"{kind}:{key}")
        if b is not None:
            return False, (
                f"own record: {kind} '{key}' is {b.wins}W/{b.losses}L "
                f"({b.win_rate:.0%}) for ${b.pnl:+.2f} over {b.n} settled trades"
            )
    return True, ""


def summarize(min_n: int = 3) -> list[str]:
    """Human-readable lines for the logs and the daily email."""
    stats = compute_bucket_stats()
    lines = []
    for k in sorted(stats, key=lambda x: stats[x].pnl):
        b = stats[k]
        if b.n >= min_n:
            lines.append(str(b))
    return lines
