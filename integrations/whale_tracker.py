"""
Public whale-wallet tracking via Polymarket's on-chain Data API.

Polymarket settles on Polygon, so every position is PUBLIC. This module READS
(never trades, never proxies) how the biggest wallets are positioned on the
international market that matches a given question, and returns that as an extra
signal for the equivalent Polymarket US market (overlap markets like the World
Cup, elections, macro). 100% public data — no geoblock circumvention.

Public endpoints (no API key):
  Gamma: https://gamma-api.polymarket.com/markets   — find market + token ids
  Data:  https://data-api.polymarket.com/holders    — top holders per market

If a match isn't found or the API shape differs, it returns 0.0 (neutral) so it
can only add signal, never break the pipeline.
"""

from __future__ import annotations

import json as _json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"

_STOP = {
    "will", "the", "a", "an", "be", "to", "of", "in", "on", "by", "vs",
    "win", "yes", "no", "and", "or", "at", "for", "is", "are", "this",
}


def _keywords(q: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (q or "").lower())
            if len(w) > 2 and w not in _STOP}


async def _find_intl_market(question: str):
    """Find the international market best matching `question`.
    Returns (condition_id, intl_question) or None."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"{GAMMA}/markets",
                params={"closed": "false", "active": "true", "limit": 250,
                        "order": "volume24hr", "ascending": "false"},
            )
            r.raise_for_status()
            markets = r.json()
    except Exception as e:
        logger.debug("Whale: Gamma fetch failed: %s", e)
        return None

    qk = _keywords(question)
    if not qk or not isinstance(markets, list):
        return None

    best, best_score = None, 0.0
    for m in markets:
        mk = _keywords(m.get("question") or m.get("title") or "")
        if not mk:
            continue
        overlap = len(qk & mk) / max(1, len(qk))
        if overlap > best_score:
            best_score, best = overlap, m

    # Require a strong match so we don't attribute the wrong market's whales
    if best and best_score >= 0.6:
        cond = best.get("conditionId") or best.get("id") or ""
        return (cond, best.get("question", "")) if cond else None
    return None


async def get_wallet_whale_signal(question: str, threshold_usd: float = 5000.0) -> float:
    """
    Return -1.0..+1.0 = net lean of large wallets (≥$5k) on the matching
    international market. +1 = whales heavily on YES, -1 = heavily on NO,
    0.0 = no match / no whales / unreadable.
    """
    found = await _find_intl_market(question)
    if not found:
        return 0.0
    cond, intl_q = found

    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{DATA}/holders", params={"market": cond, "limit": 100})
            if r.status_code != 200:
                return 0.0
            data = r.json()
    except Exception as e:
        logger.debug("Whale: Data API failed: %s", e)
        return 0.0

    if not _RAW_LOGGED[0]:
        _RAW_LOGGED[0] = True
        try:
            logger.info("Raw holders payload: %s", str(data).replace("\n", " ")[:500])
        except Exception:
            pass

    # The /holders response is a LIST of {token, holders:[{amount, outcomeIndex}]}
    # groups (not a flat list). Flatten to every holder dict, whatever the shape.
    def _flatten(obj) -> list:
        out = []
        if isinstance(obj, dict):
            if any(k in obj for k in ("amount", "shares", "size", "value")) and \
               any(k in obj for k in ("outcomeIndex", "outcome", "outcome_index")):
                out.append(obj)
            for v in obj.values():
                out.extend(_flatten(v))
        elif isinstance(obj, list):
            for v in obj:
                out.extend(_flatten(v))
        return out

    yes_sh = no_sh = 0.0   # these are share counts (the API returns shares, not USD)
    for h in _flatten(data):
        amt = 0.0
        for k in ("amount", "shares", "size", "value"):
            try:
                amt = float(h.get(k)); break
            except (TypeError, ValueError):
                continue
        if amt <= 0:
            continue
        idx = h.get("outcomeIndex", h.get("outcome", h.get("outcome_index")))
        if idx in (0, "0", "Yes", "YES", "yes"):
            yes_sh += amt
        elif idx in (1, "1", "No", "NO", "no"):
            no_sh += amt

    total = yes_sh + no_sh
    if total <= 0:
        return 0.0
    lean = (yes_sh - no_sh) / total
    logger.info(
        "Wallet-whale lean %+.2f (yes=%.0f no=%.0f shares) on intl '%s'",
        lean, yes_sh, no_sh, intl_q[:45],
    )
    return float(max(-1.0, min(1.0, lean)))


_RAW_LOGGED = [False]   # one-time raw-shape diagnostic
