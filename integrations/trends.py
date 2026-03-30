"""Google Trends integration via pytrends — no API key required."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "will", "be", "the", "a", "an", "is", "are", "was", "were",
    "by", "in", "to", "of", "for", "with", "from", "that", "this",
    "which", "have", "has", "not", "its", "at", "on", "or", "and",
}


def _keywords(question: str) -> str:
    words = [w.strip("?.,!()").lower() for w in question.split()]
    kept = [w for w in words if w and w not in _STOPWORDS and len(w) > 3]
    return " ".join(kept[:5])[:100]


async def get_trend_score(question: str) -> float:
    """
    Return Google Trends interest score (0–100) for the past 7 days.

    - 100 = peak search interest
    - 50  = average (neutral baseline)
    - 0   = essentially no interest

    Returns 50.0 on any failure so it never blocks a trade signal.
    Requires: pip install pytrends
    """
    kw = _keywords(question)
    if not kw:
        return 50.0

    try:
        from pytrends.request import TrendReq

        def _fetch() -> float:
            pt = TrendReq(
                hl="en-US",
                tz=0,
                timeout=(10, 25),
                retries=1,
                backoff_factor=0.5,
            )
            pt.build_payload([kw], timeframe="now 7-d")
            df = pt.interest_over_time()
            if df.empty:
                return 50.0
            data_cols = [c for c in df.columns if c != "isPartial"]
            if not data_cols:
                return 50.0
            return float(df[data_cols[0]].mean())

        loop = asyncio.get_event_loop()
        score = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch),
            timeout=30.0,
        )
        logger.info("Trends: %.1f/100 for '%s'", score, kw[:50])
        return score

    except ImportError:
        logger.debug("pytrends not installed — skipping Google Trends")
        return 50.0
    except asyncio.TimeoutError:
        logger.debug("Google Trends timed out for '%s'", kw[:50])
        return 50.0
    except Exception as e:
        logger.debug("Google Trends failed for '%s': %s", kw[:50], e)
        return 50.0
