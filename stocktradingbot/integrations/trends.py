"""Google Trends integration via pytrends — no API key required."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def get_trend_score(ticker: str, company_name: str = "") -> float:
    """
    Return Google Trends interest score (0–100) for a stock ticker.

    Uses the ticker symbol and optionally the company name as search terms.
    Returns 50.0 on any failure so it never blocks a trade signal.
    Requires: pip install pytrends
    """
    kw = ticker
    if company_name:
        kw = f"{ticker} {company_name.split()[0]}"
    kw = kw[:100]

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
        logger.info("Trends: %.1f/100 for $%s", score, ticker)
        return score

    except ImportError:
        return 50.0
    except asyncio.TimeoutError:
        return 50.0
    except Exception as e:
        logger.debug("Google Trends failed for $%s: %s", ticker, e)
        return 50.0
