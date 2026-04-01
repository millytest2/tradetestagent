"""Reddit scraper using public JSON API — no credentials required."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from core.models import SocialPost

logger = logging.getLogger(__name__)

STOCK_SUBREDDITS = [
    "stocks",
    "investing",
    "wallstreetbets",
    "StockMarket",
    "SecurityAnalysis",
    "ValueInvesting",
    "finance",
    "options",
    "dividends",
    "pennystocks",
]

_HEADERS = {
    "User-Agent": "stock-trading-bot/1.0 (research tool)",
    "Accept": "application/json",
}

_STOPWORDS = {"will", "be", "the", "a", "an", "is", "are", "was", "were", "in",
              "on", "at", "to", "of", "for", "by", "with", "this", "that", "which"}


def _extract_search_terms(ticker: str, company_name: str = "") -> str:
    terms = [ticker]
    if company_name:
        words = [w.strip(".,").lower() for w in company_name.split()
                 if w.lower() not in _STOPWORDS and len(w) > 2]
        terms.extend(words[:3])
    return " ".join(terms[:4])


async def search_reddit(
    ticker: str,
    company_name: str = "",
    max_posts: int = 20,
    subreddits: Optional[list[str]] = None,
) -> list[SocialPost]:
    """Search Reddit for posts about a stock ticker."""
    search_query = _extract_search_terms(ticker, company_name)
    if not search_query:
        return []

    target_subs = subreddits or STOCK_SUBREDDITS[:5]
    subreddit_str = "+".join(target_subs)
    url = f"https://www.reddit.com/r/{subreddit_str}/search.json"
    params = {
        "q": search_query,
        "sort": "relevance",
        "t": "week",
        "limit": max_posts,
        "restrict_sr": "1",
    }

    posts: list[SocialPost] = []
    try:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 429:
                logger.warning("Reddit rate-limited — backing off 10s")
                await asyncio.sleep(10)
                resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        children = data.get("data", {}).get("children", [])
        for item in children:
            d = item.get("data", {})
            try:
                created = datetime.fromtimestamp(
                    float(d.get("created_utc", 0)), tz=timezone.utc
                )
                title = d.get("title", "")
                body = d.get("selftext", "")[:500]
                text = f"{title}. {body}".strip(". ") if body else title
                permalink = d.get("permalink", "")
                posts.append(SocialPost(
                    source="reddit",
                    author=d.get("author", "[deleted]"),
                    text=text,
                    url=f"https://reddit.com{permalink}",
                    published_at=created,
                    score=int(d.get("score", 0)),
                ))
            except Exception:
                continue

        logger.info("Reddit: %d posts for $%s", len(posts), ticker)

    except httpx.HTTPStatusError as e:
        logger.debug("Reddit HTTP %s for $%s", e.response.status_code, ticker)
    except Exception as e:
        logger.debug("Reddit search failed for $%s: %s", ticker, e)

    return posts
