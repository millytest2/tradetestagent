"""Reddit scraper using public JSON API — no credentials required."""

from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx

from core.models import SocialPost

logger = logging.getLogger(__name__)

RELEVANT_SUBREDDITS = [
    "PredictionMarkets",
    "Polymarket",
    "worldnews",
    "news",
    "politics",
    "Economics",
    "stocks",
    "investing",
    "sports",
    "soccer",
    "nba",
    "nfl",
    "baseball",
    "entertainment",
    "boxoffice",
]

# Reddit blocks generic/bot User-Agents with 403. A real browser UA from a
# residential IP (the user's laptop) gets through where a bot UA does not.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_search_terms(question: str) -> str:
    """Extract the most relevant search terms from a market question."""
    stopwords = {"will", "be", "the", "a", "an", "is", "are", "was", "were", "in",
                 "on", "at", "to", "of", "for", "by", "with", "this", "that", "which"}
    words = [w.strip("?.,!").lower() for w in question.split()]
    key_words = [w for w in words if w and w not in stopwords and len(w) > 2]
    return " ".join(key_words[:5])


async def search_reddit(
    question: str,
    max_posts: int = 20,
    subreddits: Optional[list[str]] = None,
) -> list[SocialPost]:
    """Search Reddit for posts relevant to a market question.

    Uses Reddit's public JSON search endpoint — no API key or app needed.
    """
    search_query = _extract_search_terms(question)
    if not search_query:
        return []

    target_subs = subreddits or RELEVANT_SUBREDDITS[:5]
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

        logger.info("Reddit: found %d posts for '%s'", len(posts), question[:60])

    except httpx.HTTPStatusError as e:
        # Reddit routinely 403s automated/data-center requests — expected, not
        # an error. RSS feeds cover the news signal. Keep it quiet.
        logger.debug("Reddit unavailable (%s) — relying on RSS", e.response.status_code)
    except Exception as e:
        logger.debug("Reddit search skipped: %s", e)

    return posts
