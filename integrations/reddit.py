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


_TOKEN_CACHE: dict[str, object] = {}


async def _get_oauth_token() -> Optional[str]:
    """
    Fetch an application-only OAuth token from Reddit. This routes subsequent
    searches through oauth.reddit.com, which — unlike the public www JSON
    endpoint — accepts requests from data-center IPs (GitHub runners). Requires
    REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET (free "script" app at
    reddit.com/prefs/apps). Returns None if creds are absent or auth fails.
    """
    from config import settings
    cid, secret = settings.reddit_client_id, settings.reddit_client_secret
    if not cid or not secret:
        return None
    cached = _TOKEN_CACHE.get("token")
    if cached:
        return cached  # tokens last ~1h; cache for the process lifetime
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://www.reddit.com/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=(cid, secret),
                headers={"User-Agent": settings.reddit_user_agent},
            )
            resp.raise_for_status()
            token = resp.json().get("access_token")
            if token:
                _TOKEN_CACHE["token"] = token
            return token
    except Exception as e:
        logger.debug("Reddit OAuth token fetch failed: %s", e)
        return None


async def search_reddit(
    question: str,
    max_posts: int = 20,
    subreddits: Optional[list[str]] = None,
) -> list[SocialPost]:
    """Search Reddit for posts relevant to a market question.

    Prefers the authenticated OAuth API (works from data-center IPs) and falls
    back to the public JSON endpoint. Reddit routinely 403s unauthenticated
    data-center requests, so without creds this stays quiet and RSS/Google News
    carry the news signal.
    """
    search_query = _extract_search_terms(question)
    if not search_query:
        return []

    target_subs = subreddits or RELEVANT_SUBREDDITS[:10]
    subreddit_str = "+".join(target_subs)

    params = {
        "q": search_query,
        "sort": "relevance",
        "t": "week",
        "limit": max_posts,
        "restrict_sr": "1",
    }

    posts: list[SocialPost] = []

    # Preferred path: authenticated OAuth (bypasses the data-center 403).
    token = await _get_oauth_token()
    if token:
        from config import settings
        oauth_url = f"https://oauth.reddit.com/r/{subreddit_str}/search"
        oauth_headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": settings.reddit_user_agent,
        }
        try:
            async with httpx.AsyncClient(headers=oauth_headers, timeout=15.0) as client:
                resp = await client.get(oauth_url, params=params)
                resp.raise_for_status()
                data = resp.json()
            for item in data.get("data", {}).get("children", []):
                d = item.get("data", {})
                try:
                    created = datetime.fromtimestamp(
                        float(d.get("created_utc", 0)), tz=timezone.utc)
                    title = d.get("title", "")
                    body = d.get("selftext", "")[:500]
                    text = f"{title}. {body}".strip(". ") if body else title
                    posts.append(SocialPost(
                        source="reddit",
                        author=d.get("author", "[deleted]"),
                        text=text,
                        url=f"https://reddit.com{d.get('permalink', '')}",
                        published_at=created,
                        score=int(d.get("score", 0)),
                    ))
                except Exception:
                    continue
            logger.info("Reddit (OAuth): found %d posts for '%s'", len(posts), question[:60])
            return posts
        except Exception as e:
            logger.debug("Reddit OAuth search failed (%s) — trying public JSON", e)

    # Fallback path: public JSON endpoint (often 403s from data centers).
    url = f"https://www.reddit.com/r/{subreddit_str}/search.json"
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
