"""Twitter/X scraper using the official API v2 (bearer token) or a lightweight fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings
from core.models import SocialPost

logger = logging.getLogger(__name__)

TWITTER_API_BASE = "https://api.twitter.com/2"


def _build_query(question: str, max_chars: int = 200) -> str:
    """Build a Twitter search query from a market question."""
    # Extract key terms (first 5 words + strip common words)
    stopwords = {"a", "an", "the", "will", "be", "in", "of", "on", "to", "by", "for", "at"}
    words = [w.strip("?.,!") for w in question.split() if w.lower() not in stopwords]
    key_terms = " ".join(words[:6])
    query = f'({key_terms}) lang:en -is:retweet'
    return query[:max_chars]


async def search_twitter(
    question: str,
    max_results: int = 20,
) -> list[SocialPost]:
    """Search recent tweets relevant to a market question."""

    if not settings.twitter_bearer_token:
        logger.debug("No Twitter bearer token — skipping Twitter search")
        return []

    query = _build_query(question)
    headers = {"Authorization": f"Bearer {settings.twitter_bearer_token}"}
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,public_metrics,author_id",
        "expansions": "author_id",
        "user.fields": "username",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{TWITTER_API_BASE}/tweets/search/recent",
                headers=headers,
                params=params,
            )
            if resp.status_code == 429:
                logger.warning("Twitter rate limit hit — skipping")
                return []
            resp.raise_for_status()
            data = resp.json()

        tweets = data.get("data", [])
        users = {
            u["id"]: u["username"]
            for u in data.get("includes", {}).get("users", [])
        }

        posts: list[SocialPost] = []
        for tweet in tweets:
            metrics = tweet.get("public_metrics", {})
            created = tweet.get("created_at")
            published_at = None
            if created:
                try:
                    published_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except Exception:
                    pass

            posts.append(SocialPost(
                source="twitter",
                author=users.get(tweet.get("author_id", ""), ""),
                text=tweet.get("text", ""),
                url=f"https://twitter.com/i/web/status/{tweet['id']}",
                published_at=published_at,
                likes=metrics.get("like_count", 0),
                retweets=metrics.get("retweet_count", 0),
            ))

        logger.info("Twitter: found %d tweets for '%s'", len(posts), question[:60])
        return posts

    except httpx.HTTPStatusError as e:
        logger.warning("Twitter API error %d: %s", e.response.status_code, e)
        return []
    except Exception as e:
        logger.error("Twitter search failed: %s", e)
        return []
