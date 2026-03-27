"""Reddit scraper using PRAW."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config import settings
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


def _get_reddit_client():
    """Return a PRAW Reddit client or None if creds are missing."""
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        return None
    try:
        import praw
        return praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
    except ImportError:
        logger.warning("praw not installed — skipping Reddit")
        return None
    except Exception as e:
        logger.error("Reddit client init failed: %s", e)
        return None


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
    """Search Reddit for posts relevant to a market question."""

    reddit = _get_reddit_client()
    if reddit is None:
        return []

    search_query = _extract_search_terms(question)
    target_subs = subreddits or RELEVANT_SUBREDDITS[:5]
    subreddit_str = "+".join(target_subs)

    posts: list[SocialPost] = []
    try:
        subreddit = reddit.subreddit(subreddit_str)
        results = list(subreddit.search(
            query=search_query,
            sort="relevance",
            time_filter="week",
            limit=max_posts,
        ))

        for submission in results:
            try:
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                # Combine title + selftext for richer content
                text = f"{submission.title}. {submission.selftext[:500]}".strip(". ")

                posts.append(SocialPost(
                    source="reddit",
                    author=str(submission.author) if submission.author else "[deleted]",
                    text=text,
                    url=f"https://reddit.com{submission.permalink}",
                    published_at=created,
                    score=submission.score,
                ))
            except Exception:
                continue

        logger.info("Reddit: found %d posts for '%s'", len(posts), question[:60])
        return posts

    except Exception as e:
        logger.error("Reddit search failed: %s", e)
        return []
