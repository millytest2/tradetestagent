"""RSS feed scraper for stock news."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

from core.models import SocialPost

logger = logging.getLogger(__name__)

FINANCIAL_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.feedburner.com/businessinsider",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
]

_FEED_CACHE: dict[str, tuple[float, list]] = {}
_CACHE_TTL = 300

_STOPWORDS = {"will", "be", "the", "a", "an", "is", "are", "was", "were", "in",
              "on", "at", "to", "of", "for", "by", "with"}


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str).astimezone(timezone.utc)
    except Exception:
        return None


def _entry_matches(entry: dict, ticker: str, company_name: str) -> bool:
    title = (entry.get("title") or "").lower()
    summary = (entry.get("summary") or "").lower()
    text = f"{title} {summary}"

    if ticker.lower() in text:
        return True

    if company_name:
        words = [w.lower() for w in company_name.split() if len(w) > 3]
        matches = sum(1 for w in words[:3] if w in text)
        if matches >= 2:
            return True

    return False


async def search_rss(
    ticker: str,
    company_name: str = "",
    feeds: Optional[list[str]] = None,
    max_per_feed: int = 5,
) -> list[SocialPost]:
    """Search financial RSS feeds for news about a stock."""
    target_feeds = feeds or FINANCIAL_FEEDS
    posts: list[SocialPost] = []

    async with httpx.AsyncClient(timeout=15) as client:
        for feed_url in target_feeds:
            now = time.monotonic()
            cached = _FEED_CACHE.get(feed_url)
            if cached and (now - cached[0]) < _CACHE_TTL:
                entries = cached[1]
            else:
                try:
                    import feedparser
                    resp = await client.get(feed_url, follow_redirects=True)
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.text)
                    entries = feed.entries
                    _FEED_CACHE[feed_url] = (now, entries)
                except Exception as e:
                    logger.debug("RSS feed %s failed: %s", feed_url, e)
                    continue

            count = 0
            for entry in entries:
                if count >= max_per_feed:
                    break
                if not _entry_matches(entry, ticker, company_name):
                    continue
                title = entry.get("title", "")
                summary = entry.get("summary", "")[:400]
                text = f"{title}. {summary}".strip(". ")
                posts.append(SocialPost(
                    source="rss",
                    author=entry.get("author", feed_url.split("/")[2]),
                    text=text,
                    url=entry.get("link", ""),
                    published_at=_parse_date(entry.get("published")),
                ))
                count += 1

    logger.info("RSS: %d articles for $%s", len(posts), ticker)
    return posts
