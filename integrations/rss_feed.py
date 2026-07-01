"""RSS feed scraper for news relevant to prediction markets."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.models import SocialPost

logger = logging.getLogger(__name__)

# High-signal news feeds for prediction markets, spanning many domains so a
# market on any topic (politics, markets, sports, tech, science, world) has
# relevant coverage. Feeds that 404/403 are skipped silently at fetch time, so
# a broad list costs nothing when some are unavailable.
DEFAULT_FEEDS = [
    # General / world
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.apnews.com/rss/apf-topnews",
    "https://www.theguardian.com/world/rss",
    "https://feeds.feedburner.com/TheAtlanticWire",
    "https://feeds.npr.org/1001/rss.xml",
    # Politics
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "https://feeds.washingtonpost.com/rss/politics",
    # Markets / economics
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    # Sports
    "https://www.espn.com/espn/rss/news",
    "https://www.espn.com/espn/rss/soccer/news",
    # Tech / science
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.sciencedaily.com/rss/top/science.xml",
]

# In-memory cache: url -> (fetched_at_epoch, entries)
_FEED_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 300  # 5 minutes — RSS feeds don't update that fast


def _parse_rss_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse an RSS date string into a UTC datetime."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str).astimezone(timezone.utc)
    except Exception:
        return None


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
async def _fetch_feed(url: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch and parse a single RSS feed, with 5-minute in-memory cache."""
    now = time.monotonic()
    cached = _FEED_CACHE.get(url)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    try:
        import feedparser
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        entries = feed.entries
        _FEED_CACHE[url] = (now, entries)
        return entries
    except ImportError:
        logger.warning("feedparser not installed — using raw HTTP fallback")
        return []
    except Exception as e:
        logger.debug("RSS feed %s failed: %s", url, e)
        return []


def _entry_matches(entry: dict, question: str) -> bool:
    """Return True if the feed entry is relevant to the market question."""
    stopwords = {"will", "be", "the", "a", "an", "is", "are", "was", "in", "on",
                 "at", "to", "of", "for", "by", "with"}
    key_words = {
        w.strip("?.,!").lower()
        for w in question.split()
        if len(w) > 3 and w.lower() not in stopwords
    }

    title = (entry.get("title") or "").lower()
    summary = (entry.get("summary") or "").lower()
    text = f"{title} {summary}"

    # Match if at least 2 key words appear in the entry
    matches = sum(1 for w in key_words if w in text)
    return matches >= 2


def _extract_query_terms(question: str, max_terms: int = 6) -> str:
    """Pull the most meaningful keywords out of a market question."""
    stopwords = {"will", "be", "the", "a", "an", "is", "are", "was", "were", "in",
                 "on", "at", "to", "of", "for", "by", "with", "this", "that",
                 "which", "who", "what", "when", "before", "after", "than"}
    words = [w.strip("?.,!'\"").lower() for w in question.split()]
    key = [w for w in words if w and w not in stopwords and len(w) > 2]
    return " ".join(key[:max_terms])


async def _fetch_google_news(question: str, client: httpx.AsyncClient,
                             max_items: int = 10) -> list[SocialPost]:
    """
    Query-targeted news search via Google News RSS. Runs several query angles
    per market (focused multi-term + a broadened few-term query) and merges the
    results, so coverage is wider than a single keyword string. Google endpoints
    aren't IP-blocked on data-center runners the way Reddit is.
    """
    from urllib.parse import quote_plus

    focused = _extract_query_terms(question, max_terms=6)
    broad = _extract_query_terms(question, max_terms=3)
    queries = [q for q in dict.fromkeys([focused, broad]) if q]   # dedupe, keep order
    if not queries:
        return []

    posts: list[SocialPost] = []
    seen_urls: set[str] = set()
    try:
        import feedparser
        for terms in queries:
            url = (f"https://news.google.com/rss/search?q={quote_plus(terms)}"
                   f"&hl=en-US&gl=US&ceid=US:en")
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
            except Exception as e:
                logger.debug("Google News query '%s' failed: %s", terms, e)
                continue
            for entry in feed.entries[:max_items]:
                link = entry.get("link", "")
                if link and link in seen_urls:
                    continue
                seen_urls.add(link)
                title = entry.get("title", "")
                summary = (entry.get("summary", "") or "")[:500]
                text = f"{title}. {summary}".strip(". ")
                posts.append(SocialPost(
                    source="rss",
                    author=entry.get("source", {}).get("title", "news.google.com")
                        if isinstance(entry.get("source"), dict) else "news.google.com",
                    text=text,
                    url=link,
                    published_at=_parse_rss_date(entry.get("published")),
                ))
        logger.info("Google News: %d articles across %d queries for '%s'",
                    len(posts), len(queries), focused)
    except ImportError:
        logger.debug("feedparser not installed — skipping Google News")
    return posts


async def search_rss(
    question: str,
    feeds: Optional[list[str]] = None,
    max_per_feed: int = 10,
) -> list[SocialPost]:
    """Search RSS feeds for articles relevant to a market question.

    Combines a query-targeted Google News search (most relevant) with a scan of
    high-signal homepage feeds (breaking/background context).
    """

    target_feeds = feeds or DEFAULT_FEEDS
    posts: list[SocialPost] = []

    async with httpx.AsyncClient(timeout=15) as client:
        # Query-targeted search first — highest relevance per market.
        posts.extend(await _fetch_google_news(question, client, max_items=max_per_feed))
        for feed_url in target_feeds:
            try:
                entries = await _fetch_feed(feed_url, client)
                count = 0
                for entry in entries:
                    if count >= max_per_feed:
                        break
                    if not _entry_matches(entry, question):
                        continue

                    title = entry.get("title", "")
                    summary = entry.get("summary", "")[:500]
                    text = f"{title}. {summary}".strip(". ")
                    link = entry.get("link", "")
                    published = _parse_rss_date(entry.get("published"))

                    posts.append(SocialPost(
                        source="rss",
                        author=entry.get("author", feed_url.split("/")[2]),
                        text=text,
                        url=link,
                        published_at=published,
                    ))
                    count += 1
            except Exception as e:
                logger.debug("Feed %s error: %s", feed_url, e)
                continue

    logger.info("RSS: found %d articles for '%s'", len(posts), question[:60])
    return posts
