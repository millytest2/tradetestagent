"""
Step 2 — Research Agent (Parallel)
────────────────────────────────────
Runs three sub-agents concurrently for each flagged stock:
  • Reddit agent   — searches stock subreddits
  • RSS agent      — scans financial news feeds
  • Trends agent   — Google Trends search interest

VADER sentiment is computed on all collected posts.
"""

from __future__ import annotations

import asyncio
import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from core.models import (
    FlaggedStock,
    ResearchReport,
    SentimentResult,
    SocialPost,
)
from integrations.reddit import search_reddit
from integrations.rss_feed import search_rss
from integrations.trends import get_trend_score

logger = logging.getLogger(__name__)

_vader = SentimentIntensityAnalyzer()


def _compute_sentiment(posts: list[SocialPost]) -> SentimentResult:
    if not posts:
        return SentimentResult(
            compound=0.0, positive=0.0, negative=0.0, neutral=1.0,
            post_count=0, avg_engagement=0.0,
        )
    compounds, pos, neg, neu, eng = [], [], [], [], []
    for p in posts:
        s = _vader.polarity_scores(p.text)
        compounds.append(s["compound"])
        pos.append(s["pos"])
        neg.append(s["neg"])
        neu.append(s["neu"])
        eng.append(p.likes + p.retweets + p.score)

    n = len(posts)
    return SentimentResult(
        compound=sum(compounds) / n,
        positive=sum(pos) / n,
        negative=sum(neg) / n,
        neutral=sum(neu) / n,
        post_count=n,
        avg_engagement=sum(eng) / n,
    )


def _top_claims(posts: list[SocialPost], n: int = 10) -> list[str]:
    sorted_posts = sorted(posts, key=lambda p: p.likes + p.retweets + p.score, reverse=True)
    return [p.text[:200] for p in sorted_posts[:n]]


def _build_narrative(sentiment: SentimentResult, stock_ticker: str, price_change: float) -> str:
    direction = (
        "bullish" if sentiment.compound > 0.05
        else "bearish" if sentiment.compound < -0.05
        else "neutral"
    )
    move = f"{price_change:+.1%}" if price_change else "flat"
    return (
        f"Social sentiment for ${stock_ticker} is {direction} "
        f"(compound={sentiment.compound:+.3f}, {sentiment.post_count} posts). "
        f"Stock moved {move} today."
    )


async def research_stock(flagged: FlaggedStock) -> ResearchReport:
    """Research a single flagged stock using parallel data sources."""
    ticker = flagged.stock.ticker
    company = flagged.stock.company_name

    logger.info("Researching $%s (%s)...", ticker, company or "unknown")

    reddit_task = asyncio.create_task(search_reddit(ticker, company))
    rss_task = asyncio.create_task(search_rss(ticker, company))
    trends_task = asyncio.create_task(get_trend_score(ticker, company))

    reddit_posts, rss_posts, trend_score = await asyncio.gather(
        reddit_task, rss_task, trends_task,
        return_exceptions=True,
    )

    all_posts: list[SocialPost] = []
    for result in [reddit_posts, rss_posts]:
        if isinstance(result, list):
            all_posts.extend(result)
        elif isinstance(result, Exception):
            logger.debug("Source failed for $%s: %s", ticker, result)

    if isinstance(trend_score, Exception):
        trend_score = 50.0

    sentiment = _compute_sentiment(all_posts)
    narrative = _build_narrative(sentiment, ticker, flagged.stock.price_change_1d)
    claims = _top_claims(all_posts)

    logger.info(
        "$%s: sentiment=%.3f, trends=%.0f, posts=%d",
        ticker, sentiment.compound, trend_score, len(all_posts),
    )

    return ResearchReport(
        ticker=ticker,
        company_name=company,
        posts=all_posts,
        sentiment=sentiment,
        narrative_summary=narrative,
        key_claims=claims,
        trend_score=float(trend_score),
    )


async def research_stocks_parallel(
    flagged_stocks: list[FlaggedStock],
    max_concurrent: int = 3,
) -> list[tuple[FlaggedStock, ResearchReport]]:
    """Research multiple flagged stocks with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(f: FlaggedStock) -> tuple[FlaggedStock, ResearchReport]:
        async with semaphore:
            report = await research_stock(f)
            return f, report

    results = await asyncio.gather(
        *[_bounded(f) for f in flagged_stocks],
        return_exceptions=True,
    )

    out = []
    for r in results:
        if isinstance(r, tuple):
            out.append(r)
        else:
            logger.error("Research failed for a stock: %s", r)
    return out
