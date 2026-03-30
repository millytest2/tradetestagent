"""
Step 2 — Research Agents (Parallel)
─────────────────────────────────────
Three sub-agents run concurrently for each flagged market:
  • Twitter agent  — searches recent tweets
  • Reddit agent   — searches relevant subreddits
  • RSS agent      — scans news feeds

Results are aggregated and VADER sentiment is computed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from core.models import (
    FlaggedMarket,
    ResearchReport,
    SentimentResult,
    SocialPost,
)
from integrations.polymarket import get_whale_signal
from integrations.reddit import search_reddit
from integrations.rss_feed import search_rss
from integrations.trends import get_trend_score
from integrations.twitter import search_twitter

logger = logging.getLogger(__name__)

_vader = SentimentIntensityAnalyzer()


# ── Sentiment ─────────────────────────────────────────────────────────────────

def _compute_sentiment(posts: list[SocialPost]) -> SentimentResult:
    """Run VADER on all posts and return aggregate sentiment metrics."""
    if not posts:
        return SentimentResult(
            compound=0.0, positive=0.0, negative=0.0, neutral=1.0,
            post_count=0, avg_engagement=0.0,
        )

    compounds, pos, neg, neu = [], [], [], []
    engagement_scores = []

    for post in posts:
        scores = _vader.polarity_scores(post.text)
        compounds.append(scores["compound"])
        pos.append(scores["pos"])
        neg.append(scores["neg"])
        neu.append(scores["neu"])

        # Engagement = likes + retweets for Twitter, score for Reddit
        eng = post.likes + post.retweets + post.score
        engagement_scores.append(eng)

    n = len(posts)
    return SentimentResult(
        compound=sum(compounds) / n,
        positive=sum(pos) / n,
        negative=sum(neg) / n,
        neutral=sum(neu) / n,
        post_count=n,
        avg_engagement=sum(engagement_scores) / n,
    )


def _compare_narrative_to_odds(
    sentiment: SentimentResult,
    yes_price: float,
) -> str:
    """
    Generate a brief narrative comparison between social sentiment and
    market-implied probability.
    """
    market_pct = yes_price * 100
    sentiment_direction = (
        "bullish (YES-leaning)" if sentiment.compound > 0.05
        else "bearish (NO-leaning)" if sentiment.compound < -0.05
        else "neutral"
    )

    divergence = ""
    if sentiment.compound > 0.1 and yes_price < 0.45:
        divergence = (
            "⚠️  Sentiment is bullish but market prices YES at only "
            f"{market_pct:.0f}% — potential YES underpricing."
        )
    elif sentiment.compound < -0.1 and yes_price > 0.55:
        divergence = (
            "⚠️  Sentiment is bearish but market prices YES at "
            f"{market_pct:.0f}% — potential NO underpricing."
        )
    else:
        divergence = (
            f"Sentiment ({sentiment_direction}) appears roughly aligned "
            f"with market price of {market_pct:.0f}%."
        )

    return (
        f"Social sentiment: {sentiment_direction} "
        f"(compound={sentiment.compound:+.3f}, "
        f"posts={sentiment.post_count}). "
        f"Market YES price: {market_pct:.1f}%. "
        f"{divergence}"
    )


# ── Research orchestrator ─────────────────────────────────────────────────────

async def research_market(flagged: FlaggedMarket) -> ResearchReport:
    """
    Run Twitter, Reddit, and RSS research agents in parallel for one market.
    Returns a consolidated ResearchReport.
    """
    question = flagged.market.question
    market_id = flagged.market.condition_id

    logger.info("Research agents starting for: %s", question[:80])

    # Fire all scrapers + signals concurrently
    twitter_task = asyncio.create_task(search_twitter(question, max_results=25))
    reddit_task  = asyncio.create_task(search_reddit(question, max_posts=20))
    rss_task     = asyncio.create_task(search_rss(question, max_per_feed=8))
    trends_task  = asyncio.create_task(get_trend_score(question))
    whale_task   = asyncio.create_task(get_whale_signal(flagged.market))

    twitter_posts, reddit_posts, rss_posts, trend_score, whale_signal = (
        await asyncio.gather(
            twitter_task, reddit_task, rss_task, trends_task, whale_task,
            return_exceptions=True,
        )
    )

    # Flatten posts — handle any exceptions from individual scrapers gracefully
    all_posts: list[SocialPost] = []
    for result in [twitter_posts, reddit_posts, rss_posts]:
        if isinstance(result, list):
            all_posts.extend(result)
        elif isinstance(result, Exception):
            logger.warning("A research sub-agent failed: %s", result)

    # Coerce signal results (default to neutral on failure)
    trend_score_val    = trend_score    if isinstance(trend_score, float)    else 50.0
    whale_signal_val   = whale_signal   if isinstance(whale_signal, float)   else 0.0

    sentiment = _compute_sentiment(all_posts)
    narrative = _compare_narrative_to_odds(sentiment, flagged.market.yes_price)

    # Extract the most-liked/highest-score posts as key claims
    sorted_posts = sorted(
        all_posts,
        key=lambda p: p.likes + p.retweets + p.score,
        reverse=True,
    )
    key_claims = [p.text[:200] for p in sorted_posts[:5]]

    report = ResearchReport(
        market_id=market_id,
        question=question,
        posts=all_posts,
        sentiment=sentiment,
        narrative_summary=narrative,
        key_claims=key_claims,
        trend_score=trend_score_val,
        whale_bid_imbalance=whale_signal_val,
    )

    logger.info(
        "Research done — %d posts (tw=%d, rd=%d, rss=%d), "
        "sentiment=%.3f, trends=%.0f, whale=%+.2f",
        len(all_posts),
        len(twitter_posts)  if isinstance(twitter_posts,  list) else 0,
        len(reddit_posts)   if isinstance(reddit_posts,   list) else 0,
        len(rss_posts)      if isinstance(rss_posts,      list) else 0,
        sentiment.compound,
        trend_score_val,
        whale_signal_val,
    )

    return report


async def research_markets_parallel(
    flagged_markets: list[FlaggedMarket],
    max_concurrent: int = 5,
) -> list[ResearchReport]:
    """
    Run research agents across multiple markets with concurrency control.
    At most `max_concurrent` markets are researched simultaneously.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(fm: FlaggedMarket) -> Optional[ResearchReport]:
        async with semaphore:
            try:
                return await research_market(fm)
            except Exception as e:
                logger.error(
                    "Research failed for %s: %s", fm.market.question[:60], e
                )
                return None

    tasks = [asyncio.create_task(_bounded(fm)) for fm in flagged_markets]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]
