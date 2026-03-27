"""
Mock data for sandbox / offline demo runs.
Simulates realistic Polymarket markets + social posts so the full
pipeline (scan → research → predict → risk → postmortem) can be
demonstrated without any outbound network access.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from core.models import Market, SocialPost

# ── Realistic mock markets ─────────────────────────────────────────────────────

def make_markets() -> list[Market]:
    now = datetime.now(timezone.utc)

    return [
        Market(
            condition_id="0xabc001",
            question="Will the Fed cut interest rates at the June 2025 meeting?",
            description="Resolves YES if the Federal Reserve cuts the federal funds rate target at the June 2025 FOMC meeting.",
            end_date_iso=(now + timedelta(days=12)).isoformat(),
            liquidity_usdc=48_200,
            volume_24h_usdc=9_400,
            yes_price=0.38,
            no_price=0.62,
            spread=0.00,
            price_change_24h=0.07,   # 7% move — anomaly flag
            time_to_resolution_days=12,
            tags=["economics", "fed", "rates"],
        ),
        Market(
            condition_id="0xabc002",
            question="Will Bitcoin exceed $120,000 before July 1, 2025?",
            description="Resolves YES if BTC/USD closes above $120,000 on any major exchange before July 1, 2025.",
            end_date_iso=(now + timedelta(days=18)).isoformat(),
            liquidity_usdc=132_000,
            volume_24h_usdc=41_500,
            yes_price=0.29,
            no_price=0.71,
            spread=0.00,
            price_change_24h=0.09,   # big move
            time_to_resolution_days=18,
            tags=["crypto", "bitcoin"],
        ),
        Market(
            condition_id="0xabc003",
            question="Will the 2025 NBA Finals go to 7 games?",
            description="Resolves YES if the 2025 NBA Finals series requires all 7 games to determine a champion.",
            end_date_iso=(now + timedelta(days=22)).isoformat(),
            liquidity_usdc=18_700,
            volume_24h_usdc=5_200,
            yes_price=0.44,
            no_price=0.56,
            spread=0.00,
            price_change_24h=0.04,
            time_to_resolution_days=22,
            tags=["sports", "nba"],
        ),
        Market(
            condition_id="0xabc004",
            question="Will the EU impose new tariffs on US tech companies in 2025?",
            description="Resolves YES if the European Union formally adopts new targeted tariffs on US technology companies before December 31, 2025.",
            end_date_iso=(now + timedelta(days=28)).isoformat(),
            liquidity_usdc=22_100,
            volume_24h_usdc=3_800,
            yes_price=0.55,
            no_price=0.45,
            spread=0.00,
            price_change_24h=0.06,
            time_to_resolution_days=28,
            tags=["politics", "trade", "EU"],
        ),
        Market(
            condition_id="0xabc005",
            question="Will Elon Musk remain CEO of Tesla at year-end 2025?",
            description="Resolves YES if Elon Musk is serving as CEO of Tesla, Inc. on December 31, 2025.",
            end_date_iso=(now + timedelta(days=14)).isoformat(),
            liquidity_usdc=61_500,
            volume_24h_usdc=14_200,
            yes_price=0.71,
            no_price=0.29,
            spread=0.00,
            price_change_24h=0.11,   # big move + high vol
            time_to_resolution_days=14,
            tags=["tech", "tesla", "musk"],
        ),
    ]


# ── Realistic mock social posts ────────────────────────────────────────────────

def make_posts(question: str) -> list[SocialPost]:
    now = datetime.now(timezone.utc)
    q = question.lower()

    if "fed" in q or "rate" in q or "interest" in q:
        return [
            SocialPost(source="twitter", author="macro_trader99", text="Fed pivot is coming — CPI print was soft, June cut is basically locked in. Pricing at 38% is way too cheap.", published_at=now - timedelta(hours=2), likes=842, retweets=211),
            SocialPost(source="reddit", author="r/Economics user", text="The futures market moved to 62% probability of a cut after Friday's jobs data. If you're looking at Polymarket at 38% there's serious mispricing here.", published_at=now - timedelta(hours=5), score=1340),
            SocialPost(source="rss", author="Reuters", text="Federal Reserve officials signal openness to June rate cut as inflation shows further signs of cooling. Multiple FOMC members cited recent data.", published_at=now - timedelta(hours=8), likes=0),
            SocialPost(source="twitter", author="fedwatcher", text="Powell's speech yesterday was the most dovish in 18 months. Market implied rate at 38% YES is a gift.", published_at=now - timedelta(hours=3), likes=501, retweets=98),
            SocialPost(source="twitter", author="bear_case_bob", text="Don't get ahead of yourself — services inflation is still sticky. Fed won't cut in June, they'll wait for more data.", published_at=now - timedelta(hours=6), likes=213, retweets=44),
        ]
    elif "bitcoin" in q or "btc" in q:
        return [
            SocialPost(source="twitter", author="bitcoinmaxi", text="BTC dominance at all-time high, ETF inflows ripping, halving tailwind still in effect. $120k before July is very possible.", published_at=now - timedelta(hours=1), likes=2100, retweets=640),
            SocialPost(source="reddit", author="CryptoBull2025", text="On-chain data showing massive accumulation at current levels. Whales are loading up. 29% for >$120k seems underpriced.", published_at=now - timedelta(hours=4), score=2800),
            SocialPost(source="rss", author="Bloomberg", text="Bitcoin surges past $95,000 on institutional demand, but analysts warn of near-term resistance at $100,000 level.", published_at=now - timedelta(hours=6), likes=0),
            SocialPost(source="twitter", author="skeptic_macro", text="BTC at 29% for $120k by July is actually about right. We'd need 25%+ from here in 18 days. Possible but not likely.", published_at=now - timedelta(hours=9), likes=389, retweets=71),
        ]
    elif "nba" in q or "finals" in q:
        return [
            SocialPost(source="twitter", author="nbastats", text="Historical data: 7-game Finals have occurred 37% of the time since 2000. 44% on Polymarket is slightly high.", published_at=now - timedelta(hours=3), likes=456, retweets=89),
            SocialPost(source="reddit", author="HoopsAnalytics", text="Both teams are evenly matched this year. I'd put 7 games at around 40%, so Polymarket at 44% is slightly overpriced for NO bettors.", published_at=now - timedelta(hours=7), score=720),
            SocialPost(source="rss", author="ESPN", text="NBA Finals preview: experts split on series length, with most projections favoring a 6 or 7 game series given the teams' defensive parity.", published_at=now - timedelta(hours=12), likes=0),
        ]
    elif "eu" in q or "tariff" in q:
        return [
            SocialPost(source="rss", author="Politico EU", text="European Commission signals new digital market rules could include tariff provisions targeting large US technology platforms.", published_at=now - timedelta(hours=4), likes=0),
            SocialPost(source="twitter", author="eutradewatch", text="EU-US trade tensions escalating. The 55% YES on Polymarket for tech tariffs seems about right given current political dynamics.", published_at=now - timedelta(hours=2), likes=178, retweets=34),
            SocialPost(source="reddit", author="EUpolicyNerd", text="The EU has been talking about this for 2 years and keeps backing down. I'd price this at 35-40% YES, not 55%.", published_at=now - timedelta(hours=8), score=430),
        ]
    elif "musk" in q or "tesla" in q:
        return [
            SocialPost(source="twitter", author="tslaholderr", text="Musk stepping back from DOGE, reportedly refocusing on Tesla. Board has no incentive to remove him. 71% seems fair.", published_at=now - timedelta(hours=1), likes=1200, retweets=290),
            SocialPost(source="reddit", author="TeslaInvestor", text="There is genuine board pressure right now after the governance controversies. I think 71% is too high — closer to 60%.", published_at=now - timedelta(hours=5), score=980),
            SocialPost(source="rss", author="WSJ", text="Tesla board meets to discuss CEO succession planning as activist investors push for governance changes amid Musk's government role.", published_at=now - timedelta(hours=3), likes=0),
            SocialPost(source="twitter", author="shortsellerX", text="If Tesla board finally acts, Polymarket at 71% YES for Musk CEO is massively overpriced. The WSJ article is a huge signal.", published_at=now - timedelta(hours=2), likes=678, retweets=183),
        ]

    # Generic fallback
    return [
        SocialPost(source="twitter", author="trader_anon", text=f"Watching {question[:60]} closely. Market seems fairly priced.", published_at=now - timedelta(hours=3), likes=50),
        SocialPost(source="reddit", author="PM_user", text=f"Anyone have insight on: {question[:80]}? Curious about the odds.", published_at=now - timedelta(hours=6), score=120),
    ]
