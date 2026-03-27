"""
Step 3 — Prediction Agent
──────────────────────────
Combines XGBoost probability calibration with Claude Opus reasoning
to produce a calibrated YES probability for each market.

Only fires when confidence exceeds settings.min_confidence.
"""

from __future__ import annotations

import logging
from typing import Optional

import anthropic

from config import settings
from core.database import get_active_lessons
from core.models import (
    FlaggedMarket,
    MarketSide,
    Prediction,
    PredictionFeatures,
    ResearchReport,
)
from ml.calibrator import calibrator

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ── Feature extraction ────────────────────────────────────────────────────────

def _build_features(
    flagged: FlaggedMarket,
    report: ResearchReport,
) -> PredictionFeatures:
    s = report.sentiment
    m = flagged.market
    return PredictionFeatures(
        compound_sentiment=s.compound,
        positive_sentiment=s.positive,
        negative_sentiment=s.negative,
        post_count=s.post_count,
        avg_engagement=s.avg_engagement,
        price_change_24h=m.price_change_24h,
        spread=m.spread,
        liquidity_usdc=m.liquidity_usdc,
        volume_24h_usdc=m.volume_24h_usdc,
        time_to_resolution_days=m.time_to_resolution_days,
        current_yes_price=m.yes_price,
    )


# ── LLM calibration ───────────────────────────────────────────────────────────

def _build_llm_prompt(
    flagged: FlaggedMarket,
    report: ResearchReport,
    xgb_prob: float,
    lessons: list[str],
) -> str:
    m = flagged.market
    s = report.sentiment

    lessons_block = ""
    if lessons:
        lessons_block = "\n\nPAST LESSONS LEARNED (from postmortem analysis):\n" + "\n".join(
            f"  • {l}" for l in lessons[:10]
        )

    claims_block = ""
    if report.key_claims:
        claims_block = "\n\nTOP SOCIAL POSTS:\n" + "\n".join(
            f"  [{i+1}] {c}" for i, c in enumerate(report.key_claims)
        )

    return f"""You are a prediction market probability calibration expert.

MARKET QUESTION:
{m.question}

MARKET DATA:
  Current YES price:   {m.yes_price:.3f}  ({m.yes_price*100:.1f}%)
  Current NO price:    {m.no_price:.3f}   ({m.no_price*100:.1f}%)
  24h price change:    {m.price_change_24h:.3f}
  Spread:              {m.spread:.3f}
  Liquidity (USDC):    ${m.liquidity_usdc:,.0f}
  24h Volume (USDC):   ${m.volume_24h_usdc:,.0f}
  Days to resolution:  {m.time_to_resolution_days:.1f}
  Anomaly flag:        {m.flag_reason or 'none'}

SOCIAL SENTIMENT (VADER, {s.post_count} posts):
  Compound score:  {s.compound:+.3f}  (−1=very negative, +1=very positive)
  Positive:        {s.positive:.3f}
  Negative:        {s.negative:.3f}
  Avg engagement:  {s.avg_engagement:.1f}

NARRATIVE ANALYSIS:
{report.narrative_summary}
{claims_block}

ML MODEL OUTPUT:
  XGBoost calibrated P(YES): {xgb_prob:.3f}  ({xgb_prob*100:.1f}%)
{lessons_block}

TASK:
1. Critically evaluate all evidence above.
2. Identify the single strongest reason the market price might be WRONG.
3. Estimate your own P(YES) between 0.0 and 1.0.
4. State your confidence level (0.0–1.0) in this estimate.
5. Recommend YES, NO, or PASS.

Respond in this exact JSON format:
{{
  "llm_yes_probability": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "recommendation": "YES" | "NO" | "PASS",
  "reasoning": "<2-3 sentence explanation>",
  "key_insight": "<the single strongest mispricing signal you found>"
}}"""


def _parse_llm_response(text: str) -> dict:
    """Extract JSON from the LLM response."""
    import json, re
    # Try to find JSON block
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def predict_market(
    flagged: FlaggedMarket,
    report: ResearchReport,
) -> Optional[Prediction]:
    """
    Run the full prediction pipeline for one market.

    1. Extract features
    2. XGBoost → P(YES)
    3. LLM with adaptive thinking → P(YES) + reasoning
    4. Ensemble (60% XGBoost, 40% LLM)
    5. Return Prediction if confidence ≥ threshold, else None
    """
    features = _build_features(flagged, report)
    market = flagged.market

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_prob = calibrator.predict(features)
    logger.info(
        "XGBoost P(YES)=%.3f for '%s'", xgb_prob, market.question[:60]
    )

    # ── LLM (Claude Opus with adaptive thinking) ──────────────────────────────
    lessons = get_active_lessons(limit=10)
    prompt = _build_llm_prompt(flagged, report, xgb_prob, lessons)

    llm_prob = xgb_prob          # safe fallback
    confidence = 0.5
    reasoning = ""
    recommendation = "PASS"

    if not settings.anthropic_api_key:
        # Demo mode — use rule-based prior with boosted confidence from sentiment
        from ml.calibrator import _rule_based_probability
        llm_prob = _rule_based_probability(features)
        s = report.sentiment
        confidence = min(0.75, 0.55 + abs(s.compound) * 0.4)
        reasoning = (
            f"[DEMO — no API key] Rule-based prior: sentiment={s.compound:+.3f}, "
            f"market={market.yes_price:.3f}. Set ANTHROPIC_API_KEY for full LLM reasoning."
        )
        recommendation = (
            "YES" if llm_prob > market.yes_price + settings.min_edge
            else "NO" if (1 - llm_prob) > market.no_price + settings.min_edge
            else "PASS"
        )
        logger.info(
            "[DEMO] LLM P(YES)=%.3f, conf=%.2f, rec=%s for '%s'",
            llm_prob, confidence, recommendation, market.question[:60],
        )
    else:
        try:
            with _client.messages.stream(
                model=settings.llm_model,
                max_tokens=1024,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response = stream.get_final_message()

            text_content = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            parsed = _parse_llm_response(text_content)

            if parsed:
                llm_prob = float(parsed.get("llm_yes_probability", xgb_prob))
                confidence = float(parsed.get("confidence", 0.5))
                reasoning = parsed.get("reasoning", "")
                key_insight = parsed.get("key_insight", "")
                recommendation = parsed.get("recommendation", "PASS")
                if key_insight:
                    reasoning = f"{reasoning} Key insight: {key_insight}"

                logger.info(
                    "LLM P(YES)=%.3f, conf=%.2f, rec=%s for '%s'",
                    llm_prob, confidence, recommendation, market.question[:60],
                )

        except Exception as e:
            logger.error("LLM prediction failed: %s — using XGBoost only", e)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    # Weight XGBoost more heavily when it has training data
    xgb_weight = 0.60 if calibrator.is_trained else 0.40
    llm_weight = 1.0 - xgb_weight
    calibrated = xgb_weight * xgb_prob + llm_weight * llm_prob

    # ── Gate on confidence ────────────────────────────────────────────────────
    if confidence < settings.min_confidence:
        logger.info(
            "Confidence %.2f < %.2f threshold — skipping '%s'",
            confidence, settings.min_confidence, market.question[:60],
        )
        return None

    # ── Determine side and edge ────────────────────────────────────────────────
    yes_edge = calibrated - market.yes_price
    no_edge = (1 - calibrated) - market.no_price

    if yes_edge >= no_edge and yes_edge >= settings.min_edge:
        side = MarketSide.YES
        edge = yes_edge
        market_price = market.yes_price
    elif no_edge > yes_edge and no_edge >= settings.min_edge:
        side = MarketSide.NO
        edge = no_edge
        market_price = market.no_price
        calibrated = 1 - calibrated   # flip for NO side presentation
    else:
        logger.info(
            "Edge too small (YES=%.3f, NO=%.3f) — skipping '%s'",
            yes_edge, no_edge, market.question[:60],
        )
        return None

    prediction = Prediction(
        market_id=market.condition_id,
        question=market.question,
        xgb_yes_probability=xgb_prob,
        llm_yes_probability=llm_prob,
        calibrated_yes_probability=calibrated,
        market_yes_price=market.yes_price,
        edge=edge,
        confidence=confidence,
        side=side,
        reasoning=reasoning,
        should_trade=True,
    )

    logger.info(
        "Prediction: %s on '%s' — calibrated=%.3f, market=%.3f, edge=%.3f",
        side.value, market.question[:60],
        calibrated, market_price, edge,
    )
    return prediction
