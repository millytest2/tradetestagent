"""
Step 3 — Prediction Agent
──────────────────────────
Combines XGBoost probability calibration with Claude reasoning
to predict whether a stock will be UP or DOWN in 5 days.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

from config import settings
from core.database import get_active_lessons
from core.models import (
    FlaggedStock,
    Prediction,
    PredictionFeatures,
    ResearchReport,
    StockSide,
)
from ml.calibrator import calibrator

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(
    api_key=settings.anthropic_api_key,
    timeout=60.0,
    max_retries=1,   # prevents 50-min hangs on bad connections
)


def _build_features(flagged: FlaggedStock, report: ResearchReport) -> PredictionFeatures:
    s = report.sentiment
    st = flagged.stock
    return PredictionFeatures(
        rsi_14=st.rsi_14,
        macd_signal=st.macd_signal,
        bb_position=st.bb_position,
        volume_ratio=st.volume_ratio,
        price_change_1d=st.price_change_1d,
        price_change_5d=st.price_change_5d,
        price_change_20d=st.price_change_20d,
        distance_from_52w_high=st.distance_from_52w_high,
        short_interest_ratio=st.short_interest_ratio,
        compound_sentiment=s.compound,
        post_count=s.post_count,
        avg_engagement=s.avg_engagement,
        trend_score=report.trend_score,
        whale_bid_imbalance=report.whale_bid_imbalance,
    )


def _build_llm_prompt(
    flagged: FlaggedStock,
    report: ResearchReport,
    xgb_prob: float,
    lessons: list[str],
) -> str:
    st = flagged.stock
    s = report.sentiment

    lessons_block = ""
    if lessons:
        lessons_block = "\n\nPAST LESSONS LEARNED:\n" + "\n".join(f"  • {l}" for l in lessons[:10])

    claims_block = ""
    if report.key_claims:
        claims_block = "\n\nTOP SOCIAL POSTS:\n" + "\n".join(
            f"  [{i+1}] {c}" for i, c in enumerate(report.key_claims[:6])
        )

    return f"""You are a quantitative equity analyst and stock prediction expert.

STOCK: {st.ticker} ({st.company_name or 'unknown'})
SECTOR: {st.sector or 'unknown'}
Signal: {flagged.flag_reason}

TECHNICAL INDICATORS:
  Current price:        ${st.current_price:.2f}
  1-day change:         {st.price_change_1d:+.1%}
  5-day change:         {st.price_change_5d:+.1%}
  20-day change:        {st.price_change_20d:+.1%}
  RSI(14):              {st.rsi_14:.1f}  (30=oversold, 70=overbought)
  MACD histogram:       {st.macd_signal:+.3f}  (positive=bullish momentum)
  Bollinger position:   {st.bb_position:.2f}  (0=lower band, 1=upper band)
  Volume ratio (1d/20d):{st.volume_ratio:.2f}x
  Distance from 52w high: {st.distance_from_52w_high:+.1%}
  Distance from 52w low:  {st.distance_from_52w_low:+.1%}

SENTIMENT ({s.post_count} posts):
  Compound: {s.compound:+.3f}  Positive: {s.positive:.2f}  Negative: {s.negative:.2f}
  Avg engagement: {s.avg_engagement:.1f}

SEARCH TRENDS:
  Google Trends score: {report.trend_score:.0f}/100  (50=average, 100=peak interest)

NARRATIVE:
{report.narrative_summary}
{claims_block}

ML MODEL OUTPUT:
  XGBoost P(UP in 5 days): {xgb_prob:.3f}  ({xgb_prob*100:.1f}%)
{lessons_block}

TASK:
Predict whether this stock will be HIGHER or LOWER in 5 trading days.

Consider:
1. Is the technical signal (RSI/MACD/BB) reliable or a false signal?
2. Does social sentiment align with or contradict the price action?
3. Are there any red flags (earnings, macro, manipulation)?
4. What is the single strongest reason your prediction might be wrong?

Respond in this exact JSON format:
{{
  "llm_up_probability": <float 0.0-1.0, probability stock is higher in 5 days>,
  "confidence": <float 0.0-1.0>,
  "recommendation": "LONG" | "SHORT" | "PASS",
  "reasoning": "<2-3 sentence explanation>",
  "key_insight": "<the single strongest signal you found>"
}}"""


def _parse_llm_response(text: str) -> dict:
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    for attempt in [
        lambda: json.loads(text),
        lambda: json.loads(text[text.index('{'):text.rindex('}') + 1]),
        lambda: json.loads(re.search(r'\{.*\}', text, re.DOTALL).group()),
    ]:
        try:
            return attempt()
        except Exception:
            continue
    return {}


async def _call_llm(prompt: str) -> Optional[dict]:
    try:
        message = _client.messages.create(
            model=settings.llm_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in message.content if b.type == "text"), "")
        return _parse_llm_response(text)
    except anthropic.APITimeoutError:
        logger.warning("LLM timed out")
        return None
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


def _rule_based_prediction(features: PredictionFeatures) -> float:
    """Conservative prior: mean-reversion + momentum blend."""
    prob = 0.5

    # RSI mean reversion
    if features.rsi_14 < 30:
        prob += 0.10   # oversold → likely bounce
    elif features.rsi_14 > 70:
        prob -= 0.10   # overbought → likely pullback

    # MACD momentum
    prob += min(max(features.macd_signal * 0.02, -0.08), 0.08)

    # Sentiment
    prob += features.compound_sentiment * 0.08

    # Bollinger Band extremes
    if features.bb_position <= 0.1:
        prob += 0.05
    elif features.bb_position >= 0.9:
        prob -= 0.05

    return float(max(0.05, min(0.95, prob)))


async def predict_stock(
    flagged: FlaggedStock,
    report: ResearchReport,
) -> Optional[Prediction]:
    """Predict whether a stock will be up or down in 5 days."""
    ticker = flagged.stock.ticker
    features = _build_features(flagged, report)

    # XGBoost probability
    xgb_prob = calibrator.predict_proba(features)
    logger.info("$%s XGBoost P(UP)=%.3f", ticker, xgb_prob)

    # LLM probability
    lessons = get_active_lessons(limit=10)
    prompt = _build_llm_prompt(flagged, report, xgb_prob, lessons)
    llm_data = await _call_llm(prompt)

    if llm_data:
        llm_prob = float(llm_data.get("llm_up_probability", 0.5))
        confidence = float(llm_data.get("confidence", 0.5))
        reasoning = llm_data.get("reasoning", "")
        recommendation = llm_data.get("recommendation", "PASS")

        # Ensemble: 40% XGBoost, 60% LLM
        if calibrator.is_trained:
            calibrated = 0.40 * xgb_prob + 0.60 * llm_prob
        else:
            calibrated = 0.20 * xgb_prob + 0.80 * llm_prob

        if recommendation == "PASS":
            logger.info("$%s: LLM recommends PASS", ticker)
            return None
    else:
        # Rule-based fallback
        logger.warning("$%s: LLM unavailable — using rule-based fallback", ticker)
        calibrated = _rule_based_prediction(features)
        confidence = 0.55
        reasoning = "Rule-based fallback (LLM unavailable)"

    # Determine side and edge
    edge = abs(calibrated - 0.5)
    side = StockSide.LONG if calibrated > 0.5 else StockSide.SHORT

    prediction = Prediction(
        ticker=ticker,
        company_name=flagged.stock.company_name,
        xgb_up_probability=xgb_prob,
        llm_up_probability=llm_data.get("llm_up_probability", xgb_prob) if llm_data else xgb_prob,
        calibrated_up_probability=calibrated,
        edge=edge,
        confidence=confidence,
        side=side,
        reasoning=reasoning,
        should_trade=edge >= settings.min_edge and confidence >= settings.min_confidence,
    )

    logger.info(
        "$%s: calibrated=%.3f, edge=%.3f, confidence=%.2f, side=%s, trade=%s",
        ticker, calibrated, edge, confidence, side.value, prediction.should_trade,
    )
    return prediction
