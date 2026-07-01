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

_client = anthropic.Anthropic(
    api_key=settings.anthropic_api_key,
    timeout=60.0,    # hard 60-second cap per request
    max_retries=1,   # only 1 retry max — prevents 50-min hangs on bad connections
)


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
        whale_bid_imbalance=report.whale_bid_imbalance,
        trend_score=report.trend_score,
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

ON-CHAIN & SEARCH SIGNALS:
  Whale order imbalance: {report.whale_bid_imbalance:+.2f}  (−1=heavy sells, 0=neutral, +1=heavy buys)
  Google Trends score:   {report.trend_score:.0f}/100  (50=average interest, 100=peak)

NARRATIVE ANALYSIS:
{report.narrative_summary}
{claims_block}

ML MODEL OUTPUT:
  XGBoost calibrated P(YES): {xgb_prob:.3f}  ({xgb_prob*100:.1f}%)
{lessons_block}

TASK — reason it through carefully before answering:
1. Weigh the evidence above by QUALITY, not just quantity. A handful of
   low-engagement posts or generic headlines is weak evidence; treat it as such.
2. Identify the single strongest concrete reason the market price might be
   WRONG. If you cannot name a specific, credible reason, there is no edge.
3. Estimate your own P(YES) between 0.0 and 1.0.
4. State your confidence (0.0–1.0). Base it on evidence quality: thin or
   conflicting information → LOW confidence. Do not manufacture confidence.
5. Recommend YES or NO ONLY if you found a specific mispricing reason in step 2
   AND your P(YES) diverges meaningfully from the market price. Otherwise
   recommend PASS. When in doubt, PASS — a skipped trade costs nothing.

CRITICAL: Respond with ONLY the JSON object below. Do NOT write any analysis,
preamble, or explanation before the JSON. Start your response with {{ immediately.
Put all your reasoning inside the "reasoning" field.

{{
  "llm_yes_probability": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "recommendation": "YES" | "NO" | "PASS",
  "reasoning": "<2-3 sentence explanation>",
  "key_insight": "<the single strongest mispricing signal you found>"
}}"""


def _parse_llm_response(text: str) -> dict:
    """Extract JSON from the LLM response — handles markdown fences and nested objects."""
    import json, re

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r'```(?:json)?\s*', '', text).strip()

    # 1. Direct parse of whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Find outermost { ... } block (handles nested objects unlike [^{}]+)
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # 3. Last resort: greedy regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM JSON — raw: %s", text[:300])
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
            # max_tokens=1024 (not 512) so the model never runs out of room
            # before emitting the JSON. Strong "JSON only" instruction in the
            # prompt keeps preamble out. (No assistant prefill — sonnet-4-6
            # rejects it with a 400.)
            response = _client.messages.create(
                model=settings.llm_model,
                max_tokens=1536,   # more room to reason through the evidence
                messages=[{"role": "user", "content": prompt}],
            )

            text_content = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            if not text_content:
                raise ValueError("LLM returned no text block")
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
            logger.warning("LLM prediction failed: %s — falling back to rule-based", e)
            # Fall back to rule-based prior (same as demo mode) rather than
            # leaving confidence at 0.50 which would block all trades.
            from ml.calibrator import _rule_based_probability
            llm_prob = _rule_based_probability(features)
            s = report.sentiment
            confidence = min(0.72, 0.52 + abs(s.compound) * 0.35)
            recommendation = (
                "YES" if llm_prob > market.yes_price + settings.min_edge
                else "NO" if (1 - llm_prob) > market.no_price + settings.min_edge
                else "PASS"
            )
            reasoning = f"[API fallback] Rule-based prior: sentiment={s.compound:+.3f}, market={market.yes_price:.3f}."
            # Don't fire fallback trades on markets expiring very soon —
            # rule-based prior has no time-awareness and can misfire badly
            if market.time_to_resolution_days < 2:
                logger.info(
                    "API fallback blocked on near-expiry market (%.1fd) '%s'",
                    market.time_to_resolution_days, market.question[:60],
                )
                return None

    # ── Ensemble (confidence-weighted) ────────────────────────────────────────
    # Base split: trust XGBoost more once it has real training data. Then tilt
    # toward the LLM when it's confident and back toward XGBoost when it's not,
    # so a hesitant LLM can't drag the estimate around on its own.
    base_xgb_weight = 0.60 if calibrator.is_trained else 0.40
    conf_tilt = (confidence - 0.5) * 0.30            # ±0.15 at confidence extremes
    llm_weight = min(0.85, max(0.15, (1.0 - base_xgb_weight) + conf_tilt))
    xgb_weight = 1.0 - llm_weight
    calibrated = xgb_weight * xgb_prob + llm_weight * llm_prob

    # ── Model-disagreement dampener ───────────────────────────────────────────
    # When XGBoost and the LLM strongly disagree, the ensemble estimate is less
    # trustworthy — cut confidence so contested signals size down (via the
    # conviction-scaled Kelly in the risk agent) or fail the confidence gate.
    disagreement = abs(xgb_prob - llm_prob)
    if disagreement > 0.20:
        penalty = min(0.20, (disagreement - 0.20) * 0.5)
        confidence = max(0.0, confidence - penalty)
        logger.info(
            "Model disagreement %.2f (xgb=%.3f vs llm=%.3f) — confidence −%.2f → %.2f",
            disagreement, xgb_prob, llm_prob, penalty, confidence,
        )

    # ── Contra-indicator: fade when extreme sentiment is already priced in ────
    # If everyone is extremely bullish AND the market already prices YES high,
    # the crowd is already in — dampen the YES edge (mean-revert toward market).
    # Likewise for extreme bearish sentiment with a low YES price.
    sentiment_compound = features.compound_sentiment
    if abs(sentiment_compound) > 0.60:
        market_already_reflects = (
            (sentiment_compound > 0 and market.yes_price > 0.65) or
            (sentiment_compound < 0 and market.yes_price < 0.35)
        )
        if market_already_reflects:
            # Fade factor: stronger sentiment → stronger fade (max 30% pull-back)
            fade = min(0.30, (abs(sentiment_compound) - 0.60) * 0.75)
            calibrated = calibrated * (1 - fade) + market.yes_price * fade
            logger.info(
                "Contra-indicator: extreme sentiment (%.2f) already priced in "
                "(market=%.3f) — fading %.0f%% toward market",
                sentiment_compound, market.yes_price, fade * 100,
            )

    # ── Respect the LLM's own recommendation ──────────────────────────────────
    # The model returns YES / NO / PASS. This used to be ignored, so the bot
    # took positions the AI itself flagged as PASS (as happened on every trade
    # in earlier runs). Now a PASS means "no clear edge" and cuts conviction —
    # only a genuinely strong quantitative signal can still get through.
    rec = (recommendation or "PASS").upper()
    if rec == "PASS":
        # LEARNING BOOTSTRAP: soften the PASS penalty (was 0.12) so fast-settling
        # trades still place and feed the learning loop. Re-tighten once the
        # model has real settled data to train on.
        confidence = max(0.0, confidence - 0.05)
        logger.info(
            "LLM recommended PASS — confidence −0.05 → %.2f for '%s'",
            confidence, market.question[:60],
        )

    # ── Determine side and edge ────────────────────────────────────────────────
    # Require the edge to clear BOTH the minimum edge AND the exchange fee, so
    # we don't take trades whose edge is eaten by fees (Polymarket US markets
    # carry a fee coefficient ~0.05). edge_floor protects against bleeding.
    # (Done BEFORE the confidence gate so the consensus/favorite boost below can
    # legitimately help a strong, confirmed setup clear the gate.)
    edge_floor = settings.min_edge + settings.fee_buffer
    yes_edge = calibrated - market.yes_price
    no_edge = (1 - calibrated) - market.no_price

    if yes_edge >= no_edge and yes_edge >= edge_floor:
        side = MarketSide.YES
        edge = yes_edge
        market_price = market.yes_price
    elif no_edge > yes_edge and no_edge >= edge_floor:
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

    # ── Favorites mode: only bet strong favorites (targets a high win rate) ────
    # The side we'd buy must be priced at/above the entry floor. This is what
    # produces a ~70% win rate — we only back likely outcomes — at the cost of
    # small per-win payouts. Off (0.0) by default; set MIN_ENTRY_PRICE to enable.
    # EXCEPTION: a genuinely well-researched, high-conviction bet with a real
    # edge may still take a "more risky" (below-floor) position — the backed bet.
    if market_price < settings.min_entry_price:
        research_backed = (
            confidence >= 0.70 and edge >= 0.08 and rec == side.value
        )
        if not research_backed:
            logger.info(
                "Favorites mode: %s at %.3f below entry floor %.2f — skipping '%s'",
                side.value, market_price, settings.min_entry_price, market.question[:60],
            )
            return None
        logger.info(
            "Below entry floor but RESEARCH-BACKED (conf=%.2f, edge=%.3f, LLM=%s) "
            "— allowing riskier bet on '%s'",
            confidence, edge, rec, market.question[:60],
        )

    # ── Direction check: don't trade against an explicit LLM call ──────────────
    # If the model explicitly recommended one side and our edge points the other
    # way, the math and the reasoning disagree on direction — skip rather than
    # override the AI.
    if rec in ("YES", "NO") and rec != side.value:
        logger.info(
            "LLM recommended %s but edge points %s — conflicting direction, "
            "skipping '%s'", rec, side.value, market.question[:60],
        )
        return None

    # ── High-conviction consensus ("spot the obvious wins") ────────────────────
    # A genuinely strong setup is one where INDEPENDENT signals converge on the
    # same side — not merely a high market price. When the LLM explicitly backs
    # this side (not PASS) and/or whale order-flow agrees, nudge confidence up.
    # This runs BEFORE the confidence gate so a confirmed favorite/consensus can
    # clear it, AND it feeds the conviction-scaled Kelly so the bet sizes larger.
    consensus: list[str] = []
    if rec == side.value:                      # LLM explicitly recommended this side
        confidence = min(1.0, confidence + 0.10)
        consensus.append("LLM")
    whale = features.whale_bid_imbalance
    if (side == MarketSide.YES and whale > 0.20) or (side == MarketSide.NO and whale < -0.20):
        confidence = min(1.0, confidence + 0.05)
        consensus.append("whale")
    # "Obvious win": a strong favorite (our side priced ≥ the favorite floor)
    # that the model ALSO finds underpriced — the edge already cleared the floor
    # above, so this is high win-rate AND positive EV, not a fair-priced coin
    # flip. The LLM must explicitly back the side. Give it extra conviction.
    if market_price >= settings.favorite_price_floor and rec == side.value:
        confidence = min(1.0, confidence + 0.08)
        consensus.append("favorite")
    if consensus:
        logger.info(
            "Consensus backing %s (%s) — confidence → %.2f for '%s'",
            side.value, "+".join(consensus), confidence, market.question[:60],
        )

    # ── Gate on confidence (sees the consensus boost above) ────────────────────
    if confidence < settings.min_confidence:
        logger.info(
            "Confidence %.2f < %.2f threshold — skipping '%s'",
            confidence, settings.min_confidence, market.question[:60],
        )
        return None

    # ── Longshot guardrail ─────────────────────────────────────────────────────
    # Buying a cheap longshot (market_price ≤ 0.20) on a thin edge is usually a
    # trap: on rare events our probability estimate is unreliable — a few points
    # of calibration error dwarfs the whole edge — and the crowd's favorite-
    # longshot bias means longshots are typically OVER-priced, not under. So on a
    # sub-20¢ side, demand an edge worth at least half the price AND above-
    # threshold confidence before betting; otherwise skip.
    if market_price <= 0.20:
        needed_edge = max(edge_floor, 0.5 * market_price)
        if edge < needed_edge or confidence < settings.min_confidence + 0.10:
            logger.info(
                "Longshot guardrail: %s at %.3f needs edge≥%.3f and conf≥%.2f — "
                "got edge=%.3f conf=%.2f — skipping '%s'",
                side.value, market_price, needed_edge,
                settings.min_confidence + 0.10, edge, confidence,
                market.question[:60],
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
