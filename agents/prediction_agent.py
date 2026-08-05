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


def _signal_probability(features: PredictionFeatures, market_yes_price: float) -> float:
    """
    Third, INDEPENDENT probability estimate (the '2-of-3' cross-check lever).

    Deliberately NOT model-based — it starts from the market's own price and
    tilts it by smart-money (whale order flow), narrative (sentiment) and
    momentum (24h move). Because it uses different inputs than XGBoost (features)
    and the LLM (reasoning), it's a genuinely independent vote on P(YES).
    """
    p = market_yes_price
    # Whale kept mild (0.05): mega-liquid favorites carry huge market-maker ask
    # walls that read as −1.0 "whale sell pressure" — at 0.10 that alone vetoed
    # every favorite in the 2-of-3 vote.
    p += 0.05 * features.whale_bid_imbalance     # whales stacking YES → higher
    p += 0.06 * features.compound_sentiment      # bullish narrative → higher
    p += 0.30 * features.price_change_24h        # upward momentum → higher
    return max(0.02, min(0.98, p))


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

    if not settings.anthropic_api_key or not settings.llm_enabled:
        # FREE MODE / no key — rule-based prior only. Must trade TIMIDLY: the
        # old version boosted confidence to 0.75 with a real YES/NO rec, which
        # let sentiment-pumped priors through the high-conviction gates (same
        # bug class that bought a 4% longshot when the API ran out of credits).
        # Capped confidence + forced PASS → only plain favorites can trade.
        from ml.calibrator import _rule_based_probability
        llm_prob = _rule_based_probability(features)
        s = report.sentiment
        confidence = min(0.55, 0.45 + abs(s.compound) * 0.15)
        reasoning = (
            f"[FREE MODE — no LLM] Rule-based prior: sentiment={s.compound:+.3f}, "
            f"market={market.yes_price:.3f}."
        )
        recommendation = "PASS"
        logger.info(
            "[FREE MODE] P(YES)=%.3f, conf=%.2f (LLM disabled) for '%s'",
            llm_prob, confidence, market.question[:60],
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
            # Fall back to rule-based prior — but a no-LLM cycle must trade
            # TIMIDLY. This fallback once reached conf 0.72 + rec=YES and let a
            # sentiment-pumped prior take the "research-backed" longshot
            # exception (bought a 4% Medvedev longshot with the API down).
            # Cap confidence below every high-conviction gate and force PASS so
            # the consensus/favorite boosts and risky-bet exception cannot fire
            # without a real LLM opinion.
            from ml.calibrator import _rule_based_probability
            llm_prob = _rule_based_probability(features)
            s = report.sentiment
            confidence = min(0.55, 0.45 + abs(s.compound) * 0.15)
            recommendation = "PASS"
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
    # Base split: trust XGBoost in proportion to how much data it has actually
    # trained on. A 19-sample model once said P=0.22 on an 82% favorite and, at
    # 60% weight, vetoed every trade — full weight only from ~50 settled trades.
    from core.database import get_trade_stats as _gts
    try:
        _n_settled = _gts()["wins"] + _gts()["losses"]
    except Exception:
        _n_settled = 0
    if calibrator.is_trained:
        # Trust ramps with data: every settled trade retrains the model AND
        # increases its vote. 0.15 at ≤20 samples (a 19-sample model said
        # P=0.22 on 87% favorites — nearly advisory), rising linearly to the
        # full 0.60 at 60 settled trades.
        base_xgb_weight = min(0.60, 0.15 + 0.45 * max(0, _n_settled - 20) / 40.0)
    else:
        base_xgb_weight = 0.40
    conf_tilt = (confidence - 0.5) * 0.30            # ±0.15 at confidence extremes
    llm_weight = min(0.85, max(0.15, (1.0 - base_xgb_weight) + conf_tilt))
    xgb_weight = 1.0 - llm_weight
    calibrated = xgb_weight * xgb_prob + llm_weight * llm_prob

    # ── Model-disagreement dampener ───────────────────────────────────────────
    # When XGBoost and the LLM strongly disagree, the ensemble estimate is less
    # trustworthy — cut confidence so contested signals size down (via the
    # conviction-scaled Kelly in the risk agent) or fail the confidence gate.
    disagreement = abs(xgb_prob - llm_prob)
    # Disagreement from a <50-sample XGBoost isn't information — its wild
    # outputs (P=0.22 on 87% favorites) were sinking confidence below the gate
    # on every market. Only count disagreement once the model has real data.
    if disagreement > 0.20 and _n_settled >= 50:
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

    # ── FAVORITES FAST-PATH ─────────────────────────────────────────────────────
    # The strategy is to BACK strong favorites for their high win rate. The
    # market price IS the probability, so the edge/2-of-3-vote machinery below
    # (which hunts for MISpricing and demands every estimator sit above the
    # price) structurally rejects fair-priced favorites — a 95% favorite got
    # vetoed because a 19-sample XGBoost said 50%. So in favorites mode: only
    # the YES side is executable (NO/BUY_SHORT disabled); back a YES favorite
    # UNLESS the (now credit-backed) AI flags a genuine trap — it explicitly
    # says NO, its probability is FAR below the price, or whales are heavily
    # against it. Sized flat (see risk agent), because Kelly won't bet 0-edge.
    if settings.min_entry_price >= 0.5:
        if market.yes_price < settings.min_entry_price:
            logger.info(
                "Favorites mode: no YES favorite (YES=%.3f < %.2f) — skipping '%s'",
                market.yes_price, settings.min_entry_price, market.question[:60],
            )
            return None
        if market.yes_price > 0.90:
            logger.info(
                "Favorite too thin (YES=%.3f > 0.90; ~%.0f¢ payout not worth the risk) "
                "— skipping '%s'", market.yes_price, (1 - market.yes_price) * 100,
                market.question[:55],
            )
            return None
        traps = []
        if rec == "NO":
            traps.append("LLM=NO")
        if calibrated < market.yes_price - 0.20:
            traps.append(f"model {calibrated:.2f}<<price {market.yes_price:.2f}")
        if features.whale_bid_imbalance < -0.50:
            traps.append(f"whales against {features.whale_bid_imbalance:+.2f}")
        # DEGRADED MODE. With no LLM credits the AI veto never fires: llm_prob
        # falls back to xgb_prob, so `calibrated` carries no second opinion, and
        # the contra-indicator fade then pulls it toward the market price —
        # mechanically hiding how far the model actually sits below it. Trades
        # were placed at 0.69-0.90 while the raw model read 0.08-0.39, and the
        # blended figure never tripped the check above. So when the LLM is
        # unavailable, test the RAW model directly and refuse the extreme
        # disagreements. Deliberately a wide bar (not the 0.20 above): the
        # favorites thesis accepts that a thinly-trained model lags the market,
        # but not that we buy what it considers near-hopeless.
        if not settings.llm_enabled:
            raw_gap = market.yes_price - xgb_prob
            if raw_gap >= settings.free_mode_max_model_gap:
                traps.append(
                    f"FREE MODE: raw model {xgb_prob:.2f} vs price "
                    f"{market.yes_price:.2f} (gap {raw_gap:.2f} ≥ "
                    f"{settings.free_mode_max_model_gap:.2f})"
                )
        # Empirical self-check: refuse entries whose bucket (entry-price band or
        # market family) has a real losing record in OUR OWN settled trades.
        # Written lessons only ever reach an LLM prompt, so with credits out they
        # steer nothing; this is the learning loop that still bites in free mode.
        if settings.empirical_veto_enabled:
            try:
                from core.empirical import entry_verdict
                ok, why = entry_verdict(
                    market.yes_price, market.slug or "",
                    min_n=settings.empirical_min_samples,
                    max_win_rate=settings.empirical_max_win_rate,
                )
                if not ok:
                    traps.append(why)
            except Exception as e:
                logger.debug("Empirical veto check failed (non-blocking): %s", e)
        if traps:
            logger.info("Favorite trap-veto (%s) — skipping '%s'",
                        "; ".join(traps), market.question[:60])
            return None
        logger.info(
            "★ FAVORITE BACKED: YES on '%s' at %.3f (model=%.3f, raw_xgb=%.3f, "
            "LLM=%s%s, whale=%+.2f)",
            market.question[:50], market.yes_price, calibrated, xgb_prob, rec,
            "" if settings.llm_enabled else " [OFFLINE—no AI veto]",
            features.whale_bid_imbalance,
        )
        prediction = Prediction(
            market_id=market.condition_id,
            question=market.question,
            xgb_yes_probability=xgb_prob,
            llm_yes_probability=llm_prob,
            calibrated_yes_probability=market.yes_price,   # trade at the market prob
            market_yes_price=market.yes_price,
            edge=max(0.0, calibrated - market.yes_price),
            confidence=min(0.90, market.yes_price),        # favorite's own conviction
            side=MarketSide.YES,
            reasoning=f"Favorite backed at {market.yes_price:.2f}. {reasoning}"[:500],
            should_trade=True,
        )
        prediction._favorite_flat = True   # risk agent: flat-bet, don't Kelly-zero it
        return prediction

    # ── Determine side and edge (non-favorites / edge-hunting mode) ─────────────
    # Require the edge to clear BOTH the minimum edge AND the exchange fee, so
    # we don't take trades whose edge is eaten by fees (Polymarket US markets
    # carry a fee coefficient ~0.05). edge_floor protects against bleeding.
    edge_floor = settings.min_edge + settings.fee_buffer
    yes_edge = calibrated - market.yes_price
    no_edge = (1 - calibrated) - market.no_price

    fav_mode = settings.min_entry_price >= 0.5
    # Favorite fair-price tolerance scales with model maturity. A small-sample
    # XGBoost is systematically pessimistic (trained mostly on losses → it says
    # P=0.22 on 82% favorites), dragging the blend ~10pts below price even at
    # low weight. Until the model earns trust, back the favorite unless the
    # blend STRONGLY disagrees (>0.15 below price); tighten to 0.03 by 50 trades
    # so a matured model's edge check regains teeth.
    FAIR_TOL = 0.15 if _n_settled < 50 else 0.03
    candidates = []   # (is_favorite_side, edge, side, price)
    for side_, edge_, px_ in (
        (MarketSide.YES, yes_edge, market.yes_price),
        (MarketSide.NO, no_edge, market.no_price),
    ):
        is_fav = 1 if (fav_mode and px_ >= settings.min_entry_price) else 0
        floor_ = edge_floor - (FAIR_TOL if is_fav else 0.0)
        if edge_ >= floor_:
            candidates.append((is_fav, edge_, side_, px_))
    if not candidates:
        logger.info(
            "Edge too small (YES=%.3f, NO=%.3f) — skipping '%s'",
            yes_edge, no_edge, market.question[:60],
        )
        return None
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, edge, side, market_price = candidates[0]
    if side == MarketSide.NO:
        calibrated = 1 - calibrated   # flip for NO side presentation

    # ── Second lever: 2-of-3 independent vote ──────────────────────────────────
    # Require at least 2 of the 3 INDEPENDENT estimators (XGBoost, LLM, market-
    # signal) to agree there's edge on our side before we bet. One model being
    # wrong can no longer carry a trade on its own. All three are P(YES).
    signal_prob = _signal_probability(features, market.yes_price)
    estimators = [xgb_prob, llm_prob, signal_prob]
    # Small tolerance: an estimator sitting AT the market price is neutral, not
    # a disagreement. Without it, a 72c favorite needed price-anchored
    # estimators to sit strictly above 72% — an unfairly high bar that vetoed
    # 7 of 25 markets in a real run.
    VOTE_TOL = 0.05
    if side == MarketSide.YES:
        agree = sum(1 for e in estimators if e >= market.yes_price - VOTE_TOL)
    else:
        agree = sum(1 for e in estimators if e <= market.yes_price + VOTE_TOL)
    if agree < 2:
        logger.info(
            "2-of-3 vote failed (%d/3 agree on %s; xgb=%.2f llm=%.2f sig=%.2f vs %.2f) "
            "— skipping '%s'",
            agree, side.value, xgb_prob, llm_prob, signal_prob, market.yes_price,
            market.question[:60],
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
