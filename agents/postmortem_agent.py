"""
Step 5 — Postmortem Agents (5 Parallel Specialists)
─────────────────────────────────────────────────────
After every loss, five specialist agents run concurrently:

  1. SentimentReviewer     — was the social signal reliable?
  2. MarketDataReviewer    — did price/spread/volume mislead us?
  3. ModelCalibrationAgent — was the XGBoost/LLM probability right?
  4. RiskSizingAgent       — was the Kelly bet sized correctly?
  5. PatternAgent          — does this loss match a known failure pattern?

Each agent produces a PostmortemFinding. Findings are saved to the DB
and translated into system updates so the bot doesn't repeat the mistake.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

import anthropic

from config import settings
from core.database import (
    get_active_lessons,
    save_lesson,
    save_postmortem_finding,
    save_system_update,
)
from core.models import (
    MarketSide,
    PostmortemFinding,
    PostmortemReport,
    Trade,
    TradeOutcome,
    TradeStatus,
)

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ── Shared LLM helper ─────────────────────────────────────────────────────────

async def _ask_specialist(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    trade: Trade,
) -> PostmortemFinding:
    """Run a single specialist agent and return its PostmortemFinding."""
    try:
        with _client.messages.stream(
            model=settings.llm_model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            response = stream.get_final_message()

        text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )

        # Parse structured JSON from response — handles markdown fences and nested objects
        import re
        text_clean = re.sub(r'```(?:json)?\s*', '', text).strip()
        parsed = {}
        for attempt in [
            lambda: json.loads(text_clean),
            lambda: json.loads(text_clean[text_clean.index('{'):text_clean.rindex('}')+1]),
            lambda: json.loads(re.search(r'\{.*\}', text_clean, re.DOTALL).group()),
        ]:
            try:
                parsed = attempt()
                break
            except Exception:
                continue

        finding = parsed.get("finding", text[:500])
        root_cause = parsed.get("root_cause", "Unknown")
        recommendation = parsed.get("recommendation", "Review process")
        severity = parsed.get("severity", "medium")

        logger.info("[%s] Finding: %s", agent_name, finding[:120])

        return PostmortemFinding(
            trade_id=trade.id or 0,
            agent_name=agent_name,
            finding=finding,
            root_cause=root_cause,
            recommendation=recommendation,
            severity=severity,
        )

    except Exception as e:
        logger.error("[%s] Postmortem agent failed: %s", agent_name, e)
        return PostmortemFinding(
            trade_id=trade.id or 0,
            agent_name=agent_name,
            finding=f"Agent encountered an error: {e}",
            root_cause="Agent error",
            recommendation="Investigate agent failure",
            severity="low",
        )


# ── Trade context builder ─────────────────────────────────────────────────────

def _trade_context(trade: Trade) -> str:
    notes = {}
    try:
        notes = json.loads(trade.notes or "{}")
    except Exception:
        pass

    features = notes.get("features", {})

    return f"""
LOSING TRADE DETAILS
====================
Market question:  {trade.question}
Side traded:      {trade.side.value}
Entry price:      {trade.entry_price:.3f}
Bet size (USDC):  ${trade.bet_usdc:.2f}
Shares bought:    {trade.shares:.4f}
PnL (USDC):       ${trade.pnl_usdc:.2f}
Placed at:        {trade.placed_at.isoformat() if trade.placed_at else 'unknown'}
Settled at:       {trade.settled_at.isoformat() if trade.settled_at else 'unknown'}
Tx hash:          {trade.tx_hash}

FEATURE SNAPSHOT AT TRADE TIME
================================
{json.dumps(features, indent=2) if features else '(not available)'}
""".strip()


# ── Five specialist agents ─────────────────────────────────────────────────────

async def _sentiment_reviewer(trade: Trade) -> PostmortemFinding:
    system = (
        "You are a sentiment analysis expert reviewing a failed prediction market trade. "
        "Focus on whether social sentiment signals (Twitter, Reddit, RSS) were "
        "misleading, misinterpreted, or insufficient. "
        "Return findings as JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Was the social sentiment signal reliable? Was it gaming or echo-chamber noise?
Did the compound sentiment score align with actual market direction?
What would have been a better sentiment signal to use?"""
    return await _ask_specialist("SentimentReviewer", system, user, trade)


async def _market_data_reviewer(trade: Trade) -> PostmortemFinding:
    system = (
        "You are a market microstructure analyst reviewing a failed prediction market trade. "
        "Focus on whether market data (price action, spread, volume, liquidity) "
        "sent misleading signals. "
        "Return findings as JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Was the price move a genuine signal or manipulation/noise?
Was the spread too wide indicating low confidence? Did volume/liquidity indicate informed trading
or just retail noise? What market data red flags were missed?"""
    return await _ask_specialist("MarketDataReviewer", system, user, trade)


async def _model_calibration_agent(trade: Trade) -> PostmortemFinding:
    system = (
        "You are a machine learning calibration expert reviewing a misprediction. "
        "Focus on whether the XGBoost model or LLM produced overconfident or biased "
        "probability estimates. "
        "Return findings as JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Was the calibrated probability accurate? Did the ensemble weight XGBoost vs LLM correctly?
Was the confidence threshold too low? What features likely caused the miscalibration?
How should the model be updated to avoid this error class?"""
    return await _ask_specialist("ModelCalibrationAgent", system, user, trade)


async def _risk_sizing_agent(trade: Trade) -> PostmortemFinding:
    system = (
        "You are a risk management expert reviewing a position-sizing decision. "
        "Focus on whether the Kelly criterion was applied correctly and whether "
        "the bet size was appropriate given the uncertainty. "
        "Return findings as JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Was the bet size too large for the confidence level?
Should the Kelly fraction have been lower? Were there concentration risk warnings?
How should the risk parameters be adjusted to survive this type of loss?"""
    return await _ask_specialist("RiskSizingAgent", system, user, trade)


async def _pattern_agent(trade: Trade, past_lessons: list[str]) -> PostmortemFinding:
    lessons_text = "\n".join(f"  • {l}" for l in past_lessons) or "  (none yet)"
    system = (
        "You are a pattern recognition expert who identifies recurring failure modes "
        "in prediction market trading. "
        "Return findings as JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

PAST LESSONS LEARNED:
{lessons_text}

Analyze: Does this loss match any previously identified failure patterns?
Is this a new failure mode? What systemic issue does this reveal?
What concrete rule or filter would prevent this class of error in the future?"""
    return await _ask_specialist("PatternAgent", system, user, trade)


# ── Postmortem orchestrator ────────────────────────────────────────────────────

async def run_postmortem(trade: Trade) -> Optional[PostmortemReport]:
    """
    Run all 5 postmortem agents in parallel after a losing trade.
    Saves findings to DB and updates system lessons.
    """
    if trade.outcome != TradeOutcome.LOSS:
        logger.debug("Skipping postmortem — trade %d is not a loss", trade.id or 0)
        return None

    logger.info(
        "Running 5-agent postmortem on trade %d: '%s'",
        trade.id or 0, trade.question[:70],
    )

    past_lessons = get_active_lessons(limit=15)

    # Fire all 5 specialists concurrently
    results = await asyncio.gather(
        _sentiment_reviewer(trade),
        _market_data_reviewer(trade),
        _model_calibration_agent(trade),
        _risk_sizing_agent(trade),
        _pattern_agent(trade, past_lessons),
        return_exceptions=True,
    )

    findings: list[PostmortemFinding] = []
    for result in results:
        if isinstance(result, PostmortemFinding):
            findings.append(result)
            save_postmortem_finding(result)
        elif isinstance(result, Exception):
            logger.error("A postmortem agent raised an exception: %s", result)

    # ── Synthesize system updates from findings ────────────────────────────────
    system_updates: list[str] = []
    for finding in findings:
        if finding.severity in ("high", "critical"):
            update_desc = (
                f"[{finding.agent_name}] {finding.recommendation}"
            )
            system_updates.append(update_desc)
            save_system_update(
                update_type=finding.agent_name,
                description=update_desc,
                payload={"trade_id": trade.id, "severity": finding.severity},
            )
            # Save as persistent lesson
            save_lesson(
                category=finding.agent_name,
                lesson=finding.recommendation,
                trade_id=trade.id,
            )

    # ── Generate combined lesson ───────────────────────────────────────────────
    all_recommendations = [f.recommendation for f in findings if f.recommendation]
    if all_recommendations:
        combined = (
            f"Trade {trade.id} ({'YES' if trade.side else 'NO'} on '{trade.question[:60]}'): "
            + " | ".join(all_recommendations[:3])
        )
        save_lesson(
            category="combined",
            lesson=combined,
            trade_id=trade.id,
        )

    report = PostmortemReport(
        trade_id=trade.id or 0,
        question=trade.question,
        findings=findings,
        system_updates=system_updates,
        lessons_learned="\n".join(all_recommendations),
    )

    logger.info(
        "Postmortem complete — %d findings, %d high/critical, %d system updates",
        len(findings),
        sum(1 for f in findings if f.severity in ("high", "critical")),
        len(system_updates),
    )
    return report


# ── Mid-flight review — learn EARLY from a deep open loser (before settlement) ──

async def run_midflight_review(
    trade_id: int,
    current_price: float,
    unrealized_pct: float,
) -> Optional[PostmortemFinding]:
    """
    Lightweight EARLY review of an OPEN position that is deeply underwater but
    has NOT settled or stopped out yet. Runs ONE specialist to extract a lesson
    the prediction agent reads next cycle, so the bot starts learning from a
    clearly-bad bet before it fully resolves. Does NOT change the trade outcome
    (it stays PENDING) and runs at most once per trade.
    """
    from core.database import midflight_reviewed, SessionLocal, TradeRow
    if trade_id is None or midflight_reviewed(trade_id):
        return None

    with SessionLocal() as s:
        r = s.query(TradeRow).filter(TradeRow.id == trade_id).first()
        if r is None:
            return None
        question, side_str, entry, bet, notes, market_id = (
            r.question, r.side, r.entry_price or 0.0, r.bet_usdc or 0.0,
            r.notes, r.market_id,
        )

    logger.info("Mid-flight review on OPEN trade %d (%.0f%% underwater): '%s'",
                trade_id, unrealized_pct, (question or "")[:60])

    try:
        feats = json.loads(notes or "{}").get("features", {})
    except Exception:
        feats = {}
    ctx = (
        f"OPEN LOSING POSITION (not yet settled)\n"
        f"Market:      {question}\n"
        f"Side:        {side_str}\n"
        f"Entry price: {entry:.3f}\n"
        f"Current:     {current_price:.3f}\n"
        f"Unrealized:  {unrealized_pct:.0f}%\n"
        f"Bet (USDC):  ${bet:.2f}\n\n"
        f"FEATURE SNAPSHOT AT ENTRY:\n"
        f"{json.dumps(feats, indent=2) if feats else '(not available)'}"
    )
    past_lessons = get_active_lessons(limit=10)
    lessons_block = "\n".join(f"- {l}" for l in past_lessons[:8])

    _t = Trade(
        id=trade_id, market_id=market_id, question=question or market_id,
        side=MarketSide.YES if side_str == "YES" else MarketSide.NO,
        entry_price=entry, bet_usdc=bet, shares=0.0, status=TradeStatus.PLACED,
    )
    finding = await _ask_specialist(
        "MidFlightReview",
        "You review OPEN prediction-market positions that are losing badly but "
        "have not resolved yet, to extract an EARLY lesson. Respond ONLY as JSON "
        "with keys finding, root_cause, recommendation, severity.",
        f"This position is currently {unrealized_pct:.0f}% underwater and has NOT "
        f"settled. Identify the single most likely reason taking it was a mistake, "
        f"and the concrete rule the scan/model should apply to avoid repeating it.\n\n"
        f"Past lessons:\n{lessons_block}\n\n{ctx}",
        _t,
    )
    save_postmortem_finding(finding)   # also serves as the run-once marker
    save_lesson(
        category="midflight_review",
        lesson=f"EARLY WARNING (open position {unrealized_pct:.0f}% down): {finding.recommendation}",
        trade_id=trade_id,
    )
    return finding


# ── Win analysis — learn from what WORKED ───────────────────────────────────────

def _win_context(trade: Trade) -> str:
    notes = {}
    try:
        notes = json.loads(trade.notes or "{}")
    except Exception:
        pass
    features = notes.get("features", {})
    return f"""
WINNING TRADE DETAILS
=====================
Market question:  {trade.question}
Side traded:      {trade.side.value}
Entry price:      {trade.entry_price:.3f}
Bet size (USDC):  ${trade.bet_usdc:.2f}
PnL (USDC):       +${trade.pnl_usdc:.2f}

FEATURE SNAPSHOT AT TRADE TIME
================================
{json.dumps(features, indent=2) if features else '(not available)'}
""".strip()


async def run_winmortem(trade: Trade) -> Optional[PostmortemReport]:
    """
    Analyze a WINNING trade to extract WHAT WORKED, so the system can find and
    repeat more trades like it. Mirror image of run_postmortem (which studies
    losses) — together they let the agents learn from both outcomes.
    """
    if trade.outcome != TradeOutcome.WIN:
        return None

    logger.info(
        "Running win-analysis on trade %d: '%s'", trade.id or 0, trade.question[:70]
    )
    past_lessons = get_active_lessons(limit=15)
    lessons_block = "\n".join(f"- {l}" for l in past_lessons[:10])
    ctx = _win_context(trade)

    results = await asyncio.gather(
        _ask_specialist(
            "SuccessFactor",
            "You analyze WINNING prediction-market trades to find what made them "
            "work so the system can repeat it. Respond ONLY as JSON with keys "
            "finding, root_cause, recommendation, severity. Use severity 'high' "
            "when the win reflects a strong, repeatable edge worth weighting more.",
            f"This trade WON. Identify the single biggest factor that made it "
            f"work, and how the bot should find more trades like it.\n\n{ctx}",
            trade,
        ),
        _ask_specialist(
            "WinPattern",
            "You spot repeatable WINNING patterns across trades. Respond ONLY as "
            "JSON with keys finding, root_cause, recommendation, severity.",
            f"This trade WON. What repeatable pattern does it match, and which "
            f"signals should the model weight MORE going forward?\n\n"
            f"Past lessons:\n{lessons_block}\n\n{ctx}",
            trade,
        ),
        return_exceptions=True,
    )

    findings: list[PostmortemFinding] = []
    for result in results:
        if isinstance(result, PostmortemFinding):
            findings.append(result)
            save_postmortem_finding(result)
            # Persist as a positive lesson the prediction agent will read
            save_lesson(
                category=f"win_{result.agent_name}",
                lesson=f"WHAT WORKED: {result.recommendation}",
                trade_id=trade.id,
            )
        elif isinstance(result, Exception):
            logger.error("A win-analysis agent raised: %s", result)

    logger.info("Win-analysis complete — %d findings on trade %d", len(findings), trade.id or 0)
    return PostmortemReport(
        trade_id=trade.id or 0,
        question=trade.question,
        findings=findings,
        system_updates=[],
        lessons_learned="\n".join(f.recommendation for f in findings),
    )
