"""
Step 5 — Postmortem Agents (5 Parallel Specialists)
─────────────────────────────────────────────────────
After every loss, five specialist agents run concurrently:

  1. TechnicalReviewer       — were the technical indicators reliable?
  2. SentimentReviewer       — was the social signal useful?
  3. ModelCalibrationAgent   — was the XGBoost/LLM probability correct?
  4. RiskSizingAgent         — was the Kelly bet sized correctly?
  5. PatternAgent            — does this match a known failure pattern?
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
    PostmortemFinding,
    PostmortemReport,
    Trade,
    TradeOutcome,
)

logger = logging.getLogger(__name__)
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=1)


async def _ask_specialist(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    trade: Trade,
) -> PostmortemFinding:
    try:
        message = _client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = next((b.text for b in message.content if b.type == "text"), "")
        text = re.sub(r'```(?:json)?\s*', '', text).strip()

        parsed = {}
        for attempt in [
            lambda: json.loads(text),
            lambda: json.loads(text[text.index('{'):text.rindex('}') + 1]),
            lambda: json.loads(re.search(r'\{.*\}', text, re.DOTALL).group()),
        ]:
            try:
                parsed = attempt()
                break
            except Exception:
                continue

        return PostmortemFinding(
            trade_id=trade.id or 0,
            agent_name=agent_name,
            finding=parsed.get("finding", text[:500]),
            root_cause=parsed.get("root_cause", "Unknown"),
            recommendation=parsed.get("recommendation", "Review process"),
            severity=parsed.get("severity", "medium"),
        )
    except Exception as e:
        logger.error("[%s] failed: %s", agent_name, e)
        return PostmortemFinding(
            trade_id=trade.id or 0,
            agent_name=agent_name,
            finding=f"Agent error: {e}",
            root_cause="Agent error",
            recommendation="Investigate agent failure",
            severity="low",
        )


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
Ticker:        {trade.ticker} ({trade.company_name})
Side:          {trade.side.value}
Entry price:   ${trade.entry_price:.2f}
Bet size:      ${trade.bet_usd:.2f}
Shares:        {trade.shares:.4f}
Stop loss:     ${trade.stop_loss_price:.2f}
Take profit:   ${trade.take_profit_price:.2f}
PnL:           ${trade.pnl_usd:.2f}
Entry date:    {trade.entry_date.isoformat() if trade.entry_date else 'unknown'}
Exit date:     {trade.exit_date.isoformat() if trade.exit_date else 'unknown'}

FEATURE SNAPSHOT:
{json.dumps(features, indent=2) if features else '(not available)'}
""".strip()


async def _technical_reviewer(trade: Trade) -> PostmortemFinding:
    system = (
        "You are a technical analysis expert reviewing a failed stock trade. "
        "Focus on whether RSI, MACD, Bollinger Bands, and volume signals were "
        "reliable or produced false signals. "
        "Return JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Were the technical indicators reliable? Was RSI a false oversold/overbought signal?
Did volume confirm or contradict the signal? What technical filter would have avoided this loss?"""
    return await _ask_specialist("TechnicalReviewer", system, user, trade)


async def _sentiment_reviewer(trade: Trade) -> PostmortemFinding:
    system = (
        "You are a sentiment analysis expert reviewing a failed stock trade. "
        "Focus on whether Reddit/news sentiment signals were misleading or manipulated. "
        "Return JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Was the social sentiment signal reliable for this stock?
Was it retail noise vs. informed opinion? How should the sentiment filter be improved?"""
    return await _ask_specialist("SentimentReviewer", system, user, trade)


async def _model_calibration_agent(trade: Trade) -> PostmortemFinding:
    system = (
        "You are an ML calibration expert reviewing a misprediction on a stock trade. "
        "Focus on whether XGBoost or LLM produced overconfident or biased probabilities. "
        "Return JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Was the probability calibration accurate? Were there overconfidence signs?
What features likely caused the miscalibration? How should the model be updated?"""
    return await _ask_specialist("ModelCalibrationAgent", system, user, trade)


async def _risk_sizing_agent(trade: Trade) -> PostmortemFinding:
    system = (
        "You are a risk management expert reviewing a position-sizing decision on a stock trade. "
        "Focus on whether Kelly sizing and stop-loss levels were appropriate. "
        "Return JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

Analyze: Was the stop-loss too tight? Was the position size too large for the confidence level?
Should the take-profit / stop-loss ratio be adjusted? How should risk parameters change?"""
    return await _ask_specialist("RiskSizingAgent", system, user, trade)


async def _pattern_agent(trade: Trade, past_lessons: list[str]) -> PostmortemFinding:
    lessons_text = "\n".join(f"  • {l}" for l in past_lessons) or "  (none yet)"
    system = (
        "You are a pattern recognition expert identifying recurring failure modes in stock trading. "
        "Return JSON with keys: finding, root_cause, recommendation, severity."
    )
    user = f"""{_trade_context(trade)}

PAST LESSONS LEARNED:
{lessons_text}

Analyze: Does this loss match a previously identified failure pattern?
Is this a new failure mode? What concrete rule would prevent this class of error?"""
    return await _ask_specialist("PatternAgent", system, user, trade)


async def run_postmortem(trade: Trade) -> Optional[PostmortemReport]:
    if trade.outcome != TradeOutcome.LOSS:
        return None

    logger.info("Running postmortem on trade %d: $%s", trade.id or 0, trade.ticker)
    past_lessons = get_active_lessons(limit=15)

    results = await asyncio.gather(
        _technical_reviewer(trade),
        _sentiment_reviewer(trade),
        _model_calibration_agent(trade),
        _risk_sizing_agent(trade),
        _pattern_agent(trade, past_lessons),
        return_exceptions=True,
    )

    findings: list[PostmortemFinding] = []
    for r in results:
        if isinstance(r, PostmortemFinding):
            findings.append(r)
            save_postmortem_finding(r)
        elif isinstance(r, Exception):
            logger.error("Postmortem agent raised: %s", r)

    system_updates: list[str] = []
    for finding in findings:
        if finding.severity in ("high", "critical"):
            desc = f"[{finding.agent_name}] {finding.recommendation}"
            system_updates.append(desc)
            save_system_update(
                update_type=finding.agent_name,
                description=desc,
                payload={"trade_id": trade.id, "severity": finding.severity},
            )
            save_lesson(category=finding.agent_name, lesson=finding.recommendation, trade_id=trade.id)

    all_recs = [f.recommendation for f in findings if f.recommendation]
    if all_recs:
        save_lesson(
            category="combined",
            lesson=f"Trade {trade.id} (${trade.ticker} {trade.side.value}): " + " | ".join(all_recs[:3]),
            trade_id=trade.id,
        )

    logger.info("Postmortem complete — %d findings, %d system updates", len(findings), len(system_updates))
    return PostmortemReport(
        trade_id=trade.id or 0,
        ticker=trade.ticker,
        findings=findings,
        system_updates=system_updates,
        lessons_learned="\n".join(all_recs),
    )
