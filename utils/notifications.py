"""
Email / push notifications for trade milestones.

Sends alerts when:
  - A trade is placed
  - Bankroll crosses a milestone ($300, $500, $1000, $2000...)
  - Circuit breaker trips
  - Daily summary (9pm)

Setup in .env:
  NOTIFY_EMAIL=your@email.com
  NOTIFY_FROM_EMAIL=yourbot@gmail.com
  NOTIFY_SMTP_PASSWORD=your_gmail_app_password
  NOTIFY_SMTP_HOST=smtp.gmail.com
  NOTIFY_SMTP_PORT=587

Gmail setup: myaccount.google.com → Security → App Passwords → generate one
"""

from __future__ import annotations

import logging
import smtplib
import os
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# Milestones in USD — bot emails you when bankroll crosses these
MILESTONES = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]

# File to track which milestones have already been notified
_MILESTONE_FILE = ".notified_milestones.txt"


def _already_notified(milestone: float) -> bool:
    try:
        if os.path.exists(_MILESTONE_FILE):
            with open(_MILESTONE_FILE) as f:
                notified = {float(x) for x in f.read().split() if x}
            return milestone in notified
    except Exception:
        pass
    return False


def _mark_notified(milestone: float) -> None:
    try:
        existing: set[float] = set()
        if os.path.exists(_MILESTONE_FILE):
            with open(_MILESTONE_FILE) as f:
                existing = {float(x) for x in f.read().split() if x}
        existing.add(milestone)
        with open(_MILESTONE_FILE, "w") as f:
            f.write("\n".join(str(m) for m in sorted(existing)))
    except Exception:
        pass


def _send_email(subject: str, body: str) -> bool:
    """Send an email via SMTP. Returns True on success."""
    to_email = getattr(settings, "notify_email", "")
    from_email = getattr(settings, "notify_from_email", "")
    smtp_password = getattr(settings, "notify_smtp_password", "")
    smtp_host = getattr(settings, "notify_smtp_host", "smtp.gmail.com")
    smtp_port = int(getattr(settings, "notify_smtp_port", 587))

    if not all([to_email, from_email, smtp_password]):
        logger.debug("Email not configured — skipping notification")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        # Plain text
        msg.attach(MIMEText(body, "plain"))

        # HTML version
        html = body.replace("\n", "<br>").replace("$", "&#36;")
        html = f"<html><body style='font-family:monospace;font-size:14px'>{html}</body></html>"
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(from_email, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())

        logger.info("Email sent: %s → %s", subject, to_email)
        return True

    except Exception as e:
        logger.warning("Email send failed: %s", e)
        return False


def notify_trade_placed(
    question: str,
    side: str,
    bet_usdc: float,
    price: float,
    edge: float,
    bankroll: float,
) -> None:
    """Send notification when a new trade is placed."""
    subject = f"🎯 Trade placed — {side} on {question[:40]}"
    body = f"""New trade placed by your prediction market bot:

Market:    {question}
Side:      {side}
Bet:       ${bet_usdc:.2f} USDC
Price:     {price:.3f} ({price*100:.1f}%)
Edge:      {edge:+.3f}
Bankroll:  ${bankroll:.2f} USDC
Time:      {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

The bot is running. You don't need to do anything.
"""
    _send_email(subject, body)


def notify_trade_settled(
    question: str,
    side: str,
    outcome: str,
    pnl: float,
    bankroll: float,
) -> None:
    """Send notification when a trade resolves."""
    icon = "✅" if outcome == "WIN" else "❌"
    subject = f"{icon} Trade {outcome} — {'+' if pnl >= 0 else ''}${pnl:.2f}"
    body = f"""{icon} Trade settled:

Market:    {question}
Side:      {side}
Outcome:   {outcome}
PnL:       ${pnl:+.2f} USDC
Bankroll:  ${bankroll:.2f} USDC
Time:      {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
"""
    _send_email(subject, body)


def check_and_notify_milestone(bankroll: float) -> None:
    """Check if bankroll crossed a milestone and send email if so."""
    for milestone in MILESTONES:
        if bankroll >= milestone and not _already_notified(milestone):
            _mark_notified(milestone)
            subject = f"🚀 Milestone hit — bankroll crossed ${milestone:,.0f}!"
            body = f"""Your prediction market bot hit a new milestone!

💰 Bankroll: ${bankroll:.2f} USDC
🎯 Milestone: ${milestone:,.0f} crossed
📅 Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

The bot is still running and trading. You don't need to do anything.

To check the dashboard at any time:
  streamlit run dashboard.py

To stop the bot:
  Press Ctrl+C in the terminal where it's running.

Keep it running and let it compound!
"""
            _send_email(subject, body)
            logger.info("Milestone notification sent: $%s", milestone)


def notify_circuit_breaker(win_rate: float, window: int) -> None:
    """Alert if the circuit breaker trips."""
    subject = "🚨 Bot paused — circuit breaker triggered"
    body = f"""Your trading bot has automatically paused.

Reason: Win rate dropped to {win_rate:.1%} over the last {window} trades.
        (Threshold: 52% — strategy may be breaking down)

Action required:
  1. Check dashboard: streamlit run dashboard.py
  2. Review recent losses — look at the Postmortem tab
  3. If you want to resume: delete the file TRADING_PAUSED.txt
     (do this only after reviewing what went wrong)

Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
"""
    _send_email(subject, body)


def send_daily_summary(stats: dict) -> None:
    """Send a daily performance summary email."""
    wr = stats.get("win_rate", 0)
    pnl = stats.get("total_pnl_usdc", 0)
    total = stats.get("total", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pending = stats.get("pending", 0)

    icon = "📈" if pnl >= 0 else "📉"
    subject = f"{icon} Daily summary — Win rate {wr:.1%} | PnL ${pnl:+,.2f}"
    body = f"""Daily performance summary from your prediction market bot:

📊 Stats:
  Total trades:  {total}
  Wins:          {wins}
  Losses:        {losses}
  Pending:       {pending}
  Win rate:      {wr:.1%}
  Total PnL:     ${pnl:+,.2f} USDC

{"✅ On track — strategy working" if wr >= 0.60 else "⚠️  Win rate below target — monitor closely"}

Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

The bot is still running. No action needed.
"""
    _send_email(subject, body)
