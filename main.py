"""
Prediction Market Trading Bot — Main Orchestrator
══════════════════════════════════════════════════

Five-step pipeline:
  1. Scan Agent      — filter 300+ markets, flag anomalies
  2. Research Agents — parallel Twitter / Reddit / RSS + sentiment
  3. Prediction Agent — XGBoost + Claude Opus calibration
  4. Risk Agent      — Kelly sizing, gate, place trade on-chain
  5. Postmortem      — 5 agents analyze every loss and update the system

Usage
─────
  # Single run (dry-run, no real money)
  python main.py --run-once

  # Continuous loop every 5 minutes
  python main.py

  # Live trading (DANGEROUS — real USDC on Polygon)
  python main.py --live

  # Retrain ML model from settled trade history
  python main.py --retrain

  # Show performance statistics
  python main.py --stats
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import settings
from core.database import (
    get_losing_trades,
    get_trade_stats,
    init_db,
    purge_stale_lessons,
)
from agents.scan_agent import scan_markets
from agents.research_agent import research_markets_parallel
from agents.prediction_agent import predict_market
from agents.risk_agent import evaluate_and_trade, monitor_and_settle
from agents.postmortem_agent import run_postmortem
from ml.calibrator import calibrator

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
console = Console()


# ── Pretty output helpers ─────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Prediction Market Trading Bot[/bold cyan]\n"
        "[dim]Scan → Research → Predict → Risk → Postmortem[/dim]\n"
        f"[dim]Model: {settings.llm_model} | "
        f"Bankroll: ${settings.bankroll_usdc:,.0f} USDC | "
        f"Min confidence: {settings.min_confidence:.0%}[/dim]",
        border_style="cyan",
    ))


def _print_stats() -> None:
    stats = get_trade_stats()
    table = Table(title="Trade Performance", box=box.ROUNDED, border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total trades", str(stats["total"]))
    table.add_row("Wins", f"[green]{stats['wins']}[/green]")
    table.add_row("Losses", f"[red]{stats['losses']}[/red]")
    table.add_row("Pending", str(stats["pending"]))
    win_rate = stats["win_rate"]
    color = "green" if win_rate >= 0.60 else "yellow" if win_rate >= 0.45 else "red"
    table.add_row("Win rate", f"[{color}]{win_rate:.1%}[/{color}]")
    pnl = stats["total_pnl_usdc"]
    pnl_color = "green" if pnl >= 0 else "red"
    table.add_row("Total PnL", f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]")

    console.print(table)


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(dry_run: bool = True, top_n: int = 10, use_mock: bool = False,
                       max_trades: int = None) -> None:
    """Execute one full scan→research→predict→risk cycle.

    max_trades: if set, stop after this many trades are placed (used by
    --test-trade to place exactly one small verification trade).
    """

    cycle_start = datetime.utcnow()
    console.rule(f"[cyan]Cycle started {cycle_start.strftime('%H:%M:%S UTC')}[/cyan]")

    # ── Step 0: Settle resolved positions + manage open ones (buy/sell/hold) ───
    if not use_mock:
        try:
            settled = await _settle_pending_trades()
            if settled:
                console.print(f"  [green]✓ Settled {settled} resolved position(s)[/green]")
        except Exception as e:
            logger.debug("Settlement sweep failed (non-blocking): %s", e)
        # Reconcile MANUAL closes: if you sold a position yourself on Polymarket,
        # it's gone from the live account but still PENDING in our DB — detect
        # that and settle it so the bot learns from the outcome.
        try:
            reconciled = await _reconcile_exchange_positions()
            if reconciled:
                console.print(f"  [yellow]↔ Reconciled {reconciled} manually-closed position(s) for learning[/yellow]")
        except Exception as e:
            logger.debug("Reconciliation failed (non-blocking): %s", e)
        try:
            exited = await _manage_open_positions(dry_run=dry_run)
            if exited:
                console.print(f"  [yellow]✓ Exited {exited} position(s) on stop-loss/take-profit[/yellow]")
        except Exception as e:
            logger.debug("Position management failed (non-blocking): %s", e)
        # Real-time P&L: mark open positions to current market so the true
        # (unrealized) position is visible every cycle, not just at settlement.
        try:
            await _report_unrealized_pnl()
        except Exception as e:
            logger.debug("Unrealized P&L report failed (non-blocking): %s", e)

    # ── Kill switch: pause opening new positions (still settle/manage/report) ──
    if settings.pause_new_trades and not use_mock:
        console.print(
            "  [yellow]⏸ PAUSE_NEW_TRADES is ON — holding all open positions, "
            "opening nothing new this cycle.[/yellow]"
        )
        return

    # ── Step 1: Scan ──────────────────────────────────────────────────────────
    console.print("[bold]Step 1[/bold] Scanning markets...")

    if use_mock:
        from demo_data import make_markets
        from agents.scan_agent import _passes_base_filter, _detect_anomaly, _priority_score, FlaggedMarket
        raw = make_markets()
        flagged = []
        for m in raw:
            if _passes_base_filter(m):
                is_flagged, reason = _detect_anomaly(m)
                m.is_flagged = is_flagged; m.flag_reason = reason
                score = _priority_score(m, reason)
                flagged.append(FlaggedMarket(market=m, flag_reason=reason or "base filter pass", priority_score=score))
        flagged.sort(key=lambda x: x.priority_score, reverse=True)
        console.print(f"  [dim](mock data — {len(raw)} markets injected)[/dim]")
    else:
        flagged = await scan_markets(limit=settings.scan_limit)

    if not flagged:
        console.print("[yellow]No markets passed the scan filter — skipping cycle[/yellow]")
        return

    # Take the top N by priority score for deeper research
    top_flagged = flagged[:top_n]
    console.print(
        f"  ✓ [green]{len(flagged)}[/green] markets queued "
        f"→ researching top [cyan]{len(top_flagged)}[/cyan]"
    )

    # ── Step 2: Research (parallel) ────────────────────────────────────────────
    console.print("[bold]Step 2[/bold] Running parallel research agents...")
    if use_mock:
        from demo_data import make_posts
        from agents.research_agent import _compute_sentiment, _compare_narrative_to_odds
        from core.models import ResearchReport
        reports = []
        for fm in top_flagged:
            posts = make_posts(fm.market.question)
            sentiment = _compute_sentiment(posts)
            narrative = _compare_narrative_to_odds(sentiment, fm.market.yes_price)
            key_claims = [p.text[:200] for p in sorted(posts, key=lambda p: p.likes+p.retweets+p.score, reverse=True)[:5]]
            reports.append(ResearchReport(
                market_id=fm.market.condition_id, question=fm.market.question,
                posts=posts, sentiment=sentiment, narrative_summary=narrative,
                key_claims=key_claims,
            ))
        console.print(f"  [dim](mock social posts — {sum(len(r.posts) for r in reports)} posts total)[/dim]")
    else:
        reports = await research_markets_parallel(top_flagged, max_concurrent=5)
    console.print(f"  ✓ Research complete for [green]{len(reports)}[/green] markets")

    # ── SAFETY STOP: don't keep opening positions across cycles ────────────────
    # Across many 30-min GitHub runs this prevents the bot from over-deploying
    # the whole wallet. If we already hold the max number of open positions,
    # skip new trades this cycle (existing positions still settle/exit).
    if not use_mock:
        try:
            from core.database import SessionLocal, TradeRow
            with SessionLocal() as _s:
                open_count = _s.query(TradeRow).filter(TradeRow.outcome == "PENDING").count()
            if open_count >= settings.max_open_positions:
                console.print(
                    f"  [yellow]⏸ Safety stop: {open_count} open positions "
                    f"(cap {settings.max_open_positions}) — holding, no new trades[/yellow]"
                )
                await _run_pending_postmortems()
                return
        except Exception as e:
            logger.debug("Open-position check failed (non-blocking): %s", e)

    # ── Steps 3 & 4: Predict + Risk ────────────────────────────────────────────
    console.print("[bold]Step 3+4[/bold] Predicting and evaluating risk...")
    trades_placed = 0

    # Size bets off the REAL account balance minus capital already locked in
    # open positions, so the bot never commits more than the wallet holds —
    # even across many 30-minute cycles.
    live_bankroll = None
    RESERVE_FLOOR = 1.0  # stop trading once available drops below $1
    if not use_mock and settings.live_exchange.lower() in ("polymarket_us", "polymarketus", "pmus"):
        try:
            from integrations.polymarket_us import get_account_balances
            from core.database import get_committed_capital
            bals = await get_account_balances()
            total = bals.get("total")   # account equity incl. open positions
            cash = bals.get("cash")     # spendable USDC
            committed = get_committed_capital()
            # Spendable base:
            #  • exchange-reported CASH already excludes capital locked in open
            #    positions — use it directly (do NOT subtract committed again).
            #  • else fall back to total equity MINUS our committed capital.
            #  • else the configured bankroll minus committed.
            if cash is not None:
                spendable = max(0.0, cash)
            elif total is not None:
                spendable = max(0.0, total - committed)
            else:
                spendable = max(0.0, settings.bankroll_usdc - committed)
            live_bankroll = spendable
            console.print(
                f"  [dim]Equity: {('$%.2f' % total) if total is not None else 'n/a'} | "
                f"cash: {('$%.2f' % cash) if cash is not None else 'n/a'} | "
                f"committed(open): ${committed:.2f} | available: ${live_bankroll:.2f}[/dim]"
            )
            if live_bankroll < RESERVE_FLOOR:
                console.print(
                    "  [yellow]→ Available balance below $1 — skipping new "
                    "trades this cycle (managing open positions only).[/yellow]"
                )
                top_flagged = []  # no new entries; settlement/exits already ran
        except Exception as e:
            logger.debug("Balance fetch failed, using configured bankroll: %s", e)

    # Source-of-truth dedup: fetch the markets we ACTUALLY hold on the exchange
    # so we never re-buy one, even if the local trade DB (GitHub cache) was lost
    # or a prior cycle didn't persist.
    held_slugs: set = set()
    if not use_mock and settings.live_exchange.lower() in ("polymarket_us", "polymarketus", "pmus"):
        try:
            from integrations.polymarket_us import get_open_positions
            held_slugs = await get_open_positions()
        except Exception as e:
            logger.debug("Exchange position fetch failed (non-blocking): %s", e)

    # Running available balance — decremented as trades are placed so a single
    # cycle can't commit more than the wallet holds.
    running_bankroll = live_bankroll

    for flagged_market, report in zip(top_flagged, reports):
        if report is None:
            continue   # research failed for this market — skip (kept aligned above)
        question = flagged_market.market.question

        # Skip any market we already hold a position in — on the exchange OR in
        # the local DB. Prevents duplicate/repeat trades on the same market.
        mkt = flagged_market.market
        if held_slugs and (mkt.slug in held_slugs or mkt.condition_id in held_slugs):
            console.print(f"  → [dim]{question[:60]} — already held, skipping[/dim]")
            continue

        # Stop opening positions once this cycle has spent down the wallet.
        if running_bankroll is not None and running_bankroll < RESERVE_FLOOR:
            console.print(
                "  [yellow]→ Available balance spent for this cycle — "
                "stopping new trades.[/yellow]"
            )
            break

        console.print(f"  → [dim]{question[:70]}[/dim]")

        # Step 3: Prediction
        prediction = await predict_market(flagged_market, report)
        if prediction is None:
            console.print(f"    [dim]↳ No trade signal[/dim]")
            continue

        # Attach research features so risk_agent can save them for ML retraining
        s = report.sentiment
        prediction._sentiment_compound  = s.compound
        prediction._sentiment_positive  = s.positive
        prediction._sentiment_negative  = s.negative
        prediction._post_count          = s.post_count
        prediction._avg_engagement      = s.avg_engagement
        prediction._whale_bid_imbalance = report.whale_bid_imbalance
        prediction._trend_score         = report.trend_score

        console.print(
            f"    [cyan]Signal:[/cyan] {prediction.side.value} "
            f"| calibrated={prediction.calibrated_yes_probability:.3f} "
            f"| edge={prediction.edge:+.3f} "
            f"| conf={prediction.confidence:.2f}"
        )

        # Step 4: Risk + Execution (sized off remaining available balance)
        try:
            decision = await evaluate_and_trade(
                flagged_market, prediction, bankroll_usdc=running_bankroll, dry_run=dry_run
            )
        except RuntimeError as e:
            # Circuit breaker (or a hard risk stop) raised — pause cleanly for
            # the rest of this cycle instead of crashing the whole run.
            console.print(f"  [red]⏸ Trading halted: {e}[/red]")
            break

        if decision.approved:
            sz = decision.sizing
            console.print(
                f"    [green]✓ Trade placed[/green] — "
                f"${sz.bet_usdc:.2f} USDC "
                f"(Kelly={sz.kelly_fraction_used:.3f}, "
                f"odds={sz.odds:.2f}x)"
            )
            # Deduct committed capital from the cycle's running balance.
            if running_bankroll is not None:
                running_bankroll = max(0.0, running_bankroll - sz.bet_usdc)
            trades_placed += 1
            if max_trades and trades_placed >= max_trades:
                console.print(
                    f"    [yellow]→ Reached max_trades={max_trades} — "
                    f"stopping after this trade[/yellow]"
                )
                break
        else:
            console.print(
                f"    [red]✗ Blocked:[/red] {decision.rejection_reason}"
            )

    console.print(
        f"\n  ✓ Cycle done — [green]{trades_placed}[/green] trades placed "
        f"in {(datetime.utcnow() - cycle_start).seconds}s"
    )

    # ── Milestone check ───────────────────────────────────────────────────────
    try:
        from utils.notifications import check_and_notify_milestone
        stats = get_trade_stats()
        bankroll_now = settings.bankroll_usdc + stats.get("total_pnl_usdc", 0)
        check_and_notify_milestone(bankroll_now)
    except Exception as e:
        logger.debug("Milestone check failed (non-blocking): %s", e)

    # ── Step 5: Postmortems for any settled losses ────────────────────────────
    await _run_pending_postmortems()


async def _report_unrealized_pnl() -> float:
    """
    Mark every open (PENDING) position to the current market price and print the
    unrealized P&L, so the true live position is visible each cycle instead of
    the $0 the ledger shows until a market actually resolves. Returns the total
    unrealized P&L in USDC.
    """
    exchange = settings.live_exchange.lower()
    if exchange not in ("polymarket_us", "polymarketus", "pmus"):
        return 0.0
    from core.database import SessionLocal, TradeRow
    from integrations.polymarket_us import get_current_price

    with SessionLocal() as s:
        open_pos = [
            (r.id, r.market_id, r.side, r.shares, r.bet_usdc, r.entry_price)
            for r in s.query(TradeRow).filter(TradeRow.outcome == "PENDING").all()
        ]
    if not open_pos:
        return 0.0

    total_cost = 0.0
    total_value = 0.0
    priced = 0
    rows = []   # (unrealized_pct, slug, entry, cur, pnl)
    for tid, slug, side_str, shares, bet, entry in open_pos:
        bet = bet or 0.0
        entry = entry or 0.0   # guard null entry_price (would crash formatting/pct)
        total_cost += bet
        cur = await get_current_price(slug, side_str)
        if cur is None:
            total_value += bet   # unknown → assume flat (no info)
            rows.append((0.0, slug, entry, None, 0.0))
            continue
        val = shares * cur
        total_value += val
        priced += 1
        pnl = val - bet
        pct = ((cur - entry) / entry * 100.0) if entry else 0.0
        rows.append((pct, slug, entry, cur, pnl))

    unrealized = total_value - total_cost
    color = "green" if unrealized >= 0 else "red"
    console.print(
        f"  [dim]Open positions: {len(open_pos)} | cost ${total_cost:.2f} | "
        f"mark ${total_value:.2f} | [/dim][{color}]unrealized ${unrealized:+.2f}[/{color}]"
        f"[dim] ({priced}/{len(open_pos)} priced)[/dim]"
    )
    # Per-position breakdown, worst-first, so losers are obvious every run.
    for pct, slug, entry, cur, pnl in sorted(rows, key=lambda r: r[0]):
        c = "green" if pnl >= 0 else "red"
        cur_s = f"{cur:.3f}" if cur is not None else "n/a"
        console.print(
            f"      [{c}]{pnl:+.2f}[/{c}] [dim]({pct:+.0f}%)  {slug[:44]}  "
            f"entry {entry:.3f} → {cur_s}[/dim]"
        )
    return unrealized


async def _reconcile_exchange_positions() -> int:
    """
    Detect positions you CLOSED yourself on Polymarket (present in our DB as
    PENDING but absent from your live account) and settle them, so the bot
    learns from the outcome instead of tracking a position you no longer hold.

    SAFETY: only runs when the exchange returns a NON-EMPTY position set — if the
    positions endpoint can't be read (empty), we cannot tell "you sold it" from
    "couldn't read it," so we skip rather than wrongly close everything.
    """
    exchange = settings.live_exchange.lower()
    if exchange not in ("polymarket_us", "polymarketus", "pmus"):
        return 0
    from integrations.polymarket_us import get_open_positions, get_current_price
    from core.database import SessionLocal, TradeRow, update_trade_outcome
    from core.models import TradeOutcome

    held = await get_open_positions()
    if not held:
        return 0  # can't safely distinguish a manual close from an unreadable API

    with SessionLocal() as s:
        pending = [
            (r.id, r.market_id, r.side, r.shares, r.bet_usdc, r.entry_price)
            for r in s.query(TradeRow).filter(TradeRow.outcome == "PENDING").all()
        ]

    reconciled = 0
    for tid, slug, side, shares, bet, entry in pending:
        if slug in held:
            continue  # still open on the exchange
        # Gone from the live account → closed off-exchange. Estimate the outcome
        # from the current market price (best proxy for the sale price).
        cur = await get_current_price(slug, side)
        if cur is None:
            continue  # can't estimate → leave it pending for next time
        pnl = shares * cur - (bet or 0.0)
        outcome = TradeOutcome.WIN if pnl >= 0 else TradeOutcome.LOSS
        update_trade_outcome(tid, outcome, pnl)
        logger.info("Reconciled off-exchange close: trade %d (%s) → %s pnl=$%.2f",
                    tid, slug, outcome.value, pnl)
        reconciled += 1
    return reconciled


async def _settle_pending_trades() -> int:
    """
    Check every PENDING trade against the exchange and mark WIN/LOSS when its
    market has resolved. WITHOUT this, trades sit PENDING forever and the bot
    never learns from real outcomes. Returns how many settled this pass.
    """
    from core.database import SessionLocal, TradeRow, update_trade_outcome
    from core.models import TradeOutcome

    exchange = settings.live_exchange.lower()
    if exchange not in ("polymarket_us", "polymarketus", "pmus"):
        return 0  # settlement checkers for other exchanges live in their modules
    from integrations.polymarket_us import check_settlement

    with SessionLocal() as s:
        pending = [
            (r.id, r.market_id, r.side, r.shares, r.bet_usdc)
            for r in s.query(TradeRow).filter(TradeRow.outcome == "PENDING").all()
        ]

    settled = 0
    for tid, slug, side, shares, bet in pending:
        try:
            result = await check_settlement(slug)
        except Exception:
            result = None
        if not result:
            continue
        won = (result == "yes" and side == "YES") or (result == "no" and side == "NO")
        outcome = TradeOutcome.WIN if won else TradeOutcome.LOSS
        pnl = (shares - bet) if won else -bet
        update_trade_outcome(tid, outcome, pnl)
        settled += 1
    return settled


async def _manage_open_positions(dry_run: bool = True) -> int:
    """
    Decide BUY/SELL/HOLD on every open position each cycle (the bot doesn't
    just buy-and-pray). Checks the current market price vs entry and EXITS
    early when a position falls past the stop-loss or rises past the
    take-profit. Returns how many positions it exited.
    """
    from core.database import SessionLocal, TradeRow, update_trade_outcome
    from core.models import TradeOutcome, MarketSide

    exchange = settings.live_exchange.lower()
    if exchange not in ("polymarket_us", "polymarketus", "pmus"):
        return 0
    from integrations.polymarket_us import get_current_price, close_position

    with SessionLocal() as s:
        open_pos = [
            (r.id, r.market_id, r.side, r.shares, r.bet_usdc, r.entry_price)
            for r in s.query(TradeRow).filter(TradeRow.outcome == "PENDING").all()
        ]

    exited = 0
    for tid, slug, side_str, shares, bet, entry in open_pos:
        side = MarketSide.YES if side_str == "YES" else MarketSide.NO
        cur = await get_current_price(slug, side_str)
        if cur is None or not entry or entry <= 0:
            continue  # can't price it → HOLD
        change = (cur - entry) / entry        # +ve = winning, -ve = losing
        # HOLD unless it crosses a stop-loss or take-profit band
        if change <= -settings.stop_loss_pct:
            reason, outcome = "stop-loss", TradeOutcome.LOSS
        elif change >= settings.take_profit_pct:
            reason, outcome = "take-profit", TradeOutcome.WIN
        else:
            # Mid-flight review: a deep unrealized loss (past -30% but not yet
            # stopped out at -stop_loss_pct) triggers an EARLY, one-time lesson so
            # the agents start learning from a clearly-bad bet before it settles.
            if -settings.stop_loss_pct < change <= -0.30:
                try:
                    from agents.postmortem_agent import run_midflight_review
                    await run_midflight_review(tid, cur, change * 100)
                except Exception as e:
                    logger.debug("Mid-flight review failed (non-blocking): %s", e)
            continue  # HOLD
        ok = await close_position(slug, side, shares, cur, dry_run=dry_run)
        if ok:
            # realised pnl at the current price (shares valued at `cur`)
            pnl = shares * cur - bet
            update_trade_outcome(tid, outcome, pnl)
            logger.info("Exited trade %d on %s (%.0f%% move) pnl=$%.2f",
                        tid, reason, change * 100, pnl)
            exited += 1
    return exited


async def _run_pending_postmortems() -> None:
    """
    Analyze recently settled trades the system hasn't studied yet — BOTH wins
    and losses. Losses → run_postmortem (why it failed). Wins → run_winmortem
    (what worked, so we repeat it). The agents learn from both outcomes.
    """
    from core.database import SessionLocal, TradeRow, PostmortemRow
    from agents.postmortem_agent import run_winmortem

    try:
        with SessionLocal() as session:
            analyzed_ids = session.query(PostmortemRow.trade_id).distinct()
            unanalyzed = (
                session.query(TradeRow)
                .filter(
                    TradeRow.outcome.in_(["WIN", "LOSS"]),
                    TradeRow.id.not_in(analyzed_ids),
                )
                .limit(8)
                .all()
            )

        if not unanalyzed:
            return

        n_win = sum(1 for r in unanalyzed if r.outcome == "WIN")
        n_loss = sum(1 for r in unanalyzed if r.outcome == "LOSS")
        console.print(
            f"\n[bold]Step 5[/bold] Learning from "
            f"[green]{n_win} wins[/green] + [red]{n_loss} losses[/red]..."
        )

        from core.models import Trade, MarketSide, TradeStatus, TradeOutcome
        for row in unanalyzed:
            trade = Trade(
                id=row.id,
                market_id=row.market_id,
                question=row.question or "",
                side=MarketSide(row.side),
                entry_price=row.entry_price,
                bet_usdc=row.bet_usdc,
                shares=row.shares,
                status=TradeStatus(row.status),
                outcome=TradeOutcome(row.outcome),
                pnl_usdc=row.pnl_usdc,
                tx_hash=row.tx_hash or "",
                placed_at=row.placed_at,
                settled_at=row.settled_at,
                notes=row.notes or "{}",
            )
            if trade.outcome == TradeOutcome.WIN:
                report = await run_winmortem(trade)
                tag = "[green]✓ Win analyzed[/green]"
            else:
                report = await run_postmortem(trade)
                tag = "[red]✓ Loss analyzed[/red]"
            if report:
                console.print(f"  {tag} — {len(report.findings)} findings")

    except Exception as e:
        logger.error("Learning sweep failed: %s", e)


# ── Retrain command ────────────────────────────────────────────────────────────

def retrain_model() -> None:
    console.print("[bold]Retraining XGBoost calibrator from trade history...[/bold]")
    records = calibrator.collect_training_data()
    metrics = calibrator.train(records)
    console.print(f"  Result: {metrics}")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main_loop(dry_run: bool, interval_seconds: int, max_trades: int = None) -> None:
    """Run the pipeline continuously at the specified interval."""
    _print_banner()
    init_db()

    if not settings.anthropic_api_key:
        console.print(
            "[yellow]⚠  ANTHROPIC_API_KEY not set — running in demo mode "
            "(rule-based predictions, no LLM calls)[/yellow]"
        )

    _last_daily_summary_date = None

    while True:
        try:
            await run_pipeline(dry_run=dry_run, max_trades=max_trades)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — exiting.[/yellow]")
            break
        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)

        # Daily summary email at ~9pm UTC
        now = datetime.now(timezone.utc)
        today = now.date()
        if now.hour >= 21 and _last_daily_summary_date != today:
            _last_daily_summary_date = today
            try:
                from utils.notifications import send_daily_summary
                stats = get_trade_stats()
                send_daily_summary({
                    "win_rate":       stats.get("win_rate", 0),
                    "total_pnl_usdc": stats.get("total_pnl_usdc", 0),
                    "total":          stats.get("total", 0),
                    "wins":           stats.get("wins", 0),
                    "losses":         stats.get("losses", 0),
                    "pending":        stats.get("pending", 0),
                })
            except Exception as e:
                logger.debug("Daily summary failed (non-blocking): %s", e)

        console.print(
            f"\n[dim]Next scan in {interval_seconds}s "
            f"({interval_seconds // 60}m)...[/dim]\n"
        )
        await asyncio.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prediction Market Trading Bot")
    p.add_argument(
        "--run-once", action="store_true",
        help="Run a single pipeline cycle then exit",
    )
    p.add_argument(
        "--live", action="store_true",
        help="Enable LIVE trading (spends real USDC — use with caution!)",
    )
    p.add_argument(
        "--retrain", action="store_true",
        help="Retrain the XGBoost calibrator from trade history then exit",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print performance statistics then exit",
    )
    p.add_argument(
        "--interval", type=int, default=settings.scan_interval_seconds,
        help=f"Scan interval in seconds (default: {settings.scan_interval_seconds})",
    )
    p.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top-priority markets to research per cycle (default: 10)",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Demo mode: run full pipeline with rule-based predictions (no API key needed)",
    )
    p.add_argument(
        "--cycles", type=int, default=1,
        help="Number of back-to-back pipeline cycles to run (paper trading only)",
    )
    p.add_argument(
        "--paper-blast", action="store_true",
        help="Relaxed thresholds for fast paper trade accumulation (min_confidence=0.60, top-n=20)",
    )
    p.add_argument(
        "--test-trade", action="store_true",
        help="Place exactly ONE small (~$1-2) real trade to verify the live "
             "order pipeline works end-to-end. Use with --live.",
    )
    p.add_argument(
        "--daemon", action="store_true",
        help="Run forever in background, logging to bot.log (use with nohup or screen)",
    )
    p.add_argument(
        "--max-trades", type=int, default=None,
        help="Cap how many trades to place per cycle (safety limit for live runs)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_db()

    purged = purge_stale_lessons()
    if purged:
        logger.info("Purged %d stale placeholder lessons from DB", purged)

    if args.stats:
        _print_banner()
        _print_stats()
        sys.exit(0)

    if args.retrain:
        retrain_model()
        sys.exit(0)

    dry_run = not args.live
    if args.live:
        console.print(
            Panel(
                "[bold red]⚠  LIVE TRADING MODE[/bold red]\n"
                "Real USDC will be spent on Polygon mainnet.\n"
                "Ensure POLYMARKET_PRIVATE_KEY is set and funded.\n"
                "Press Ctrl+C within 5 seconds to abort...",
                border_style="red",
            )
        )
        import time
        time.sleep(5)

    # Mock markets are ONLY for explicit --demo. Live and normal runs always
    # scan REAL Polymarket markets. Without an Anthropic key the prediction
    # agent falls back to XGBoost + rule-based (no LLM) — but on real markets.
    use_mock = args.demo and not args.live
    if use_mock:
        # Lower confidence threshold for demo so we can see trade signals fire
        settings.min_confidence = 0.51
    if args.live and not settings.anthropic_api_key:
        console.print(
            "[yellow]⚠  No ANTHROPIC_API_KEY — predictions use XGBoost + "
            "rule-based only (no LLM layer). Scanning REAL markets.[/yellow]"
        )

    if args.paper_blast:
        if args.live:
            console.print("[red]--paper-blast cannot be used with --live[/red]")
            sys.exit(1)
        settings.min_confidence = 0.60
        settings.min_edge = 0.03
        args.top_n = max(args.top_n, 20)
        console.print(Panel(
            "[bold yellow]📄 Paper Blast Mode[/bold yellow]\n"
            "Relaxed thresholds for fast paper trade accumulation.\n"
            f"min_confidence=0.60  min_edge=0.03  top_n={args.top_n}\n"
            "[dim]These settings are paper-only and reset on next run.[/dim]",
            border_style="yellow",
        ))

    if args.test_trade:
        # One small REAL trade to verify the live order pipeline end-to-end.
        # Relax the gates so something qualifies, and shrink the effective
        # bankroll so Kelly sizes a tiny (~$1-2) bet — risk is trivial.
        settings.min_confidence = 0.50
        settings.min_edge = 0.01
        settings.bankroll_usdc = 20.0      # 10% max-bet → ~$2 cap
        settings.max_bet_fraction = 0.10
        settings.max_time_to_resolution_days = 400   # allow long-dated US futures
        args.top_n = max(args.top_n, 25)   # widen the net to find a candidate
        console.print(Panel(
            "[bold magenta]🧪 TEST TRADE MODE[/bold magenta]\n"
            "Places ONE small (~$1-2) real trade to verify live execution.\n"
            "This is a pipeline smoke test, NOT a strategy trade.\n"
            f"[dim]min_confidence=0.50  effective bankroll=$20 → ~$2 max bet[/dim]",
            border_style="magenta",
        ))

    if args.daemon:
        # Redirect all logging to bot.log for background operation
        file_handler = logging.FileHandler("bot.log")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)
        console.print(
            "[bold green]🤖 Bot started in daemon mode[/bold green] — "
            "logging to [cyan]bot.log[/cyan]\n"
            "  Stop with: [dim]kill $(cat bot.pid)[/dim]\n"
            f"  Exchange: [cyan]{settings.live_exchange}[/cyan] | "
            f"Bankroll: [cyan]${settings.bankroll_usdc:,.0f}[/cyan] | "
            f"Live: [{'red' if not dry_run else 'dim'}]{'YES' if not dry_run else 'NO (paper)'}[/{'red' if not dry_run else 'dim'}]"
        )
        import os
        with open("bot.pid", "w") as f:
            f.write(str(os.getpid()))

    if args.test_trade:
        _print_banner()
        asyncio.run(run_pipeline(dry_run=dry_run, top_n=args.top_n,
                                 use_mock=use_mock, max_trades=1))
        _print_stats()
    elif args.run_once or args.demo or args.cycles > 1 or args.paper_blast:
        _print_banner()
        cycles = args.cycles if not args.run_once else 1
        for i in range(cycles):
            if cycles > 1:
                console.rule(f"[cyan]Cycle {i+1} / {cycles}[/cyan]")
            asyncio.run(run_pipeline(dry_run=dry_run, top_n=args.top_n,
                                     use_mock=use_mock, max_trades=args.max_trades))
        _print_stats()
    else:
        asyncio.run(main_loop(dry_run=dry_run, interval_seconds=args.interval,
                              max_trades=args.max_trades))
