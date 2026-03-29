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
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import settings
from core.database import (
    get_losing_trades,
    get_trade_stats,
    init_db,
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

async def run_pipeline(dry_run: bool = True, top_n: int = 10, use_mock: bool = False) -> None:
    """Execute one full scan→research→predict→risk cycle."""

    cycle_start = datetime.utcnow()
    console.rule(f"[cyan]Cycle started {cycle_start.strftime('%H:%M:%S UTC')}[/cyan]")

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

    # ── Steps 3 & 4: Predict + Risk ────────────────────────────────────────────
    console.print("[bold]Step 3+4[/bold] Predicting and evaluating risk...")
    trades_placed = 0

    for flagged_market, report in zip(top_flagged, reports):
        question = flagged_market.market.question
        console.print(f"  → [dim]{question[:70]}[/dim]")

        # Step 3: Prediction
        prediction = await predict_market(flagged_market, report)
        if prediction is None:
            console.print(f"    [dim]↳ No trade signal[/dim]")
            continue

        console.print(
            f"    [cyan]Signal:[/cyan] {prediction.side.value} "
            f"| calibrated={prediction.calibrated_yes_probability:.3f} "
            f"| edge={prediction.edge:+.3f} "
            f"| conf={prediction.confidence:.2f}"
        )

        # Step 4: Risk + Execution
        decision = await evaluate_and_trade(
            flagged_market, prediction, dry_run=dry_run
        )

        if decision.approved:
            sz = decision.sizing
            console.print(
                f"    [green]✓ Trade placed[/green] — "
                f"${sz.bet_usdc:.2f} USDC "
                f"(Kelly={sz.kelly_fraction_used:.3f}, "
                f"odds={sz.odds:.2f}x)"
            )
            trades_placed += 1
        else:
            console.print(
                f"    [red]✗ Blocked:[/red] {decision.rejection_reason}"
            )

    console.print(
        f"\n  ✓ Cycle done — [green]{trades_placed}[/green] trades placed "
        f"in {(datetime.utcnow() - cycle_start).seconds}s"
    )

    # ── Step 5: Postmortems for any settled losses ────────────────────────────
    await _run_pending_postmortems()


async def _run_pending_postmortems() -> None:
    """Run postmortems on any recently settled losing trades that need analysis."""
    from core.database import SessionLocal, TradeRow, PostmortemRow
    from sqlalchemy import func

    try:
        with SessionLocal() as session:
            # Find losses that don't yet have a postmortem
            subq = session.query(PostmortemRow.trade_id).distinct().subquery()
            analyzed_ids = session.query(PostmortemRow.trade_id).distinct()
            unanalyzed = (
                session.query(TradeRow)
                .filter(
                    TradeRow.outcome == "LOSS",
                    TradeRow.id.not_in(analyzed_ids),
                )
                .limit(5)
                .all()
            )

        if not unanalyzed:
            return

        console.print(
            f"\n[bold]Step 5[/bold] Running postmortems on "
            f"[red]{len(unanalyzed)}[/red] unanalyzed losses..."
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
            report = await run_postmortem(trade)
            if report:
                console.print(
                    f"  ✓ Postmortem done — "
                    f"{len(report.findings)} findings, "
                    f"{len(report.system_updates)} system updates"
                )

    except Exception as e:
        logger.error("Postmortem sweep failed: %s", e)


# ── Retrain command ────────────────────────────────────────────────────────────

def retrain_model() -> None:
    console.print("[bold]Retraining XGBoost calibrator from trade history...[/bold]")
    records = calibrator.collect_training_data()
    metrics = calibrator.train(records)
    console.print(f"  Result: {metrics}")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main_loop(dry_run: bool, interval_seconds: int) -> None:
    """Run the pipeline continuously at the specified interval."""
    _print_banner()
    init_db()

    if not settings.anthropic_api_key:
        console.print(
            "[yellow]⚠  ANTHROPIC_API_KEY not set — running in demo mode "
            "(rule-based predictions, no LLM calls)[/yellow]"
        )

    while True:
        try:
            await run_pipeline(dry_run=dry_run)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — exiting.[/yellow]")
            break
        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)

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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_db()

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

    use_mock = args.demo or not settings.anthropic_api_key
    if use_mock:
        # Lower confidence threshold for demo so we can see trade signals fire
        settings.min_confidence = 0.51

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

    if args.run_once or args.demo or args.cycles > 1 or args.paper_blast:
        _print_banner()
        cycles = args.cycles if not args.run_once else 1
        for i in range(cycles):
            if cycles > 1:
                console.rule(f"[cyan]Cycle {i+1} / {cycles}[/cyan]")
            asyncio.run(run_pipeline(dry_run=dry_run, top_n=args.top_n, use_mock=use_mock))
        _print_stats()
    else:
        asyncio.run(main_loop(dry_run=dry_run, interval_seconds=args.interval))
