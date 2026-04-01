"""
Stock Trading Bot — Main Orchestrator
══════════════════════════════════════

Five-step pipeline:
  1. Scan Agent      — screen S&P 500, flag technical signals
  2. Research Agents — parallel Reddit / RSS / Trends + VADER sentiment
  3. Prediction Agent — XGBoost + Claude calibration
  4. Risk Agent      — Kelly sizing, bracket order via Alpaca
  5. Postmortem      — 5 agents analyze every loss, update system

Usage
─────
  # Paper trading (dry run — no real money)
  python main.py

  # Paper trading, 5 cycles back-to-back
  python main.py --cycles 5

  # Live trading (real money via Alpaca)
  python main.py --live

  # Retrain XGBoost on settled history
  python main.py --retrain

  # Show performance stats
  python main.py --stats

  # Settle open positions (run after market close)
  python settle.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import settings
from core.database import (
    get_open_positions,
    get_trade_stats,
    init_db,
    purge_stale_lessons,
)
from agents.scan_agent import scan_stocks
from agents.research_agent import research_stocks_parallel
from agents.prediction_agent import predict_stock
from agents.risk_agent import evaluate_and_trade
from ml.calibrator import calibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
console = Console()


def _print_banner(live: bool = False) -> None:
    mode = "[bold red]LIVE TRADING[/bold red]" if live else "[bold yellow]PAPER TRADING[/bold yellow]"
    console.print(Panel.fit(
        f"[bold cyan]Stock Trading Bot[/bold cyan]\n"
        f"[dim]Scan → Research → Predict → Risk → Postmortem[/dim]\n"
        f"[dim]Model: {settings.llm_model} | "
        f"Bankroll: ${settings.bankroll_usd:,.0f} | "
        f"Kelly: {settings.kelly_fraction:.0%}[/dim]\n"
        f"Mode: {mode}",
        border_style="cyan",
    ))


def _print_stats() -> None:
    stats = get_trade_stats()
    from core.analytics import sharpe_ratio, max_drawdown, win_rate_by_side, avg_edge

    table = Table(title="Performance", box=box.ROUNDED, border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total trades", str(stats["total"]))
    table.add_row("Wins", f"[green]{stats['wins']}[/green]")
    table.add_row("Losses", f"[red]{stats['losses']}[/red]")
    table.add_row("Pending", str(stats["pending"]))

    wr = stats["win_rate"]
    wc = "green" if wr >= 0.60 else "yellow" if wr >= 0.45 else "red"
    table.add_row("Win rate", f"[{wc}]{wr:.1%}[/{wc}]")

    pnl = stats["total_pnl_usd"]
    pc = "green" if pnl >= 0 else "red"
    table.add_row("Total PnL", f"[{pc}]${pnl:+,.2f}[/{pc}]")
    table.add_row("Sharpe ratio", f"{sharpe_ratio():.2f}")
    table.add_row("Max drawdown", f"{max_drawdown():.1%}")

    side_wr = win_rate_by_side()
    table.add_row("Win rate LONG", f"{side_wr.get('LONG', 0):.1%}")
    table.add_row("Win rate SHORT", f"{side_wr.get('SHORT', 0):.1%}")
    table.add_row("Avg edge/trade", f"{avg_edge():+.3f}")

    console.print(table)


async def run_cycle(dry_run: bool = True, top_n: int = 5) -> None:
    """Run one full pipeline cycle."""
    console.print(f"\n[dim]── Cycle at {datetime.utcnow().strftime('%H:%M:%S UTC')} ──[/dim]")

    # Check circuit breaker
    if os.path.exists("TRADING_PAUSED.txt"):
        console.print("[bold red]🚨 CIRCUIT BREAKER ACTIVE — TRADING PAUSED[/bold red]")
        console.print("   Delete TRADING_PAUSED.txt to resume.")
        return

    # Step 1: Scan
    flagged = await scan_stocks()
    if not flagged:
        console.print("[yellow]No flagged stocks — nothing to trade.[/yellow]")
        return

    console.print(f"[cyan]Scan:[/cyan] {len(flagged)} flagged stocks")

    # Step 2: Research top N
    top_flagged = flagged[:top_n]
    researched = await research_stocks_parallel(top_flagged)
    console.print(f"[cyan]Research:[/cyan] completed {len(researched)} stocks")

    trades_placed = 0
    for flagged_stock, report in researched:
        ticker = flagged_stock.stock.ticker

        # Step 3: Predict
        prediction = await predict_stock(flagged_stock, report)
        if prediction is None or not prediction.should_trade:
            logger.info("$%s: PASS (no trade signal)", ticker)
            continue

        # Step 4: Risk + Execute
        try:
            from integrations.alpaca import get_bankroll
            bankroll = await get_bankroll()
        except Exception:
            bankroll = settings.bankroll_usd

        decision = await evaluate_and_trade(
            flagged_stock, prediction, bankroll_usd=bankroll, dry_run=dry_run
        )

        if decision.approved:
            console.print(
                f"[green]✓ TRADE[/green] ${ticker} "
                f"{prediction.side.value} | "
                f"edge={prediction.edge:.3f} conf={prediction.confidence:.2f} "
                f"bet=${decision.sizing.bet_usd:.2f}"
            )
            trades_placed += 1
        else:
            console.print(
                f"[dim]✗ BLOCKED[/dim] ${ticker}: {decision.rejection_reason}"
            )

    console.print(f"\n[bold]Cycle complete — {trades_placed} trades placed.[/bold]")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    parser.add_argument("--live", action="store_true", help="Live trading (real money)")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0 = infinite)")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default 300)")
    parser.add_argument("--retrain", action="store_true", help="Retrain XGBoost model")
    parser.add_argument("--stats", action="store_true", help="Show performance stats")
    parser.add_argument("--top-n", type=int, default=5, help="Research top N flagged stocks per cycle")
    args = parser.parse_args()

    init_db()
    purge_stale_lessons()
    _print_banner(live=args.live)

    if args.stats:
        _print_stats()
        return

    if args.retrain:
        console.print("[cyan]Retraining XGBoost calibrator...[/cyan]")
        success = calibrator.train()
        if success:
            console.print("[green]✓ Model retrained successfully.[/green]")
        else:
            console.print("[red]✗ Retraining failed — not enough data yet.[/red]")
        return

    dry_run = not args.live
    if args.live:
        console.print(
            "\n[bold red]⚠ LIVE TRADING MODE — real money will be used.[/bold red]\n"
            "Press Ctrl+C within 5 seconds to cancel..."
        )
        await asyncio.sleep(5)

    cycle_num = 0
    while True:
        cycle_num += 1
        try:
            await run_cycle(dry_run=dry_run, top_n=args.top_n)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user.[/yellow]")
            break
        except Exception as e:
            logger.error("Cycle %d failed: %s", cycle_num, e)

        if args.cycles and cycle_num >= args.cycles:
            break

        if args.cycles == 0 or cycle_num < args.cycles:
            console.print(f"[dim]Sleeping {args.interval}s until next cycle...[/dim]")
            try:
                await asyncio.sleep(args.interval)
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped by user.[/yellow]")
                break

    _print_stats()


if __name__ == "__main__":
    asyncio.run(main())
