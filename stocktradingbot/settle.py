"""
settle.py — Check open stock positions and mark wins/losses.

Run once a day (or after market close) to settle trades that hit
their stop-loss or take-profit, or have been held past holding_days.

Usage:
  python settle.py
  python settle.py --force   # settle all pending regardless of age
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box

from config import settings
from core.database import (
    get_open_positions,
    init_db,
    update_trade_outcome,
)
from core.models import Trade, TradeOutcome, TradeStatus, StockSide
from integrations.alpaca import check_trade_outcome, get_current_price

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("settle")
console = Console()


def _orm_to_trade(row) -> Trade:
    from core.models import TradeStatus, TradeOutcome
    return Trade(
        id=row.id,
        ticker=row.ticker,
        company_name=row.company_name or "",
        side=StockSide(row.side),
        entry_price=row.entry_price,
        bet_usd=row.bet_usd,
        shares=row.shares or 0.0,
        stop_loss_price=row.stop_loss_price or 0.0,
        take_profit_price=row.take_profit_price or 0.0,
        status=TradeStatus(row.status),
        outcome=TradeOutcome(row.outcome),
        pnl_usd=row.pnl_usd or 0.0,
        alpaca_order_id=row.alpaca_order_id or "",
        entry_date=row.entry_date,
        exit_date=row.exit_date,
        notes=row.notes or "",
    )


async def settle_all(force: bool = False) -> None:
    init_db()
    open_rows = get_open_positions()

    if not open_rows:
        console.print("[green]No open positions to settle.[/green]")
        return

    table = Table(title=f"Settling {len(open_rows)} open positions", box=box.ROUNDED)
    table.add_column("ID")
    table.add_column("Ticker")
    table.add_column("Side")
    table.add_column("Entry $")
    table.add_column("Current $")
    table.add_column("SL")
    table.add_column("TP")
    table.add_column("Outcome")
    table.add_column("PnL")

    settled_count = 0
    now = datetime.now(timezone.utc)

    for row in open_rows:
        trade = _orm_to_trade(row)

        # Check if holding period expired
        days_held = (now - trade.entry_date.replace(tzinfo=timezone.utc)).days if trade.entry_date else 0
        timed_out = days_held >= settings.holding_days

        current_price = await get_current_price(trade.ticker)
        if current_price is None:
            logger.warning("Could not fetch price for $%s", trade.ticker)
            continue

        outcome, pnl = await check_trade_outcome(trade)

        # Force settle if timed out
        if outcome == TradeOutcome.PENDING and (timed_out or force):
            if trade.side == StockSide.LONG:
                pnl = (current_price - trade.entry_price) * trade.shares
            else:
                pnl = (trade.entry_price - current_price) * trade.shares
            outcome = TradeOutcome.WIN if pnl > 0 else TradeOutcome.LOSS
            logger.info("Time-expiry settle: $%s after %d days", trade.ticker, days_held)

        outcome_str = outcome.value
        pnl_str = f"${pnl:+.2f}"
        if outcome == TradeOutcome.WIN:
            outcome_color = "[green]WIN[/green]"
            pnl_color = f"[green]{pnl_str}[/green]"
        elif outcome == TradeOutcome.LOSS:
            outcome_color = "[red]LOSS[/red]"
            pnl_color = f"[red]{pnl_str}[/red]"
        else:
            outcome_color = "[dim]PENDING[/dim]"
            pnl_color = "[dim]—[/dim]"

        table.add_row(
            str(trade.id),
            f"${trade.ticker}",
            trade.side.value,
            f"${trade.entry_price:.2f}",
            f"${current_price:.2f}",
            f"${trade.stop_loss_price:.2f}",
            f"${trade.take_profit_price:.2f}",
            outcome_color,
            pnl_color,
        )

        if outcome != TradeOutcome.PENDING:
            update_trade_outcome(
                trade_id=trade.id,
                outcome=outcome,
                pnl_usd=round(pnl, 2),
            )
            settled_count += 1

            # Run postmortem on losses
            if outcome == TradeOutcome.LOSS:
                try:
                    from agents.postmortem_agent import run_postmortem
                    trade.outcome = outcome
                    trade.pnl_usd = pnl
                    trade.exit_date = now
                    asyncio.create_task(run_postmortem(trade))
                    logger.info("Postmortem queued for $%s (trade #%d)", trade.ticker, trade.id)
                except Exception as e:
                    logger.warning("Could not queue postmortem: %s", e)

    console.print(table)
    console.print(f"\n[bold]Settled {settled_count} / {len(open_rows)} positions.[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settle open stock positions")
    parser.add_argument("--force", action="store_true", help="Force settle all pending trades")
    args = parser.parse_args()
    asyncio.run(settle_all(force=args.force))
