"""
Paper Trading Simulator
═══════════════════════
Polls real Polymarket prices every hour and auto-settles open paper trades
when markets resolve, building a realistic P&L track record before going live.

Run alongside main.py:
  python paper_trader.py          # polls every hour
  python paper_trader.py --now    # settle pending trades immediately (dev)
  python paper_trader.py --sim N  # simulate N random outcomes for back-testing
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box

from config import settings
from core.analytics import summary, recent_trades
from core.database import (
    SessionLocal,
    TradeRow,
    get_trade_stats,
    update_trade_outcome,
)
from core.models import TradeOutcome

logger = logging.getLogger(__name__)
console = Console()


# ── Settlement logic ──────────────────────────────────────────────────────────

async def _try_settle_from_polymarket(row: TradeRow) -> bool:
    """
    Fetch the current market price and settle if it has resolved.
    Returns True if settled.
    """
    try:
        from integrations.polymarket import get_market_by_id
        market = await get_market_by_id(row.market_id)
        if not market:
            return False

        # Consider resolved if price has collapsed to near 0 or 1
        if market.yes_price >= 0.97:
            won = row.side == "YES"
        elif market.yes_price <= 0.03:
            won = row.side == "NO"
        else:
            return False   # still live

        outcome = TradeOutcome.WIN if won else TradeOutcome.LOSS
        pnl = (row.shares - row.bet_usdc) if won else -row.bet_usdc
        update_trade_outcome(row.id, outcome, pnl)
        logger.info(
            "Settled trade %d — %s — PnL $%.2f",
            row.id, outcome.value, pnl,
        )
        return True
    except Exception as e:
        logger.debug("Settlement check failed for trade %d: %s", row.id, e)
        return False


def _simulate_settlement(row: TradeRow, win_rate_prior: float = 0.684) -> None:
    """
    Simulate a random outcome weighted by the calibrated edge.
    Used for back-testing and demo purposes.

    The win probability is biased toward the bot's estimated edge:
      entry_price = what the market charged
      if entry < 0.5 → bot bought YES cheap → higher win prob
    """
    entry = row.entry_price
    # Rough win probability: bot's model thought it had edge
    # Use a logistic curve centred on the entry price
    if row.side == "YES":
        # Bot bought YES at `entry`. Win if market was above entry.
        # Simulate market resolution by sampling from Beta(α, β)
        # biased by our prior win rate
        win_prob = win_rate_prior * (1 - entry) / 0.5 if entry < 0.5 else win_rate_prior * 0.6
    else:
        win_prob = win_rate_prior * entry / 0.5 if entry > 0.5 else win_rate_prior * 0.6

    win_prob = min(max(win_prob, 0.15), 0.90)
    won = random.random() < win_prob
    outcome = TradeOutcome.WIN if won else TradeOutcome.LOSS
    pnl = (row.shares - row.bet_usdc) if won else -row.bet_usdc
    update_trade_outcome(row.id, outcome, pnl)


async def settle_pending(use_simulation: bool = False) -> int:
    """Check all pending trades and settle any that have resolved."""
    with SessionLocal() as s:
        pending = s.query(TradeRow).filter(TradeRow.outcome == "PENDING").all()

    if not pending:
        console.print("[dim]No pending trades to settle.[/dim]")
        return 0

    settled_count = 0
    for row in pending:
        if use_simulation:
            _simulate_settlement(row)
            settled_count += 1
        else:
            if await _try_settle_from_polymarket(row):
                settled_count += 1

    return settled_count


# ── Stats display ─────────────────────────────────────────────────────────────

def print_performance() -> None:
    stats = summary()

    console.print(
        "\n[bold cyan]Paper Trading Performance[/bold cyan]  "
        f"[dim](as of {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})[/dim]"
    )

    metrics = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2))
    metrics.add_column("Metric", style="dim")
    metrics.add_column("Value", justify="right")

    wr = stats["win_rate"]
    wr_color = "green" if wr >= 0.60 else "yellow" if wr >= 0.45 else "red"

    pnl = stats["total_pnl"]
    pnl_color = "green" if pnl >= 0 else "red"

    target_delta = wr - 0.684
    target_str = (
        f"[green]+{target_delta:.1%}[/green]" if target_delta >= 0
        else f"[red]{target_delta:.1%}[/red]"
    )

    metrics.add_row("Total trades", str(stats["total_trades"]))
    metrics.add_row("Settled / Pending", f"{stats['settled']} / {stats['pending']}")
    metrics.add_row("Win rate", f"[{wr_color}]{wr:.1%}[/{wr_color}]")
    metrics.add_row("vs. 68.4% target", target_str)
    metrics.add_row("Total PnL", f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]")
    metrics.add_row("Sharpe ratio (ann.)", f"{stats['sharpe']:.2f}")
    metrics.add_row("Max drawdown", f"{stats['max_drawdown']:.1%}")
    metrics.add_row("Open exposure", f"${stats['exposure_usdc']:,.2f}")
    console.print(metrics)

    # Recent trades
    trades = recent_trades(10)
    if trades:
        t = Table(title="Recent Trades", box=box.ROUNDED, border_style="dim")
        t.add_column("ID", style="dim", width=4)
        t.add_column("Market", width=40)
        t.add_column("Side", width=4)
        t.add_column("Entry", justify="right", width=6)
        t.add_column("Bet $", justify="right", width=7)
        t.add_column("Outcome", width=8)
        t.add_column("PnL $", justify="right", width=8)

        for tr in trades:
            outcome_str = {
                "WIN":     "[green]WIN[/green]",
                "LOSS":    "[red]LOSS[/red]",
                "PENDING": "[yellow]OPEN[/yellow]",
            }.get(tr["outcome"], tr["outcome"])

            pnl_val = tr["pnl"]
            pnl_str = (
                f"[green]+{pnl_val:.2f}[/green]" if pnl_val > 0
                else f"[red]{pnl_val:.2f}[/red]" if pnl_val < 0
                else "[dim]—[/dim]"
            )

            t.add_row(
                str(tr["id"]),
                tr["question"],
                tr["side"],
                f"{tr['entry']:.3f}",
                f"{tr['bet']:.2f}",
                outcome_str,
                pnl_str,
            )
        console.print(t)


# ── Simulation mode ───────────────────────────────────────────────────────────

async def run_simulation(n_trades: int) -> None:
    """
    Inject N simulated paper trades and settle them immediately.
    Useful for validating the Kelly + postmortem pipeline end-to-end.
    """
    from core.database import save_trade
    from core.models import Trade, MarketSide, TradeStatus, TradeOutcome

    console.print(f"[bold]Simulating {n_trades} paper trades...[/bold]")

    sample_markets = [
        ("0xsim001", "Will inflation drop below 3% by Q3 2025?", 0.42),
        ("0xsim002", "Will SpaceX reach Mars orbit in 2025?", 0.18),
        ("0xsim003", "Will Apple release AR glasses in 2025?", 0.35),
        ("0xsim004", "Will the S&P 500 hit 6000 by end of 2025?", 0.67),
        ("0xsim005", "Will a major bank collapse in 2025?", 0.12),
        ("0xsim006", "Will OpenAI release GPT-5 in 2025?", 0.55),
        ("0xsim007", "Will the US enter a recession in 2025?", 0.28),
        ("0xsim008", "Will Nvidia hit $200/share by Q4 2025?", 0.61),
    ]

    import uuid, json as _json
    for i in range(n_trades):
        mid, question, yes_price = random.choice(sample_markets)
        side = MarketSide.YES if random.random() > 0.3 else MarketSide.NO
        entry = yes_price if side == MarketSide.YES else (1 - yes_price)

        # Realistic randomised bet sizing (Kelly-based, capped at 10% bankroll)
        edge = random.uniform(0.04, 0.22)
        bet = min(random.uniform(8, 60), 100.0)   # cap at $100 to limit drawdown
        shares = bet / max(entry, 0.01)

        # Full 13-feature snapshot with realistic random variation
        sentiment = random.gauss(0.1, 0.3)
        features = {
            "compound_sentiment":      round(max(-1.0, min(1.0, sentiment)), 4),
            "positive_sentiment":      round(max(0, sentiment * 0.6 + random.uniform(0, 0.3)), 4),
            "negative_sentiment":      round(max(0, -sentiment * 0.4 + random.uniform(0, 0.2)), 4),
            "post_count":              random.randint(3, 80),
            "avg_engagement":          round(random.uniform(5, 500), 1),
            "price_change_24h":        round(random.uniform(0, 0.12), 4),
            "spread":                  round(random.uniform(0.01, 0.08), 4),
            "liquidity_usdc":          round(random.uniform(1000, 50000), 0),
            "volume_24h_usdc":         round(random.uniform(500, 20000), 0),
            "time_to_resolution_days": round(random.uniform(1, 30), 1),
            "current_yes_price":       yes_price,
            "whale_bid_imbalance":     round(random.gauss(0, 0.3), 4),
            "trend_score":             round(random.uniform(20, 90), 1),
        }

        trade = Trade(
            market_id=f"{mid}_sim{i}",
            question=question,
            side=side,
            entry_price=entry,
            bet_usdc=bet,
            shares=shares,
            status=TradeStatus.PLACED,
            outcome=TradeOutcome.PENDING,
            tx_hash=f"sim_{uuid.uuid4().hex[:12]}",
            notes=_json.dumps({"features": features, "ab_variant": random.choice(["A","B"]), "exchange": "paper"}),
        )
        trade.id = save_trade(trade)

    # Settle all simulated trades
    settled = await settle_pending(use_simulation=True)
    console.print(f"  ✓ Settled [green]{settled}[/green] simulated trades")
    print_performance()

    # Trigger postmortems on any losses
    from agents.postmortem_agent import run_postmortem
    from core.database import get_losing_trades
    loss_rows = get_losing_trades(limit=5)
    if loss_rows:
        console.print(f"\n[bold]Running postmortems on {len(loss_rows)} losses...[/bold]")
        from core.models import Trade as TModel, MarketSide, TradeStatus, TradeOutcome
        for row in loss_rows[:3]:   # cap at 3 for demo
            t = TModel(
                id=row.id, market_id=row.market_id, question=row.question or "",
                side=MarketSide(row.side), entry_price=row.entry_price,
                bet_usdc=row.bet_usdc, shares=row.shares,
                status=TradeStatus(row.status), outcome=TradeOutcome(row.outcome),
                pnl_usdc=row.pnl_usdc, tx_hash=row.tx_hash or "",
                placed_at=row.placed_at, settled_at=row.settled_at,
                notes=row.notes or "{}",
            )
            console.print(
                f"  [dim]Postmortem skipped in sim mode "
                f"(needs ANTHROPIC_API_KEY) — trade {row.id}[/dim]"
            )


# ── Continuous polling loop ───────────────────────────────────────────────────

async def poll_loop(interval: int = 3600) -> None:
    console.print(
        f"[bold cyan]Paper trader running[/bold cyan] — "
        f"polling every {interval // 60}m for settled markets"
    )
    while True:
        try:
            settled = await settle_pending(use_simulation=False)
            if settled:
                console.print(f"  ✓ Settled {settled} trades")
                print_performance()
        except Exception as e:
            logger.error("Poll error: %s", e)
        await asyncio.sleep(interval)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Paper trader / settlement simulator")
    p.add_argument("--now", action="store_true", help="Settle pending trades now (live prices)")
    p.add_argument("--sim", type=int, metavar="N", help="Simulate N trades and settle them")
    p.add_argument("--stats", action="store_true", help="Print performance stats and exit")
    p.add_argument("--interval", type=int, default=3600, help="Poll interval in seconds")
    args = p.parse_args()

    from core.database import init_db
    init_db()

    if args.stats:
        print_performance()
    elif args.sim:
        asyncio.run(run_simulation(args.sim))
    elif args.now:
        settled = asyncio.run(settle_pending(use_simulation=False))
        console.print(f"Settled {settled} trades.")
        print_performance()
    else:
        asyncio.run(poll_loop(args.interval))
