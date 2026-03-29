"""
200-Trade Review — Deep Performance Analysis
════════════════════════════════════════════
Run after accumulating settled trades to understand what's working
and what to tune before going live.

Usage:
  python review.py              # full analysis (needs 30+ settled trades)
  python review.py --threshold 50  # use a custom minimum trade count
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import math
import argparse
from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule

from core.database import SessionLocal, TradeRow, PostmortemRow, LessonRow, init_db
from config import settings

console = Console()
init_db()

PAPER_TRADE_TARGET = 200


# ── Data loading ───────────────────────────────────────────────────────────────

def _settled() -> list[TradeRow]:
    with SessionLocal() as s:
        return (
            s.query(TradeRow)
            .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
            .order_by(TradeRow.placed_at)
            .all()
        )

def _all_trades() -> list[TradeRow]:
    with SessionLocal() as s:
        return s.query(TradeRow).order_by(TradeRow.placed_at).all()

def _postmortems() -> list[PostmortemRow]:
    with SessionLocal() as s:
        return s.query(PostmortemRow).all()

def _lessons() -> list[LessonRow]:
    with SessionLocal() as s:
        return s.query(LessonRow).filter(LessonRow.active == True).all()


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _win_rate(rows: list[TradeRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.outcome == "WIN") / len(rows)

def _pnl(rows: list[TradeRow]) -> float:
    return sum(r.pnl_usdc for r in rows)

def _sharpe(rows: list[TradeRow]) -> float:
    if len(rows) < 2:
        return 0.0
    pnls = [r.pnl_usdc for r in rows]
    mean = sum(pnls) / len(pnls)
    std = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1))
    return (mean / std * math.sqrt(365)) if std > 0 else 0.0

def _max_dd(rows: list[TradeRow]) -> float:
    if not rows:
        return 0.0
    denominator = max(settings.bankroll_usdc, 1.0)
    cumulative, peak_pnl, max_dd = 0.0, 0.0, 0.0
    for r in rows:
        cumulative += r.pnl_usdc
        if cumulative > peak_pnl:
            peak_pnl = cumulative
        dd = (peak_pnl - cumulative) / denominator
        if dd > max_dd:
            max_dd = dd
    return max_dd

def _color(val: float, good: float, warn: float) -> str:
    """green if >= good, yellow if >= warn, else red."""
    if val >= good:
        return "green"
    if val >= warn:
        return "yellow"
    return "red"


# ── Section printers ───────────────────────────────────────────────────────────

def _section_overview(rows: list[TradeRow], all_rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]1. Overall Performance[/bold cyan]")
    total = len(all_rows)
    settled = len(rows)
    pending = total - settled
    wr = _win_rate(rows)
    pnl = _pnl(rows)
    sharpe = _sharpe(rows)
    dd = _max_dd(rows)
    roi = pnl / settings.bankroll_usdc if settings.bankroll_usdc else 0.0

    t = Table(box=box.SIMPLE_HEAD, border_style="cyan", show_header=False)
    t.add_column("Metric", style="bold", width=28)
    t.add_column("Value", justify="right", width=16)
    t.add_column("Notes", style="dim", width=40)

    wr_c = _color(wr, 0.65, 0.55)
    t.add_row("Settled trades",   f"{settled}",
              f"({pending} still pending, {total} total)")
    t.add_row("Win rate",         f"[{wr_c}]{wr:.1%}[/{wr_c}]",
              "Target ≥65% for go-live")
    t.add_row("Total PnL",        f"[{'green' if pnl>=0 else 'red'}]${pnl:+,.2f}[/{'green' if pnl>=0 else 'red'}]",
              f"ROI: {roi:+.1%} on ${settings.bankroll_usdc:,.0f} bankroll")
    t.add_row("Annualised Sharpe",f"{sharpe:.2f}",
              ">1.5 is strong, >2.0 is excellent")
    t.add_row("Max drawdown",     f"[{'green' if dd<0.1 else 'yellow' if dd<0.2 else 'red'}]{dd:.1%}[/{'green' if dd<0.1 else 'yellow' if dd<0.2 else 'red'}]",
              "<10% comfortable, >20% high risk")
    t.add_row("Current bankroll", f"${settings.bankroll_usdc + pnl:,.2f}",
              f"Goal: ${settings.bankroll_usdc * 2:,.0f} (2x)")

    console.print(t)

    progress_pct = min(100, int(settled / PAPER_TRADE_TARGET * 100))
    bar_filled = int(progress_pct / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    console.print(
        f"\n  Paper trade progress: [{bar}] {settled}/{PAPER_TRADE_TARGET} "
        f"({progress_pct}%)\n"
    )


def _section_by_side(rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]2. Win Rate by Side (YES vs NO)[/bold cyan]")

    t = Table(box=box.SIMPLE_HEAD, border_style="cyan")
    t.add_column("Side",   style="bold", width=8)
    t.add_column("Trades", justify="right", width=8)
    t.add_column("Wins",   justify="right", width=8)
    t.add_column("Losses", justify="right", width=8)
    t.add_column("Win %",  justify="right", width=10)
    t.add_column("PnL",    justify="right", width=12)
    t.add_column("Avg bet",justify="right", width=10)

    for side in ["YES", "NO"]:
        subset = [r for r in rows if r.side == side]
        if not subset:
            t.add_row(side, "0", "—", "—", "—", "—", "—")
            continue
        wr = _win_rate(subset)
        pnl = _pnl(subset)
        avg_bet = sum(r.bet_usdc for r in subset) / len(subset)
        wins = sum(1 for r in subset if r.outcome == "WIN")
        losses = len(subset) - wins
        c = _color(wr, 0.65, 0.50)
        t.add_row(
            side,
            str(len(subset)),
            f"[green]{wins}[/green]",
            f"[red]{losses}[/red]",
            f"[{c}]{wr:.1%}[/{c}]",
            f"[{'green' if pnl>=0 else 'red'}]${pnl:+.2f}[/{'green' if pnl>=0 else 'red'}]",
            f"${avg_bet:.2f}",
        )

    console.print(t)
    console.print(
        "  [dim]If one side is consistently losing, consider raising min_edge "
        "for that side or disabling it temporarily.[/dim]\n"
    )


def _section_by_price(rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]3. Win Rate by Entry Price Bucket[/bold cyan]")

    buckets: dict[str, list[TradeRow]] = {
        "0.03–0.15 (long-shot YES / near-certain NO)": [],
        "0.15–0.30": [],
        "0.30–0.45": [],
        "0.45–0.55 (coin-flip)": [],
        "0.55–0.70": [],
        "0.70–0.85": [],
        "0.85–0.97 (near-certain YES / long-shot NO)": [],
    }

    def _bucket(price: float) -> str:
        if price < 0.15:  return "0.03–0.15 (long-shot YES / near-certain NO)"
        if price < 0.30:  return "0.15–0.30"
        if price < 0.45:  return "0.30–0.45"
        if price < 0.55:  return "0.45–0.55 (coin-flip)"
        if price < 0.70:  return "0.55–0.70"
        if price < 0.85:  return "0.70–0.85"
        return "0.85–0.97 (near-certain YES / long-shot NO)"

    for r in rows:
        buckets[_bucket(r.entry_price)].append(r)

    t = Table(box=box.SIMPLE_HEAD, border_style="cyan")
    t.add_column("Price range",  style="bold", width=44)
    t.add_column("Trades", justify="right", width=8)
    t.add_column("Win %",  justify="right", width=10)
    t.add_column("PnL",    justify="right", width=12)

    for label, subset in buckets.items():
        if not subset:
            t.add_row(label, "0", "—", "—")
            continue
        wr = _win_rate(subset)
        pnl = _pnl(subset)
        c = _color(wr, 0.65, 0.50)
        t.add_row(
            label,
            str(len(subset)),
            f"[{c}]{wr:.1%}[/{c}]",
            f"[{'green' if pnl>=0 else 'red'}]${pnl:+.2f}[/{'green' if pnl>=0 else 'red'}]",
        )

    console.print(t)
    console.print(
        "  [dim]Avoid buckets with <5 trades (too small to conclude anything). "
        "Focus on the profitable buckets when going live.[/dim]\n"
    )


def _section_by_confidence(rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]4. Win Rate by Confidence Band (Calibration Check)[/bold cyan]")

    # Pull confidence from notes JSON
    bands: dict[str, list[tuple[TradeRow, float]]] = {
        "0.65–0.70": [],
        "0.70–0.75": [],
        "0.75–0.80": [],
        "0.80–0.85": [],
        "0.85–1.00": [],
        "unknown":   [],
    }

    def _band(conf: float) -> str:
        if conf < 0.70: return "0.65–0.70"
        if conf < 0.75: return "0.70–0.75"
        if conf < 0.80: return "0.75–0.80"
        if conf < 0.85: return "0.80–0.85"
        return "0.85–1.00"

    for r in rows:
        try:
            notes = json.loads(r.notes or "{}")
            conf = float(notes.get("features", {}).get("confidence", notes.get("confidence", -1)))
        except Exception:
            conf = -1
        if conf < 0:
            bands["unknown"].append((r, conf))
        else:
            bands[_band(conf)].append((r, conf))

    t = Table(box=box.SIMPLE_HEAD, border_style="cyan")
    t.add_column("Confidence band", style="bold", width=18)
    t.add_column("Trades", justify="right", width=8)
    t.add_column("Win %",  justify="right", width=10)
    t.add_column("Avg conf", justify="right", width=10)
    t.add_column("Calibrated?", width=20)

    for label, items in bands.items():
        if not items:
            t.add_row(label, "0", "—", "—", "—")
            continue
        subset = [r for r, _ in items]
        confs = [c for _, c in items if c >= 0]
        wr = _win_rate(subset)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        c = _color(wr, 0.65, 0.50)
        # Calibrated = win rate roughly equals avg confidence
        gap = abs(wr - avg_conf) if avg_conf > 0 else 0
        cal = "[green]✓ well-calibrated[/green]" if gap < 0.10 else "[yellow]±slightly off[/yellow]" if gap < 0.20 else "[red]⚠ overconfident[/red]"
        t.add_row(
            label,
            str(len(subset)),
            f"[{c}]{wr:.1%}[/{c}]",
            f"{avg_conf:.2f}" if avg_conf > 0 else "—",
            cal if avg_conf > 0 else "—",
        )

    console.print(t)
    console.print(
        "  [dim]Ideal: win rate ≈ confidence. If you win 62% in the 0.70–0.75 band, "
        "the model is well-calibrated. If win rate is much lower, raise min_confidence.[/dim]\n"
    )


def _section_by_time(rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]5. Win Rate by Time to Resolution[/bold cyan]")

    # Reconstruct approximate days-to-resolution from notes
    # (stored as features.time_to_resolution_days)
    buckets: dict[str, list[TradeRow]] = {
        "1–3 days (imminent)":  [],
        "4–7 days (1 week)":    [],
        "8–14 days (2 weeks)":  [],
        "15–30 days (1 month)": [],
        "unknown":              [],
    }

    def _ttr_bucket(days: float) -> str:
        if days <= 3:   return "1–3 days (imminent)"
        if days <= 7:   return "4–7 days (1 week)"
        if days <= 14:  return "8–14 days (2 weeks)"
        return "15–30 days (1 month)"

    for r in rows:
        try:
            notes = json.loads(r.notes or "{}")
            days = float(notes.get("features", {}).get("time_to_resolution_days", -1))
        except Exception:
            days = -1
        if days < 0:
            buckets["unknown"].append(r)
        else:
            buckets[_ttr_bucket(days)].append(r)

    t = Table(box=box.SIMPLE_HEAD, border_style="cyan")
    t.add_column("Time to resolution", style="bold", width=22)
    t.add_column("Trades", justify="right", width=8)
    t.add_column("Win %",  justify="right", width=10)
    t.add_column("PnL",    justify="right", width=12)

    for label, subset in buckets.items():
        if not subset:
            t.add_row(label, "0", "—", "—")
            continue
        wr = _win_rate(subset)
        pnl = _pnl(subset)
        c = _color(wr, 0.65, 0.50)
        t.add_row(
            label,
            str(len(subset)),
            f"[{c}]{wr:.1%}[/{c}]",
            f"[{'green' if pnl>=0 else 'red'}]${pnl:+.2f}[/{'green' if pnl>=0 else 'red'}]",
        )

    console.print(t)


def _section_worst_trades(rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]6. Worst 10 Trades (Learn From These)[/bold cyan]")

    losses = [r for r in rows if r.outcome == "LOSS"]
    losses.sort(key=lambda r: r.pnl_usdc)
    worst = losses[:10]

    if not worst:
        console.print("  [green]No losses yet![/green]\n")
        return

    t = Table(box=box.SIMPLE_HEAD, border_style="cyan")
    t.add_column("ID",  width=5)
    t.add_column("Market",  width=50, no_wrap=True)
    t.add_column("Side", width=5)
    t.add_column("Entry", justify="right", width=7)
    t.add_column("Bet",   justify="right", width=8)
    t.add_column("Loss",  justify="right", width=9)

    for r in worst:
        t.add_row(
            str(r.id),
            (r.question or "—")[:48],
            r.side,
            f"{r.entry_price:.3f}",
            f"${r.bet_usdc:.2f}",
            f"[red]${r.pnl_usdc:.2f}[/red]",
        )

    console.print(t)
    console.print()


def _section_postmortem_patterns(postmortems: list[PostmortemRow]) -> None:
    console.rule("[bold cyan]7. Postmortem Patterns (Most Common Root Causes)[/bold cyan]")

    if not postmortems:
        console.print("  [dim]No postmortems yet — they run automatically after every loss.[/dim]\n")
        return

    # Count root causes
    cause_counts: dict[str, int] = defaultdict(int)
    agent_severity: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pm in postmortems:
        cause = (pm.root_cause or "Unknown")[:60]
        cause_counts[cause] += 1
        agent_severity[pm.agent_name][pm.severity] += 1

    # Top root causes
    t = Table(title="Top Root Causes", box=box.SIMPLE_HEAD, border_style="cyan")
    t.add_column("Root cause", style="bold", width=60)
    t.add_column("Count", justify="right", width=8)

    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1])[:10]:
        t.add_row(cause, str(count))
    console.print(t)

    # Agent severity breakdown
    t2 = Table(title="Severity by Agent", box=box.SIMPLE_HEAD, border_style="cyan")
    t2.add_column("Agent", style="bold", width=24)
    t2.add_column("Critical", justify="right", width=10)
    t2.add_column("High",     justify="right", width=8)
    t2.add_column("Medium",   justify="right", width=8)
    t2.add_column("Low",      justify="right", width=8)

    for agent, sevs in sorted(agent_severity.items()):
        t2.add_row(
            agent,
            f"[red]{sevs.get('critical',0)}[/red]",
            f"[orange3]{sevs.get('high',0)}[/orange3]",
            f"[yellow]{sevs.get('medium',0)}[/yellow]",
            f"[green]{sevs.get('low',0)}[/green]",
        )
    console.print(t2)
    console.print()


def _section_lessons(lessons: list[LessonRow]) -> None:
    console.rule("[bold cyan]8. Active System Lessons[/bold cyan]")

    if not lessons:
        console.print("  [dim]No lessons yet.[/dim]\n")
        return

    by_category: dict[str, list[str]] = defaultdict(list)
    for l in lessons:
        by_category[l.category].append(l.lesson)

    for cat, items in sorted(by_category.items()):
        console.print(f"  [bold]{cat}[/bold] ({len(items)} lessons)")
        for item in items[:3]:
            console.print(f"    • {item[:120]}")
        if len(items) > 3:
            console.print(f"    [dim]... and {len(items)-3} more[/dim]")
        console.print()


def _section_recommendations(rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]9. Tuning Recommendations[/bold cyan]")

    if len(rows) < 20:
        console.print("  [dim]Need at least 20 settled trades for reliable recommendations.[/dim]\n")
        return

    wr = _win_rate(rows)
    pnl = _pnl(rows)
    yes_rows = [r for r in rows if r.side == "YES"]
    no_rows  = [r for r in rows if r.side == "NO"]
    wr_yes = _win_rate(yes_rows)
    wr_no  = _win_rate(no_rows)

    recs: list[tuple[str, str]] = []

    # Overall win rate
    if wr >= 0.68:
        recs.append(("✅ Win rate", f"{wr:.1%} — strong. Consider increasing kelly_fraction from {settings.kelly_fraction:.0%} to {min(settings.kelly_fraction+0.05, 0.35):.0%} to grow faster."))
    elif wr >= 0.60:
        recs.append(("⚠️  Win rate", f"{wr:.1%} — decent but needs 65% for go-live. Raise min_confidence slightly."))
    else:
        recs.append(("🔴 Win rate", f"{wr:.1%} — too low for go-live. Raise min_edge (currently {settings.min_edge:.0%}) to filter weaker trades."))

    # Side-specific
    if len(yes_rows) >= 10 and wr_yes < 0.55:
        recs.append(("🔴 YES trades", f"Win rate {wr_yes:.1%} — underperforming. Consider raising min_edge for YES trades or filtering lower-liquidity markets."))
    if len(no_rows) >= 10 and wr_no < 0.55:
        recs.append(("🔴 NO trades", f"Win rate {wr_no:.1%} — underperforming. The model may be systematically overconfident on NO side."))
    if len(no_rows) >= 10 and wr_no > 0.70:
        recs.append(("✅ NO trades", f"Win rate {wr_no:.1%} — excellent. NO trades are your edge — prioritise them."))

    # PnL
    if pnl > 0 and wr >= 0.65:
        recs.append(("✅ Profitability", f"${pnl:+,.2f} positive PnL. Ready to consider real money at this performance level."))
    elif pnl < 0:
        recs.append(("🔴 PnL negative", f"${pnl:+,.2f}. More paper trading needed before going live."))

    # Bet sizing
    avg_bet = sum(r.bet_usdc for r in rows) / len(rows)
    max_bankroll_pct = max(r.bet_usdc for r in rows) / settings.bankroll_usdc
    if max_bankroll_pct > 0.12:
        recs.append(("⚠️  Max bet size", f"Largest single bet was {max_bankroll_pct:.1%} of bankroll — consider lowering max_bet_fraction to 0.08 for tighter risk control."))

    # Retrain recommendation
    if len(rows) >= 50:
        recs.append(("💡 Retrain ML", f"{len(rows)} settled trades available. Run [bold]python main.py --retrain[/bold] to update XGBoost on real data — this is the most impactful improvement."))

    t = Table(box=box.SIMPLE_HEAD, border_style="cyan", show_header=False)
    t.add_column("Tag",   width=22, style="bold")
    t.add_column("Recommendation", width=70)

    for tag, rec in recs:
        t.add_row(tag, rec)

    console.print(t)
    console.print()


def _section_go_live_verdict(rows: list[TradeRow]) -> None:
    console.rule("[bold cyan]10. Go-Live Verdict[/bold cyan]")

    wr = _win_rate(rows)
    pnl = _pnl(rows)
    current_bankroll = settings.bankroll_usdc + pnl
    goal_bankroll = settings.bankroll_usdc * 2
    settled = len(rows)

    c1 = current_bankroll >= goal_bankroll
    c2 = wr >= 0.65
    c3 = settled >= PAPER_TRADE_TARGET

    checks = [
        (c1, f"Bankroll ≥ 2x",    f"${current_bankroll:,.0f} / ${goal_bankroll:,.0f}"),
        (c2, "Win rate ≥ 65%",    f"{wr:.1%} / 65%"),
        (c3, f"{PAPER_TRADE_TARGET}+ settled trades", f"{settled} / {PAPER_TRADE_TARGET}"),
    ]

    all_pass = all(c for c, _, _ in checks)

    for passed, label, detail in checks:
        icon = "[green]✅[/green]" if passed else "[yellow]⏳[/yellow]"
        console.print(f"  {icon}  [bold]{label}[/bold]  —  {detail}")

    console.print()

    if all_pass:
        console.print(Panel(
            "[bold green]🚀 ALL CONDITIONS MET — READY TO GO LIVE[/bold green]\n\n"
            "Next steps:\n"
            "  1. [bold]python main.py --retrain[/bold]  (train XGBoost on your real data)\n"
            "  2. Set [bold]DRY_RUN=false[/bold] in your .env file\n"
            "  3. Set [bold]BANKROLL_USDC=200[/bold] (or your chosen amount)\n"
            "  4. Fund your Polygon wallet with USDC\n"
            "  5. [bold]python main.py --live[/bold]",
            border_style="green",
        ))
    else:
        remaining = []
        if not c1:
            remaining.append(f"  • ${goal_bankroll - current_bankroll:,.0f} more paper PnL to reach 2x")
        if not c2:
            remaining.append(f"  • Win rate needs {0.65 - wr:+.1%} more to hit 65%")
        if not c3:
            remaining.append(f"  • {PAPER_TRADE_TARGET - settled} more settled trades to go")
        console.print(Panel(
            "[bold yellow]⏳ Keep paper trading[/bold yellow]\n\n"
            "Remaining before go-live:\n" + "\n".join(remaining),
            border_style="yellow",
        ))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="200-Trade Deep Review")
    p.add_argument("--threshold", type=int, default=10,
                   help="Minimum settled trades to run analysis (default: 10)")
    args = p.parse_args()

    all_rows = _all_trades()
    rows = _settled()
    postmortems = _postmortems()
    lessons = _lessons()

    console.print(Panel.fit(
        "[bold cyan]TradeBot — Deep Performance Review[/bold cyan]\n"
        f"[dim]{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
        f"{len(rows)} settled / {len(all_rows)} total trades[/dim]",
        border_style="cyan",
    ))

    if len(rows) < args.threshold:
        console.print(
            f"\n[yellow]Only {len(rows)} settled trades — need at least {args.threshold} "
            f"for meaningful analysis. Keep the bot running![/yellow]\n"
        )
        console.print(
            "  Tip: Trades settle when Polymarket resolves the market (could take days).\n"
            "  In paper trading mode, settlements are simulated automatically.\n"
        )
        # Still show go-live verdict so user sees progress
        _section_go_live_verdict(rows)
        return

    _section_overview(rows, all_rows)
    _section_by_side(rows)
    _section_by_price(rows)
    _section_by_confidence(rows)
    _section_by_time(rows)
    _section_worst_trades(rows)
    _section_postmortem_patterns(postmortems)
    _section_lessons(lessons)
    _section_recommendations(rows)
    _section_go_live_verdict(rows)


if __name__ == "__main__":
    main()
