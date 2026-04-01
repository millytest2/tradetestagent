"""Streamlit dashboard for the stock trading bot."""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from core.database import init_db, get_trade_stats
from core.analytics import (
    bankroll_series,
    max_drawdown,
    recent_trades,
    sharpe_ratio,
    summary,
    win_rate_by_side,
    avg_edge,
    exposure_usd,
)

init_db()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Stock Trading Bot")
st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ── Circuit breaker banner ─────────────────────────────────────────────────────
if os.path.exists("TRADING_PAUSED.txt"):
    with open("TRADING_PAUSED.txt") as f:
        cb_msg = f.read().strip()
    st.error(
        f"🚨 **CIRCUIT BREAKER ACTIVE — ALL TRADING PAUSED**\n\n{cb_msg}\n\n"
        "_Delete `TRADING_PAUSED.txt` to resume trading._"
    )
else:
    st.success("✅ Circuit breaker: OK — trading active")

st.divider()

# ── Top KPI metrics ────────────────────────────────────────────────────────────
stats = get_trade_stats()
s = summary()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Trades", stats["total"])
col2.metric("Win Rate", f"{stats['win_rate']:.1%}", f"{stats['wins']}W / {stats['losses']}L")
col3.metric(
    "Total PnL",
    f"${s['total_pnl']:+,.2f}",
    f"${s['total_pnl']:+.2f}",
    delta_color="normal",
)
col4.metric("Sharpe Ratio", f"{sharpe_ratio():.2f}")
col5.metric("Max Drawdown", f"{max_drawdown():.1%}")
col6.metric("Open Exposure", f"${exposure_usd():,.2f}")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Cumulative PnL")
    series = bankroll_series(initial=settings.bankroll_usd)
    if series:
        df_pnl = pd.DataFrame(series, columns=["date", "bankroll"])
        df_pnl["date"] = pd.to_datetime(df_pnl["date"])
        st.line_chart(df_pnl.set_index("date")["bankroll"])
    else:
        st.info("No settled trades yet.")

with col_right:
    st.subheader("Win Rate by Side")
    side_wr = win_rate_by_side()
    df_side = pd.DataFrame([
        {"Side": "LONG", "Win Rate": side_wr.get("LONG", 0)},
        {"Side": "SHORT", "Win Rate": side_wr.get("SHORT", 0)},
    ])
    st.bar_chart(df_side.set_index("Side"))

st.divider()

# ── Open positions panel ────────────────────────────────────────────────────────
st.subheader("Open Positions")
from core.database import get_open_positions
open_rows = get_open_positions()
if open_rows:
    open_data = []
    for r in open_rows:
        open_data.append({
            "ID": r.id,
            "Ticker": f"${r.ticker}",
            "Side": r.side,
            "Entry $": f"${r.entry_price:.2f}",
            "Bet $": f"${r.bet_usd:.2f}",
            "Stop Loss": f"${r.stop_loss_price:.2f}",
            "Take Profit": f"${r.take_profit_price:.2f}",
            "Entry Date": r.entry_date.strftime("%m-%d %H:%M") if r.entry_date else "",
        })
    st.dataframe(pd.DataFrame(open_data), use_container_width=True)
else:
    st.info("No open positions.")

st.divider()

# ── Recent trades ──────────────────────────────────────────────────────────────
st.subheader("Recent Trades (last 20)")
trades = recent_trades(n=20)
if trades:
    df_trades = pd.DataFrame(trades)
    # Color outcome column
    def _color_outcome(val):
        if val == "WIN":
            return "background-color: #1a4a1a"
        elif val == "LOSS":
            return "background-color: #4a1a1a"
        return ""

    def _color_pnl(val):
        try:
            v = float(str(val).replace("$", "").replace(",", ""))
            return "color: #44ff44" if v > 0 else "color: #ff4444" if v < 0 else ""
        except Exception:
            return ""

    styled = df_trades.style.applymap(_color_outcome, subset=["outcome"]).applymap(
        _color_pnl, subset=["pnl"]
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.info("No trades yet.")

st.divider()

# ── Performance stats ──────────────────────────────────────────────────────────
st.subheader("Performance Summary")
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Avg Edge per Trade", f"{avg_edge():+.3f}")
    st.metric("Pending trades", s["pending"])
with col_b:
    st.metric("Settled (WIN+LOSS)", s["settled"])
    st.metric("Total PnL", f"${s['total_pnl']:+,.2f}")

# ── Refresh button ─────────────────────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.rerun()
