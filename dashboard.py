"""
Live Trading Dashboard
══════════════════════
Run with:  streamlit run dashboard.py

Auto-refreshes every 5 seconds from the SQLite database.
Works in paper trading mode AND live mode — same dashboard.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.database import init_db, SessionLocal, TradeRow, PostmortemRow, LessonRow
from core.analytics import (
    summary,
    pnl_series,
    bankroll_series,
    win_rate_by_side,
    recent_trades,
)
from config import settings

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prediction Market Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ── Auto-refresh ──────────────────────────────────────────────────────────────

REFRESH_INTERVAL = 5   # seconds

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Config")
    st.metric("Bankroll", f"${settings.bankroll_usdc:,.0f} USDC")
    st.metric("Min confidence", f"{settings.min_confidence:.0%}")
    st.metric("Min edge", f"{settings.min_edge:.0%}")
    st.metric("Kelly fraction", f"{settings.kelly_fraction:.0%}")
    st.metric("Max bet", f"{settings.max_bet_fraction:.0%} of bankroll")

    st.divider()
    api_status = "🟢 Connected" if settings.anthropic_api_key else "🔴 No API key (demo mode)"
    pm_status = "🟢 Live trading" if settings.polymarket_private_key else "🟡 Paper trading"
    st.write(f"**LLM:** {api_status}")
    st.write(f"**Polymarket:** {pm_status}")

    st.divider()
    st.caption(f"Last refresh: {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    if st.button("🔄 Refresh now"):
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📈 Prediction Market Trading Bot")
st.caption("Scan → Research → Predict → Risk → Postmortem")

# ── Stats row ─────────────────────────────────────────────────────────────────

stats = summary()
c1, c2, c3, c4, c5, c6 = st.columns(6)

wr = stats["win_rate"]
wr_delta = wr - 0.684
c1.metric(
    "Win Rate",
    f"{wr:.1%}",
    delta=f"{wr_delta:+.1%} vs 68.4% target",
    delta_color="normal",
)
c2.metric(
    "Total PnL",
    f"${stats['total_pnl']:+,.2f}",
    delta_color="normal",
)
c3.metric("Sharpe (ann.)", f"{stats['sharpe']:.2f}")
c4.metric("Max Drawdown", f"{stats['max_drawdown']:.1%}")
c5.metric(
    "Trades",
    f"{stats['total_trades']}",
    delta=f"{stats['pending']} open",
    delta_color="off",
)
c6.metric("Open Exposure", f"${stats['exposure_usdc']:,.2f}")

st.divider()

# ── PnL Chart ─────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Cumulative PnL")

    series = bankroll_series(initial=settings.bankroll_usdc)
    if series:
        dts = [dt for dt, _ in series]
        vals = [v for _, v in series]
        color = "green" if vals[-1] >= settings.bankroll_usdc else "red"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dts, y=vals,
            mode="lines+markers",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'0,200,100' if color == 'green' else '220,50,50'},0.08)",
            name="Bankroll",
        ))
        fig.add_hline(
            y=settings.bankroll_usdc, line_dash="dash",
            line_color="gray", annotation_text="Starting bankroll",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=300,
            yaxis_title="USDC",
            xaxis_title=None,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No settled trades yet — run some paper trades to see the curve.")

with col_right:
    st.subheader("🎯 Win Rate by Side")

    wr_by_side = win_rate_by_side()
    if any(wr_by_side.values()):
        fig2 = go.Figure(go.Bar(
            x=list(wr_by_side.keys()),
            y=[v * 100 for v in wr_by_side.values()],
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{v:.1%}" for v in wr_by_side.values()],
            textposition="outside",
        ))
        fig2.add_hline(y=68.4, line_dash="dot", line_color="gold",
                       annotation_text="68.4% target")
        fig2.update_layout(
            height=300,
            yaxis=dict(range=[0, 100], title="Win %"),
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No settled trades yet.")

# ── Trade table ────────────────────────────────────────────────────────────────

st.subheader("📋 Recent Trades")

trades = recent_trades(20)
if trades:
    df = pd.DataFrame(trades)
    df["pnl"] = df["pnl"].map(lambda x: f"+${x:.2f}" if x > 0 else (f"-${abs(x):.2f}" if x < 0 else "—"))
    df["entry"] = df["entry"].map(lambda x: f"{x:.3f}")
    df["bet"] = df["bet"].map(lambda x: f"${x:.2f}")
    df.columns = ["ID", "Market", "Side", "Entry", "Bet", "Outcome", "PnL", "Placed"]

    def color_outcome(val):
        if val == "WIN":
            return "color: #2ecc71"
        elif val == "LOSS":
            return "color: #e74c3c"
        elif val == "PENDING":
            return "color: #f39c12"
        return ""

    st.dataframe(
        df.style.map(color_outcome, subset=["Outcome"]),
        use_container_width=True,
        height=350,
        hide_index=True,
    )
else:
    st.info("No trades in the database yet. Run `python main.py --demo` to generate paper trades.")

# ── Postmortem insights ────────────────────────────────────────────────────────

st.divider()
col_pm, col_lessons = st.columns(2)

with col_pm:
    st.subheader("🔬 Postmortem Findings")
    try:
        with SessionLocal() as s:
            rows = (
                s.query(PostmortemRow)
                .order_by(PostmortemRow.created_at.desc())
                .limit(10)
                .all()
            )
        if rows:
            for r in rows:
                severity_color = {
                    "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"
                }.get(r.severity, "⚪")
                with st.expander(f"{severity_color} [{r.agent_name}] Trade #{r.trade_id}"):
                    st.write(f"**Finding:** {r.finding}")
                    st.write(f"**Root cause:** {r.root_cause}")
                    st.write(f"**Fix:** {r.recommendation}")
        else:
            st.info("No postmortems yet — losses trigger automatic analysis.")
    except Exception as e:
        st.error(f"Error loading postmortems: {e}")

with col_lessons:
    st.subheader("💡 System Lessons")
    try:
        with SessionLocal() as s:
            lessons = (
                s.query(LessonRow)
                .filter(LessonRow.active == True)
                .order_by(LessonRow.created_at.desc())
                .limit(15)
                .all()
            )
        if lessons:
            for l in lessons:
                st.markdown(f"- **[{l.category}]** {l.lesson}")
        else:
            st.info("Lessons accumulate as postmortems run after losses.")
    except Exception as e:
        st.error(f"Error loading lessons: {e}")

# ── Market scan preview ────────────────────────────────────────────────────────

st.divider()
st.subheader("🔭 Market Scanner — Last Cycle")

try:
    with SessionLocal() as s:
        recent = (
            s.query(TradeRow)
            .order_by(TradeRow.placed_at.desc())
            .limit(5)
            .all()
        )
    if recent:
        scan_data = []
        for r in recent:
            scan_data.append({
                "Market": (r.question or "")[:65],
                "Side": r.side,
                "Entry price": f"{r.entry_price:.3f}",
                "Bet USDC": f"${r.bet_usdc:.2f}",
                "Status": r.outcome,
                "Tx": r.tx_hash[:16] + "…" if r.tx_hash and len(r.tx_hash) > 16 else r.tx_hash,
            })
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True, hide_index=True)
except Exception as e:
    st.error(str(e))

# ── Auto-refresh ──────────────────────────────────────────────────────────────

time.sleep(REFRESH_INTERVAL)
st.rerun()
