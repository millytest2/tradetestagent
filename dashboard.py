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
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.database import init_db, SessionLocal, TradeRow, PostmortemRow, LessonRow
from core.analytics import (
    summary,
    bankroll_series,
    win_rate_by_side,
    recent_trades,
)
from config import settings

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TradeBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 12px;
}
.stDataFrame { border-radius: 8px; }
.big-win { color: #2ecc71; font-weight: bold; }
.big-loss { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

init_db()
REFRESH_INTERVAL = 5
GO_LIVE_TARGET = settings.bankroll_usdc * 2   # 2x = go live milestone
GO_LIVE_WIN_RATE = 0.65                        # minimum win rate to go live
GO_LIVE_MIN_TRADES = 30                        # minimum settled trades

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### TradeBot")
    st.caption(f"v1.0 · {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    st.divider()

    st.markdown("**Trading Config**")
    st.metric("Bankroll", f"${settings.bankroll_usdc:,.0f}")
    col_a, col_b = st.columns(2)
    col_a.metric("Min conf", f"{settings.min_confidence:.0%}")
    col_b.metric("Min edge", f"{settings.min_edge:.0%}")
    col_a.metric("Kelly", f"{settings.kelly_fraction:.0%}")
    col_b.metric("Max bet", f"{settings.max_bet_fraction:.0%}")

    st.divider()
    st.markdown("**System Status**")

    llm_ok = bool(settings.anthropic_api_key)
    pm_ok = bool(settings.polymarket_private_key)
    dry = getattr(settings, "dry_run", True)

    st.write(f"{'🟢' if llm_ok else '🔴'} LLM: {'**' + settings.llm_model + '**' if llm_ok else 'No API key'}")
    st.write(f"{'🟢' if pm_ok else '🔴'} Wallet: {'Connected' if pm_ok else 'Not set'}")
    st.write(f"{'🟡' if dry else '🟢'} Mode: **{'Paper trading' if dry else 'LIVE'}**")

    st.divider()
    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()
    st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## 📈 Prediction Market Trading Bot")
st.caption("Scan → Research → Predict → Risk → Postmortem")

# ── KPI row ───────────────────────────────────────────────────────────────────

stats = summary()
wr = stats["win_rate"]
wr_delta = wr - 0.684
current_bankroll = settings.bankroll_usdc + stats["total_pnl"]
settled = stats["total_trades"] - stats["pending"]

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

c1.metric("Win Rate", f"{wr:.1%}", delta=f"{wr_delta:+.1%} vs target")
c2.metric("Bankroll", f"${current_bankroll:,.2f}", delta=f"{stats['total_pnl']:+,.2f}")
c3.metric("Total PnL", f"${stats['total_pnl']:+,.2f}")
c4.metric("Sharpe (ann.)", f"{stats['sharpe']:.2f}")
c5.metric("Max Drawdown", f"{stats['max_drawdown']:.1%}")
c6.metric("Settled / Open", f"{settled} / {stats['pending']}")
c7.metric("Exposure", f"${stats['exposure_usdc']:,.2f}")

# ── Go-live progress bar ──────────────────────────────────────────────────────

dry = getattr(settings, "dry_run", True)
if dry:
    st.markdown("---")
    st.markdown("### 🚀 Go-Live Checklist (Paper → Real Money)")

    check1 = current_bankroll >= GO_LIVE_TARGET
    check2 = wr >= GO_LIVE_WIN_RATE
    check3 = settled >= GO_LIVE_MIN_TRADES

    bar_pct = min(100, int((current_bankroll - settings.bankroll_usdc) / settings.bankroll_usdc * 100))
    wr_pct  = min(100, int(wr / GO_LIVE_WIN_RATE * 100))
    tr_pct  = min(100, int(settled / GO_LIVE_MIN_TRADES * 100))

    gl1, gl2, gl3 = st.columns(3)

    with gl1:
        icon = "✅" if check1 else "⏳"
        st.markdown(f"**{icon} Bankroll 2x** — ${current_bankroll:,.0f} / ${GO_LIVE_TARGET:,.0f}")
        st.progress(bar_pct, text=f"{bar_pct}% of way to 2x")

    with gl2:
        icon = "✅" if check2 else "⏳"
        st.markdown(f"**{icon} Win Rate ≥ 65%** — {wr:.1%}")
        st.progress(wr_pct, text=f"{wr:.1%} / 65% target")

    with gl3:
        icon = "✅" if check3 else "⏳"
        st.markdown(f"**{icon} 30+ Settled Trades** — {settled}")
        st.progress(tr_pct, text=f"{settled} / {GO_LIVE_MIN_TRADES} trades")

    if check1 and check2 and check3:
        st.success("✅ All conditions met — you're ready to go live! Set `DRY_RUN=false` in your .env file.")
    else:
        remaining = []
        if not check1: remaining.append(f"${GO_LIVE_TARGET - current_bankroll:,.0f} more PnL to 2x")
        if not check2: remaining.append(f"win rate needs {GO_LIVE_WIN_RATE - wr:+.1%}")
        if not check3: remaining.append(f"{GO_LIVE_MIN_TRADES - settled} more settled trades")
        st.info("Still paper trading. Remaining: " + " · ".join(remaining))
else:
    st.success("🟢 **LIVE MODE** — Trading with real USDC on Polygon")

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────

col_pnl, col_wr, col_dist = st.columns([3, 2, 2])

with col_pnl:
    st.markdown("#### Bankroll Curve")
    series = bankroll_series(initial=settings.bankroll_usdc)
    if series:
        dts = [dt for dt, _ in series]
        vals = [v for _, v in series]
        last = vals[-1]
        color = "#2ecc71" if last >= settings.bankroll_usdc else "#e74c3c"
        fill_color = "rgba(46,204,113,0.12)" if last >= settings.bankroll_usdc else "rgba(231,76,60,0.12)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dts, y=vals,
            mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=fill_color,
            name="Bankroll",
            hovertemplate="$%{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(y=settings.bankroll_usdc, line_dash="dash",
                      line_color="rgba(255,255,255,0.3)",
                      annotation_text=f"Start ${settings.bankroll_usdc:,.0f}",
                      annotation_font_color="gray")
        fig.add_hline(y=GO_LIVE_TARGET, line_dash="dot",
                      line_color="#f1c40f",
                      annotation_text=f"🚀 Go-live at ${GO_LIVE_TARGET:,.0f}",
                      annotation_font_color="#f1c40f")
        fig.update_layout(
            height=240, margin=dict(l=0, r=0, t=4, b=0),
            yaxis_title="USDC", xaxis_title=None,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No settled trades yet.")

with col_wr:
    st.markdown("#### Win Rate by Side")
    wr_by_side = win_rate_by_side()
    if any(wr_by_side.values()):
        sides = list(wr_by_side.keys())
        rates = [v * 100 for v in wr_by_side.values()]
        colors = ["#2ecc71" if r >= 68.4 else "#e67e22" for r in rates]
        fig2 = go.Figure(go.Bar(
            x=sides, y=rates,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in rates],
            textposition="outside",
        ))
        fig2.add_hline(y=68.4, line_dash="dot", line_color="#f1c40f",
                       annotation_text="68.4% target", annotation_font_color="#f1c40f")
        fig2.update_layout(
            height=240, margin=dict(l=0, r=0, t=4, b=0),
            yaxis=dict(range=[0, 105], title="Win %", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No settled trades yet.")

with col_dist:
    st.markdown("#### Trade Outcomes")
    try:
        with SessionLocal() as s:
            all_trades = s.query(TradeRow).all()
        wins = sum(1 for t in all_trades if t.outcome == "WIN")
        losses = sum(1 for t in all_trades if t.outcome == "LOSS")
        pending = sum(1 for t in all_trades if t.outcome == "PENDING")
        if wins + losses + pending > 0:
            fig3 = go.Figure(go.Pie(
                labels=["Wins", "Losses", "Pending"],
                values=[wins, losses, pending],
                marker_colors=["#2ecc71", "#e74c3c", "#f39c12"],
                hole=0.55,
                textinfo="label+percent",
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig3.update_layout(
                height=240, margin=dict(l=0, r=0, t=4, b=0),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(text=f"{wins+losses+pending}", font_size=22, showarrow=False)],
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No trades yet.")
    except Exception as e:
        st.error(str(e))

# ── Trade table ───────────────────────────────────────────────────────────────

st.divider()
st.markdown("#### Recent Trades")

trades = recent_trades(25)
if trades:
    df = pd.DataFrame(trades)

    def fmt_pnl(x):
        if x > 0:   return f"+${x:.2f}"
        if x < 0:   return f"-${abs(x):.2f}"
        return "—"

    def fmt_outcome(val):
        icons = {"WIN": "✅ WIN", "LOSS": "❌ LOSS", "PENDING": "⏳ Pending"}
        return icons.get(val, val)

    df["pnl"]     = df["pnl"].map(fmt_pnl)
    df["outcome"] = df["outcome"].map(fmt_outcome)
    df["entry"]   = df["entry"].map(lambda x: f"{x:.3f}")
    df["bet"]     = df["bet"].map(lambda x: f"${x:.2f}")
    df.columns    = ["ID", "Market", "Side", "Entry", "Bet", "Outcome", "PnL", "Placed"]

    st.dataframe(df, use_container_width=True, height=380, hide_index=True)
else:
    st.info("No trades yet — run `python main.py --run-once` to start.")

# ── Postmortems + Lessons ─────────────────────────────────────────────────────

st.divider()
col_pm, col_lessons = st.columns([3, 2])

with col_pm:
    st.markdown("#### Postmortem Findings")
    try:
        with SessionLocal() as s:
            rows = (
                s.query(PostmortemRow)
                .order_by(PostmortemRow.created_at.desc())
                .limit(15)
                .all()
            )
        if rows:
            sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            for r in rows:
                icon = sev_icon.get(r.severity, "⚪")
                label = f"{icon} Trade #{r.trade_id} · {r.agent_name}"
                with st.expander(label, expanded=(r.severity in ("critical", "high"))):
                    st.markdown(f"**Finding:** {r.finding}")
                    if r.root_cause:
                        st.markdown(f"**Root cause:** {r.root_cause}")
                    if r.recommendation:
                        st.markdown(f"**Fix:** {r.recommendation}")
                    st.caption(f"Severity: {r.severity}")
        else:
            st.info("No postmortems yet — they run automatically after each loss.")
    except Exception as e:
        st.error(str(e))

with col_lessons:
    st.markdown("#### System Lessons Learned")
    try:
        with SessionLocal() as s:
            lessons = (
                s.query(LessonRow)
                .filter(LessonRow.active == True)
                .order_by(LessonRow.created_at.desc())
                .limit(20)
                .all()
            )
        if lessons:
            for l in lessons:
                st.markdown(f"- **[{l.category}]** {l.lesson}")
        else:
            st.info("Lessons build up as postmortems analyze losses.")
    except Exception as e:
        st.error(str(e))

# ── Recent bot activity ───────────────────────────────────────────────────────

st.divider()
st.markdown("#### Last 5 Trades Placed")
try:
    with SessionLocal() as s:
        recent = (
            s.query(TradeRow)
            .order_by(TradeRow.placed_at.desc())
            .limit(5)
            .all()
        )
    if recent:
        scan_data = [{
            "Market": (r.question or "")[:70],
            "Side": r.side,
            "Entry": f"{r.entry_price:.3f}",
            "Bet": f"${r.bet_usdc:.2f}",
            "Status": r.outcome,
            "TX": (r.tx_hash[:20] + "…") if r.tx_hash and len(r.tx_hash) > 20 else (r.tx_hash or "—"),
        } for r in recent]
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True, hide_index=True)
    else:
        st.info("No trades placed yet.")
except Exception as e:
    st.error(str(e))

# ── Auto-refresh ──────────────────────────────────────────────────────────────

time.sleep(REFRESH_INTERVAL)
st.rerun()
