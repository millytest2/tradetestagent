# Stock Trading Bot — Architecture Blueprint
# Use this as the starting prompt in a new Claude Code session

---

## What We're Building

A stock trading bot using the same 5-step pipeline as the prediction market bot,
adapted for US equities. Legal, regulated, and works with US brokerages.

Same core logic:
  1. Scan Agent       — filter stocks by momentum, volume, anomalies
  2. Research Agents  — Reddit (r/stocks, r/wallstreetbets), RSS news, Google Trends
  3. Prediction Agent — XGBoost + Claude Sonnet probability calibration
  4. Risk Agent       — Kelly criterion sizing, place order via brokerage API
  5. Postmortem       — analyze every losing trade, update lessons DB

---

## Key Differences from Prediction Market Bot

| Component         | Prediction Market        | Stocks                          |
|-------------------|--------------------------|---------------------------------|
| Market data       | Polymarket Gamma API     | yfinance / Alpaca / Polygon.io  |
| Execution         | Polymarket CLOB API      | Alpaca Paper/Live API           |
| Instrument        | Binary YES/NO            | Long/Short equity               |
| Settlement        | Market resolves at 0/1   | Stop-loss or take-profit target |
| Edge              | prob vs market price     | predicted move vs entry price   |
| Bankroll currency | USDC on Polygon          | USD in brokerage account        |
| Regulation        | Polymarket (blocked US)  | SEC/FINRA regulated — US legal  |

---

## Brokerage: Alpaca (recommended)

- Free paper trading + live trading API
- No commission on stocks
- Python SDK: `pip install alpaca-trade-api`
- Sign up: alpaca.markets
- Paper trading key + live key (same code, different env var)
- Supports fractional shares (can bet $5 on Apple)

```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # paper
# ALPACA_BASE_URL=https://api.alpaca.markets       # live
```

---

## Market Data

Use `yfinance` (free, no key needed) for:
- Current price, volume, 52-week high/low
- RSI, MACD, Bollinger Bands (via `ta` library)
- Earnings dates, float, short interest

Use Alpaca data API for real-time quotes during market hours.

```python
import yfinance as yf
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="30d")
info = ticker.info
```

---

## Scan Agent — Stock Universe

Instead of 300 Polymarket markets, scan:
- S&P 500 components (500 stocks)
- Filter by: volume > 1M/day, market cap > $1B, not near earnings
- Flag: unusual volume spike (>2x avg), large price move (>3% in a day),
         RSI oversold (<30) or overbought (>70), earnings catalyst upcoming

```python
SCAN_FILTERS = {
    "min_volume": 1_000_000,
    "min_market_cap": 1_000_000_000,
    "max_days_to_earnings": 3,   # avoid earnings uncertainty
    "min_price": 5.0,            # avoid penny stocks
}
```

---

## Prediction Features (XGBoost)

Replace prediction market features with stock features:

```python
FEATURE_COLS = [
    # Technical
    "rsi_14",              # RSI (0-100)
    "macd_signal",         # MACD signal line
    "bb_position",         # where price is in Bollinger Band (0-1)
    "volume_ratio",        # today volume / 20-day avg volume
    "price_change_1d",     # 1-day return
    "price_change_5d",     # 5-day return
    "price_change_20d",    # 20-day return
    # Sentiment
    "compound_sentiment",  # VADER on news/Reddit
    "post_count",          # number of relevant posts
    "trend_score",         # Google Trends score
    # Fundamentals
    "short_interest_ratio",# days to cover short
    "distance_from_52w_high",  # % below 52-week high
]
```

---

## Prediction Agent — Claude Prompt

Change the question from "will YES win?" to:

```
"Will [TICKER] be higher or lower 5 trading days from now?"

Given:
- Current price: $X
- RSI: Y (overbought/oversold signal)
- Recent news sentiment: Z
- Unusual volume: Nx average
- Short interest: X days to cover

Estimate P(price higher in 5 days): 0.0 to 1.0
```

Trade if P(up) > 0.65 → buy
Trade if P(up) < 0.35 → short (or skip if no shorting)

---

## Risk Agent — Kelly for Stocks

Same Kelly formula, but:
- "Odds" = expected return if correct (e.g., 5% move = 1.05x)
- "Win prob" = calibrated P(up) from prediction agent
- Add stop-loss: exit at -3% to cap downside
- Add take-profit: exit at +5% to lock gains
- Max position = 10% of portfolio per stock
- Max 5 open positions at once (correlation risk)

```python
# Stop-loss order placed immediately after entry
alpaca.submit_order(
    symbol="AAPL",
    qty=shares,
    side="buy",
    type="market",
    order_class="bracket",
    stop_loss={"stop_price": entry * 0.97},    # -3%
    take_profit={"limit_price": entry * 1.05},  # +5%
)
```

---

## Settlement

Stocks don't "resolve" like prediction markets.
Instead, check open positions daily:
- If stop-loss hit → LOSS recorded
- If take-profit hit → WIN recorded
- If 5 trading days elapsed → close at market, record outcome

Run `settler.py` daily after market close (4:30 PM ET).

---

## Research Agents

Same as prediction market bot, just different subreddits:
- Reddit: r/stocks, r/investing, r/wallstreetbets, r/SecurityAnalysis
- RSS: Yahoo Finance, Reuters Business, Bloomberg, Seeking Alpha
- Google Trends: company name + "stock" keyword
- Twitter/X: $TICKER cashtag search (needs Twitter API)

---

## Schedule

```
Market hours: 9:30 AM - 4:00 PM ET (weekdays only)
Scan: 9:45 AM ET daily (after open volatility settles)
Research + Predict + Trade: 10:00 AM ET
Settler: 4:30 PM ET daily
Retrain XGBoost: Weekly on Sunday
```

Windows Task Scheduler tasks:
- TradeBot_Scan: daily 9:45 AM ET
- TradeBot_Settler: daily 4:30 PM ET
- TradeBot_Retrain: weekly Sunday 8 AM ET

---

## Paper Trading First

Alpaca has free paper trading — identical API to live, fake money.
Run paper for 30-60 trading days (6-12 weeks) before going live.
Target win rate: >55% (stocks are harder than prediction markets).

---

## Files to Reuse Unchanged

- core/database.py       — same SQLite schema works perfectly
- core/models.py         — minor field renames only
- core/analytics.py      — all metrics work the same
- ml/calibrator.py       — XGBoost training logic unchanged
- agents/postmortem_agent.py — works identically
- dashboard.py           — works identically
- integrations/reddit.py — works identically
- integrations/rss_feed.py — works identically
- integrations/trends.py — works identically

## Files to Rewrite

- integrations/polymarket.py → integrations/alpaca.py + integrations/market_data.py
- agents/scan_agent.py       → adapt for S&P 500 universe
- agents/prediction_agent.py → adapt features + prompt for stocks
- agents/risk_agent.py       → add bracket orders, stop-loss/take-profit
- main.py                    → market hours check, schedule changes

---

## Startup Prompt for New Claude Code Session

Paste this into a new session:

"Build a stock trading bot using this exact architecture from a prediction market bot
we already built. Reuse as much code as possible.

Five-step pipeline:
1. Scan S&P 500 for unusual volume/momentum/RSI signals
2. Research via Reddit (r/stocks, r/wallstreetbets), RSS news, Google Trends
3. XGBoost + Claude Sonnet predict P(stock higher in 5 days)
4. Kelly criterion sizing, place bracket orders via Alpaca API
5. Postmortem on every losing trade, update lessons DB

Tech stack:
- Alpaca API for execution (paper first, then live)
- yfinance for market data
- Same SQLite database, XGBoost, Streamlit dashboard
- Same Kelly criterion, dynamic Kelly, circuit breaker logic

Reference architecture: [paste this file contents]

Start by creating the project structure and the Alpaca integration."
