"""Alpaca Markets integration — stock data, scanning, and order execution."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from config import settings
from core.models import Stock, Trade, TradeOutcome, TradeStatus, StockSide

logger = logging.getLogger(__name__)

_BASE_DATA = "https://data.alpaca.markets/v2"
_BASE_TRADE = settings.alpaca_base_url + "/v2"
_HEADERS = {
    "APCA-API-KEY-ID": settings.alpaca_api_key,
    "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
    "Content-Type": "application/json",
}

# S&P 500 core universe — diversified across sectors
SP500_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC", "CRM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "V", "MA", "C",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "CMG", "DIS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "PXD", "DVN",
    # Industrial
    "CAT", "BA", "GE", "HON", "UPS", "RTX", "LMT", "MMM", "DE", "EMR",
    # Utilities / REIT
    "NEE", "DUK", "SO", "D", "AEP", "SPG", "PLD", "AMT", "CCI", "EQIX",
]


async def _get(url: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(headers=_HEADERS, timeout=20.0) as client:
        resp = await client.get(url, params=params or {})
        resp.raise_for_status()
        return resp.json()


async def _post(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(headers=_HEADERS, timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Account ───────────────────────────────────────────────────────────────────

async def get_account() -> dict:
    return await _get(f"{_BASE_TRADE}/account")


async def get_bankroll() -> float:
    data = await get_account()
    return float(data.get("portfolio_value", data.get("equity", 0.0)))


# ── Market data ───────────────────────────────────────────────────────────────

async def get_latest_bars(tickers: list[str]) -> dict[str, dict]:
    """Fetch latest OHLCV bar for a batch of tickers."""
    url = f"{_BASE_DATA}/stocks/bars/latest"
    symbols = ",".join(tickers)
    data = await _get(url, params={"symbols": symbols, "feed": "iex"})
    return data.get("bars", {})


async def get_bars_history(ticker: str, days: int = 60) -> list[dict]:
    """Fetch daily OHLCV bars going back `days` calendar days."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = f"{_BASE_DATA}/stocks/{ticker}/bars"
    data = await _get(url, params={
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "limit": days,
        "feed": "iex",
    })
    return data.get("bars", [])


async def get_snapshot(ticker: str) -> dict:
    """Get full snapshot (latest quote + trade + daily bar) for a ticker."""
    url = f"{_BASE_DATA}/stocks/{ticker}/snapshot"
    try:
        return await _get(url, params={"feed": "iex"})
    except Exception:
        return {}


# ── Technical indicators (computed from bar history) ──────────────────────────

def _compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def _compute_macd_histogram(closes: list[float]) -> float:
    """Returns MACD histogram (MACD line - signal line). Positive = bullish."""
    def ema(data: list[float], period: int) -> list[float]:
        k = 2 / (period + 1)
        result = [data[0]]
        for p in data[1:]:
            result.append(p * k + result[-1] * (1 - k))
        return result

    if len(closes) < 35:
        return 0.0
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [e12 - e26 for e12, e26 in zip(ema12[25:], ema26[25:])]
    if len(macd_line) < 9:
        return 0.0
    signal = ema(macd_line, 9)
    return macd_line[-1] - signal[-1]


def _compute_bb_position(closes: list[float], period: int = 20) -> float:
    """Bollinger Band position: 0 = at lower band, 1 = at upper band."""
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mean = sum(window) / period
    std = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
    if std == 0:
        return 0.5
    upper = mean + 2 * std
    lower = mean - 2 * std
    pos = (closes[-1] - lower) / (upper - lower)
    return max(0.0, min(1.0, pos))


async def build_stock(ticker: str) -> Optional[Stock]:
    """Construct a fully-featured Stock object for a given ticker."""
    try:
        bars = await get_bars_history(ticker, days=260)
        if not bars:
            return None

        closes = [b["c"] for b in bars]
        volumes = [b["v"] for b in bars]
        current_price = closes[-1]

        avg_vol_20d = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        today_vol = volumes[-1]
        volume_ratio = today_vol / avg_vol_20d if avg_vol_20d > 0 else 1.0

        pc_1d = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0.0
        pc_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0.0
        pc_20d = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0.0

        high_52w = max(closes[-252:]) if len(closes) >= 252 else max(closes)
        low_52w = min(closes[-252:]) if len(closes) >= 252 else min(closes)

        return Stock(
            ticker=ticker,
            current_price=current_price,
            volume_today=today_vol,
            avg_volume_20d=avg_vol_20d,
            volume_ratio=volume_ratio,
            rsi_14=_compute_rsi(closes),
            macd_signal=_compute_macd_histogram(closes),
            bb_position=_compute_bb_position(closes),
            price_change_1d=pc_1d,
            price_change_5d=pc_5d,
            price_change_20d=pc_20d,
            distance_from_52w_high=current_price / high_52w - 1,
            distance_from_52w_low=current_price / low_52w - 1,
        )
    except Exception as e:
        logger.debug("build_stock(%s) failed: %s", ticker, e)
        return None


# ── Open positions ────────────────────────────────────────────────────────────

async def get_open_positions_alpaca() -> list[dict]:
    """Return all open positions from the Alpaca account."""
    try:
        return await _get(f"{_BASE_TRADE}/positions")
    except Exception as e:
        logger.warning("get_open_positions failed: %s", e)
        return []


async def get_current_price(ticker: str) -> Optional[float]:
    """Fetch the latest trade price for a ticker."""
    try:
        url = f"{_BASE_DATA}/stocks/{ticker}/trades/latest"
        data = await _get(url, params={"feed": "iex"})
        return float(data["trade"]["p"])
    except Exception:
        # fallback to snapshot
        snap = await get_snapshot(ticker)
        try:
            return float(snap["latestTrade"]["p"])
        except Exception:
            return None


# ── Order execution ───────────────────────────────────────────────────────────

async def place_bracket_order(
    ticker: str,
    side: StockSide,
    shares: float,
    stop_loss_price: float,
    take_profit_price: float,
    dry_run: bool = True,
) -> dict:
    """
    Place a bracket order: market entry with stop-loss and take-profit legs.

    Set dry_run=True to simulate without spending real money.
    """
    if dry_run:
        logger.info(
            "[DRY RUN] Bracket order: %s %s x %.4f shares | SL=%.2f TP=%.2f",
            side.value, ticker, shares, stop_loss_price, take_profit_price,
        )
        return {
            "id": f"dryrun-{ticker}-{datetime.utcnow().strftime('%H%M%S')}",
            "status": "dry_run",
        }

    alpaca_side = "buy" if side == StockSide.LONG else "sell"
    stop_side = "sell" if side == StockSide.LONG else "buy"

    payload = {
        "symbol": ticker,
        "qty": str(round(shares, 4)),
        "side": alpaca_side,
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "stop_loss": {"stop_price": str(round(stop_loss_price, 2))},
        "take_profit": {"limit_price": str(round(take_profit_price, 2))},
    }

    url = f"{_BASE_TRADE}/orders"
    result = await _post(url, payload)
    logger.info("Order placed: %s", result.get("id"))
    return result


async def cancel_order(order_id: str) -> None:
    async with httpx.AsyncClient(headers=_HEADERS, timeout=10.0) as client:
        await client.delete(f"{_BASE_TRADE}/orders/{order_id}")


async def close_position(ticker: str) -> dict:
    """Market sell / buy-to-cover to close an open position."""
    async with httpx.AsyncClient(headers=_HEADERS, timeout=10.0) as client:
        resp = await client.delete(f"{_BASE_TRADE}/positions/{ticker}")
        resp.raise_for_status()
        return resp.json()


# ── Settlement check ──────────────────────────────────────────────────────────

async def check_trade_outcome(trade: Trade) -> tuple[TradeOutcome, float]:
    """
    Compare current price against stop-loss / take-profit to determine outcome.

    Returns (outcome, pnl_usd).
    """
    current = await get_current_price(trade.ticker)
    if current is None:
        return TradeOutcome.PENDING, 0.0

    if trade.side == StockSide.LONG:
        if current >= trade.take_profit_price:
            pnl = (trade.take_profit_price - trade.entry_price) * trade.shares
            return TradeOutcome.WIN, round(pnl, 2)
        if current <= trade.stop_loss_price:
            pnl = (trade.stop_loss_price - trade.entry_price) * trade.shares
            return TradeOutcome.LOSS, round(pnl, 2)
    else:  # SHORT
        if current <= trade.take_profit_price:
            pnl = (trade.entry_price - trade.take_profit_price) * trade.shares
            return TradeOutcome.WIN, round(pnl, 2)
        if current >= trade.stop_loss_price:
            pnl = (trade.entry_price - trade.stop_loss_price) * trade.shares
            return TradeOutcome.LOSS, round(pnl, 2)

    return TradeOutcome.PENDING, 0.0
