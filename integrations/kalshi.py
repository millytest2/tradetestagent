"""
Kalshi integration — US-legal CFTC-regulated prediction market exchange.

Kalshi is legal for US residents and has a full REST API.
Sign up at kalshi.com → Account → API Access to get credentials.

Authentication: API key ID + RSA private key (generated in Kalshi dashboard).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings
from core.models import Market, MarketSide, Trade, TradeStatus

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _build_headers(method: str, path: str) -> dict:
    """Generate Kalshi HMAC-signed request headers."""
    timestamp = str(int(time.time() * 1000))
    msg = timestamp + method.upper() + path
    try:
        sig = hmac.new(
            settings.kalshi_api_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).digest()
        signature = base64.b64encode(sig).decode()
    except Exception:
        signature = ""

    return {
        "KALSHI-ACCESS-KEY": settings.kalshi_api_key,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def _get(path: str, params: dict = None) -> dict:
    headers = _build_headers("GET", path)
    url = KALSHI_BASE + path
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=headers, params=params or {})
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, payload: dict) -> dict:
    headers = _build_headers("POST", path)
    url = KALSHI_BASE + path
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Account ───────────────────────────────────────────────────────────────────

async def get_balance() -> float:
    """Return account balance in USD."""
    try:
        data = await _get("/portfolio/balance")
        # Kalshi returns balance in cents
        return float(data.get("balance", 0)) / 100.0
    except Exception as e:
        logger.warning("Kalshi balance fetch failed: %s", e)
        return settings.bankroll_usdc


# ── Markets ───────────────────────────────────────────────────────────────────

def _parse_kalshi_market(raw: dict) -> Optional[Market]:
    """Parse a Kalshi market response into our Market model."""
    try:
        ticker = raw.get("ticker", "")
        yes_ask = float(raw.get("yes_ask", 50)) / 100.0   # Kalshi uses cents (0-99)
        yes_bid = float(raw.get("yes_bid", 50)) / 100.0
        no_ask = float(raw.get("no_ask", 50)) / 100.0
        no_bid = float(raw.get("no_bid", 50)) / 100.0

        yes_price = (yes_ask + yes_bid) / 2.0
        no_price = (no_ask + no_bid) / 2.0
        spread = yes_ask - yes_bid

        close_time = raw.get("close_time", "")
        days_left = 999
        if close_time:
            try:
                dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                days_left = (dt - datetime.now(timezone.utc)).total_seconds() / 86400
            except Exception:
                pass

        volume = float(raw.get("volume", 0) or 0)
        liquidity = float(raw.get("open_interest", 0) or 0) * yes_price

        # Estimate 24h price change from last_price
        last_price = float(raw.get("last_price", yes_price * 100) or yes_price * 100) / 100.0
        price_change_24h = abs(yes_price - last_price)

        return Market(
            condition_id=ticker,   # use ticker as condition_id
            question=raw.get("title", raw.get("subtitle", ticker)),
            description=raw.get("rules_primary", ""),
            end_date_iso=close_time,
            liquidity_usdc=max(liquidity, volume * 0.1),
            volume_24h_usdc=volume * yes_price,
            yes_price=max(0.01, min(0.99, yes_price)),
            no_price=max(0.01, min(0.99, no_price)),
            spread=spread,
            price_change_24h=price_change_24h,
            time_to_resolution_days=max(0, days_left),
            tags=[raw.get("category", "")],
            yes_token_id=ticker + "-YES",
            no_token_id=ticker + "-NO",
        )
    except Exception as e:
        logger.debug("Failed to parse Kalshi market: %s", e)
        return None


async def get_active_markets(limit: int = 200) -> list[Market]:
    """Fetch active Kalshi markets, return as our Market model list."""
    try:
        data = await _get("/markets", params={
            "status": "open",
            "limit": min(limit, 200),
        })
        raw_markets = data.get("markets", [])
        markets = []
        for raw in raw_markets:
            m = _parse_kalshi_market(raw)
            if m and m.condition_id:
                markets.append(m)
        logger.info("Fetched %d active markets from Kalshi", len(markets))
        return markets
    except Exception as e:
        logger.error("Kalshi market fetch failed: %s", e)
        return []


async def get_order_book(ticker: str) -> Optional[dict]:
    """Fetch Kalshi order book for whale detection."""
    try:
        data = await _get(f"/markets/{ticker}/orderbook")
        return data
    except Exception as e:
        logger.debug("Kalshi order book failed for %s: %s", ticker, e)
        return None


async def get_whale_signal(market: Market, threshold_usdc: float = 2000.0) -> float:
    """
    Detect whale activity in Kalshi order book.
    Returns +1.0 (heavy YES buying) to -1.0 (heavy NO buying).
    """
    book = await get_order_book(market.condition_id)
    if not book:
        return 0.0
    try:
        def _whale_usdc(orders: list, price_multiplier: float = 1.0) -> float:
            total = 0.0
            for o in orders:
                price = float(o.get("price", 0)) / 100.0
                size = float(o.get("quantity", 0))
                notional = price * size * price_multiplier
                if notional >= threshold_usdc:
                    total += notional
            return total

        yes_bids = _whale_usdc(book.get("yes", []))
        no_bids = _whale_usdc(book.get("no", []))
        total = yes_bids + no_bids

        if total < threshold_usdc:
            return 0.0

        imbalance = (yes_bids - no_bids) / total
        logger.info("Kalshi whale: %.2f (yes=$%.0f no=$%.0f) for %s",
                    imbalance, yes_bids, no_bids, market.question[:50])
        return float(max(-1.0, min(1.0, imbalance)))
    except Exception as e:
        logger.debug("Kalshi whale signal failed: %s", e)
        return 0.0


# ── Trade execution ───────────────────────────────────────────────────────────

async def place_trade(
    condition_id: str,
    side: MarketSide,
    bet_usdc: float,
    price: float,
    dry_run: bool = True,
) -> Trade:
    """Place a trade on Kalshi."""
    import uuid as _uuid

    # Kalshi: YES side = buy YES contracts, NO side = buy NO contracts
    kalshi_side = "yes" if side == MarketSide.YES else "no"

    # Contracts = dollar amount / price (Kalshi pays $1 per contract on win)
    contracts = int(bet_usdc / max(price, 0.01))
    # Kalshi price is in cents (1-99)
    price_cents = max(1, min(99, int(price * 100)))

    if dry_run:
        logger.info(
            "[DRY RUN] Kalshi: %s %s | %d contracts @ %d¢ ($%.2f)",
            kalshi_side.upper(), condition_id, contracts, price_cents, bet_usdc,
        )
        return Trade(
            market_id=condition_id,
            question=condition_id,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=float(contracts),
            status=TradeStatus.PLACED,
            tx_hash=f"kalshi_dry_{_uuid.uuid4().hex[:12]}",
        )

    # Live order
    try:
        order_id = str(_uuid.uuid4())
        payload = {
            "action": "buy",
            "client_order_id": order_id,
            "count": contracts,
            "side": kalshi_side,
            "ticker": condition_id,
            "type": "limit",
            "yes_price": price_cents if kalshi_side == "yes" else (100 - price_cents),
        }
        resp = await _post("/portfolio/orders", payload)

        order_resp_id = resp.get("order", {}).get("order_id", order_id)
        logger.info(
            "Kalshi order placed: %s %s %d contracts @ %d¢ — id=%s",
            kalshi_side.upper(), condition_id, contracts, price_cents, order_resp_id,
        )

        return Trade(
            market_id=condition_id,
            question=condition_id,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=float(contracts),
            status=TradeStatus.PLACED,
            tx_hash=order_resp_id,
        )

    except Exception as e:
        logger.error("Kalshi trade failed: %s", e)
        raise


async def check_settlement(market_ticker: str) -> Optional[dict]:
    """Check if a Kalshi market has resolved."""
    try:
        data = await _get(f"/markets/{market_ticker}")
        market = data.get("market", {})
        status = market.get("status", "")
        result = market.get("result", "")
        if status in ("finalized", "settled"):
            return {
                "resolved": True,
                "result": result,   # "yes" or "no"
                "market": market,
            }
        return {"resolved": False}
    except Exception as e:
        logger.debug("Kalshi settlement check failed: %s", e)
        return None
