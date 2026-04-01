"""
Kalshi integration — US-legal CFTC-regulated prediction market exchange.

Fully legal for US residents. Sign up at kalshi.com.
Authentication uses RSA-PSS with a private key PEM file.

Setup:
  1. Go to kalshi.com → Account Settings → API
  2. Generate a key pair — save the private key PEM to a file (e.g. kalshi_private.pem)
  3. Copy the Key ID shown in the dashboard
  4. Set in .env:
       KALSHI_API_KEY=<key-id-uuid>
       KALSHI_PRIVATE_KEY_PATH=./kalshi_private.pem

API docs: https://docs.kalshi.com
"""

from __future__ import annotations

import base64
import datetime
import logging
import uuid
from pathlib import Path
from typing import Optional

import httpx

from config import settings
from core.models import Market, MarketSide, Trade, TradeStatus

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_DEMO = "https://demo-api.kalshi.co/trade-api/v2"


# ── RSA-PSS Authentication ────────────────────────────────────────────────────

def _load_private_key():
    """Load RSA private key from PEM file."""
    from cryptography.hazmat.primitives import serialization
    key_path = Path(getattr(settings, "kalshi_private_key_path", "./kalshi_private.pem"))
    if not key_path.exists():
        raise FileNotFoundError(
            f"Kalshi private key not found at {key_path}. "
            "Set KALSHI_PRIVATE_KEY_PATH in .env"
        )
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign_pss(private_key, text: str) -> str:
    """Sign text with RSA-PSS SHA-256."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    sig = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def _build_headers(method: str, path: str) -> dict:
    """Generate Kalshi RSA-PSS signed request headers."""
    ts = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))
    # Strip query string for signature
    path_no_qs = path.split("?")[0]
    msg = ts + method.upper() + path_no_qs

    try:
        private_key = _load_private_key()
        signature = _sign_pss(private_key, msg)
    except Exception as e:
        logger.warning("Kalshi auth signing failed: %s", e)
        signature = ""

    return {
        "KALSHI-ACCESS-KEY": settings.kalshi_api_key,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _base_url() -> str:
    return KALSHI_DEMO if getattr(settings, "kalshi_demo", False) else KALSHI_BASE


async def _get(path: str, params: dict = None) -> dict:
    headers = _build_headers("GET", path)
    url = _base_url() + path
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=headers, params=params or {})
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, payload: dict) -> dict:
    headers = _build_headers("POST", path)
    url = _base_url() + path
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


async def _delete(path: str) -> dict:
    headers = _build_headers("DELETE", path)
    url = _base_url() + path
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


# ── Account ───────────────────────────────────────────────────────────────────

async def get_balance() -> float:
    """Return account balance in USD."""
    try:
        data = await _get("/portfolio/balance")
        # balance field is in cents
        return float(data.get("balance", 0)) / 100.0
    except Exception as e:
        logger.warning("Kalshi balance fetch failed: %s", e)
        return settings.bankroll_usdc


# ── Market parsing ────────────────────────────────────────────────────────────

def _parse_kalshi_market(raw: dict) -> Optional[Market]:
    """Parse a Kalshi market dict into our Market model."""
    try:
        ticker = raw.get("ticker", "")

        # Prices come as dollar strings: "0.4800" or cent integers
        def _parse_price(field_dollars: str, field_cents: str, default: float = 0.5) -> float:
            if raw.get(field_dollars):
                return float(raw[field_dollars])
            if raw.get(field_cents):
                return float(raw[field_cents]) / 100.0
            return default

        yes_bid = _parse_price("yes_bid_dollars", "yes_bid", 0.48)
        yes_ask = _parse_price("yes_ask_dollars", "yes_ask", 0.52)
        no_bid  = _parse_price("no_bid_dollars",  "no_bid",  0.48)
        no_ask  = _parse_price("no_ask_dollars",  "no_ask",  0.52)

        yes_price = (yes_bid + yes_ask) / 2.0
        no_price  = (no_bid  + no_ask)  / 2.0
        spread    = yes_ask - yes_bid

        # Time to resolution
        close_time = raw.get("close_time") or raw.get("expiration_time", "")
        days_left = 999.0
        if close_time:
            try:
                dt = datetime.datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                days_left = (dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 86400
            except Exception:
                pass

        volume_24h = float(raw.get("volume_24h_fp", raw.get("volume_24h", 0)) or 0)
        open_interest = float(raw.get("open_interest_fp", raw.get("open_interest", 0)) or 0)
        liquidity = open_interest * yes_price

        last_price = _parse_price("last_price_dollars", "last_price", yes_price)
        price_change_24h = abs(yes_price - last_price)

        return Market(
            condition_id=ticker,
            question=raw.get("title") or raw.get("subtitle") or ticker,
            description=raw.get("rules_primary", ""),
            end_date_iso=close_time,
            liquidity_usdc=max(liquidity, volume_24h * 0.1),
            volume_24h_usdc=volume_24h,
            yes_price=max(0.01, min(0.99, yes_price)),
            no_price=max(0.01, min(0.99, no_price)),
            spread=spread,
            price_change_24h=price_change_24h,
            time_to_resolution_days=max(0.0, days_left),
            tags=[raw.get("category", "kalshi")],
            yes_token_id=ticker + "-YES",
            no_token_id=ticker + "-NO",
        )
    except Exception as e:
        logger.debug("Failed to parse Kalshi market %s: %s", raw.get("ticker"), e)
        return None


# ── Markets ───────────────────────────────────────────────────────────────────

async def get_active_markets(limit: int = 200) -> list[Market]:
    """Fetch open Kalshi markets as our Market model list."""
    markets: list[Market] = []
    cursor = None
    fetched = 0

    while fetched < limit:
        params = {"status": "open", "limit": min(200, limit - fetched)}
        if cursor:
            params["cursor"] = cursor

        try:
            data = await _get("/markets", params=params)
        except Exception as e:
            logger.error("Kalshi market fetch failed: %s", e)
            break

        raw_markets = data.get("markets", [])
        for raw in raw_markets:
            m = _parse_kalshi_market(raw)
            if m and m.condition_id:
                markets.append(m)

        fetched += len(raw_markets)
        cursor = data.get("cursor")
        if not cursor or not raw_markets:
            break

    logger.info("Fetched %d active markets from Kalshi", len(markets))
    return markets


# ── Order book / whale detection ──────────────────────────────────────────────

async def get_order_book(ticker: str) -> Optional[dict]:
    """Fetch Kalshi order book for a market."""
    try:
        return await _get(f"/markets/{ticker}/orderbook")
    except Exception as e:
        logger.debug("Kalshi order book failed for %s: %s", ticker, e)
        return None


async def get_whale_signal(market: Market, threshold_usdc: float = 2000.0) -> float:
    """
    Detect whale activity in the Kalshi order book.
    Returns +1.0 (heavy YES buying) to -1.0 (heavy NO buying).
    """
    book = await get_order_book(market.condition_id)
    if not book:
        return 0.0
    try:
        def _whale_usdc(orders: list) -> float:
            total = 0.0
            for o in orders:
                # Kalshi book: price in cents, quantity in contracts
                price = float(o.get("price", 0)) / 100.0
                qty   = float(o.get("quantity", 0))
                notional = price * qty
                if notional >= threshold_usdc:
                    total += notional
            return total

        yes_orders = book.get("orderbook", {}).get("yes", [])
        no_orders  = book.get("orderbook", {}).get("no", [])

        yes_usdc = _whale_usdc(yes_orders)
        no_usdc  = _whale_usdc(no_orders)
        total    = yes_usdc + no_usdc

        if total < threshold_usdc:
            return 0.0

        imbalance = (yes_usdc - no_usdc) / total
        logger.info(
            "Kalshi whale: %+.2f (yes=$%.0f no=$%.0f) for %s",
            imbalance, yes_usdc, no_usdc, market.question[:50],
        )
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
    """
    Place a limit order on Kalshi.

    Kalshi pricing: yes_price is integer cents (1-99).
    Contracts: each contract pays $1 on win. Cost = price_dollars * count.
    """
    kalshi_side   = "yes" if side == MarketSide.YES else "no"
    price_cents   = max(1, min(99, round(price * 100)))
    # Number of contracts we can buy with our bet
    contracts     = max(1, int(bet_usdc / max(price, 0.01)))
    client_oid    = str(uuid.uuid4())

    if dry_run:
        logger.info(
            "[DRY RUN] Kalshi: %s %s | %d contracts @ %d¢ (~$%.2f)",
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
            tx_hash=f"kalshi_dry_{client_oid[:12]}",
        )

    # Live order
    payload = {
        "ticker":           condition_id,
        "action":           "buy",
        "side":             kalshi_side,
        "type":             "limit",
        "count":            contracts,
        "yes_price":        price_cents if kalshi_side == "yes" else (100 - price_cents),
        "client_order_id":  client_oid,
        "time_in_force":    "fill_or_kill",
    }

    try:
        resp = await _post("/portfolio/orders", payload)
        order_id = resp.get("order", {}).get("order_id", client_oid)
        logger.info(
            "Kalshi order placed: %s %s %d contracts @ %d¢ — id=%s",
            kalshi_side.upper(), condition_id, contracts, price_cents, order_id,
        )
        return Trade(
            market_id=condition_id,
            question=condition_id,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=float(contracts),
            status=TradeStatus.PLACED,
            tx_hash=order_id,
        )
    except Exception as e:
        logger.error("Kalshi order failed: %s", e)
        raise


async def cancel_order(order_id: str) -> None:
    """Cancel an open Kalshi order."""
    try:
        await _delete(f"/portfolio/orders/{order_id}")
        logger.info("Kalshi order %s cancelled", order_id)
    except Exception as e:
        logger.warning("Kalshi cancel failed for %s: %s", order_id, e)


# ── Settlement check ──────────────────────────────────────────────────────────

async def check_settlement(ticker: str) -> Optional[dict]:
    """Check if a Kalshi market has resolved."""
    try:
        data = await _get(f"/markets/{ticker}")
        market = data.get("market", {})
        status = market.get("status", "")
        result = market.get("result", "")
        if status in ("finalized", "settled", "determined"):
            return {
                "resolved": True,
                "result": result,   # "yes" or "no"
                "market": market,
            }
        return {"resolved": False}
    except Exception as e:
        logger.debug("Kalshi settlement check failed for %s: %s", ticker, e)
        return None
