"""Polymarket Gamma API client for market data and CLOB for trade execution."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from core.models import Market, MarketSide, Trade, TradeStatus

logger = logging.getLogger(__name__)

GAMMA_BASE = settings.gamma_api_url
CLOB_BASE = settings.clob_api_url


# ── Market Data ───────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def fetch_markets(limit: int = 300, offset: int = 0) -> list[dict]:
    """Fetch active markets from the Gamma API."""
    params = {
        "limit": limit,
        "offset": offset,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{GAMMA_BASE}/markets", params=params)
        resp.raise_for_status()
        return resp.json()


def _parse_market(raw: dict) -> Optional[Market]:
    """Parse a raw Gamma API market response into a Market model."""
    try:
        # Extract prices from tokens (YES/NO)
        tokens = raw.get("tokens", [])
        yes_price = 0.5
        no_price = 0.5

        for token in tokens:
            outcome = token.get("outcome", "").upper()
            price = float(token.get("price", 0.5))
            if outcome == "YES":
                yes_price = price
            elif outcome == "NO":
                no_price = price

        # Resolve end date
        end_date = raw.get("endDate") or raw.get("endDateIso") or ""

        if end_date:
            try:
                dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days_left = (dt - now).total_seconds() / 86400
            except Exception:
                days_left = 999
        else:
            days_left = 999

        volume_24h = float(raw.get("volume24hr", 0) or 0)
        liquidity = float(raw.get("liquidity", 0) or raw.get("liquidityNum", 0) or 0)

        # Compute spread and 24h price change
        spread = abs(no_price - (1.0 - yes_price))
        last_price = float(raw.get("lastTradePrice", yes_price) or yes_price)
        price_change_24h = abs(yes_price - last_price)

        return Market(
            condition_id=raw.get("conditionId") or raw.get("id") or "",
            question=raw.get("question") or raw.get("title") or "Unknown",
            description=raw.get("description") or "",
            end_date_iso=end_date,
            liquidity_usdc=liquidity,
            volume_24h_usdc=volume_24h,
            yes_price=yes_price,
            no_price=no_price,
            spread=spread,
            price_change_24h=price_change_24h,
            time_to_resolution_days=max(0, days_left),
            tags=raw.get("tags") or [],
        )
    except Exception as e:
        logger.debug("Failed to parse market %s: %s", raw.get("id"), e)
        return None


async def get_active_markets(limit: int = 300) -> list[Market]:
    """Fetch and parse active prediction markets."""
    try:
        raw_markets = await fetch_markets(limit=limit)
        markets = []
        for raw in raw_markets:
            m = _parse_market(raw)
            if m and m.condition_id:
                markets.append(m)
        logger.info("Fetched %d active markets from Polymarket", len(markets))
        return markets
    except Exception as e:
        logger.error("Failed to fetch markets: %s", e)
        return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def get_market_by_id(condition_id: str) -> Optional[Market]:
    """Fetch a single market by condition ID."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{GAMMA_BASE}/markets",
                params={"conditionId": condition_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                raw = data[0] if isinstance(data, list) else data
                return _parse_market(raw)
    except Exception as e:
        logger.error("Failed to fetch market %s: %s", condition_id, e)
    return None


# ── Order Book ────────────────────────────────────────────────────────────────

async def get_order_book(token_id: str) -> Optional[dict]:
    """Fetch the CLOB order book for a token."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{CLOB_BASE}/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.debug("Order book fetch failed for %s: %s", token_id, e)
        return None


# ── Trade Execution (stub + real path) ───────────────────────────────────────

async def place_trade(
    condition_id: str,
    side: MarketSide,
    bet_usdc: float,
    price: float,
    dry_run: bool = True,
) -> Trade:
    """
    Place a trade on Polymarket.

    In dry_run mode (default) the trade is simulated — no real money is spent.
    Set dry_run=False and configure POLYMARKET_PRIVATE_KEY for live trading.
    """
    import uuid

    if dry_run:
        logger.info(
            "[DRY RUN] Would place %s on %s for $%.2f at %.3f",
            side.value, condition_id, bet_usdc, price,
        )
        shares = bet_usdc / max(price, 1e-6)
        return Trade(
            market_id=condition_id,
            question=condition_id,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=shares,
            status=TradeStatus.PLACED,
            tx_hash=f"dry_run_{uuid.uuid4().hex[:12]}",
        )

    # ── Live execution via py-clob-client ─────────────────────────────────────
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType

        from py_clob_client.clob_types import ApiCreds
        client = ClobClient(
            host=CLOB_BASE,
            chain_id=137,  # Polygon mainnet
            key=settings.polymarket_private_key,
            signature_type=2,  # EIP-712
        )
        creds = client.create_or_derive_api_creds()
        client = ClobClient(
            host=CLOB_BASE,
            chain_id=137,
            key=settings.polymarket_private_key,
            creds=creds,
            signature_type=2,
        )

        # Determine token_id for the outcome
        market = await get_market_by_id(condition_id)
        if not market:
            raise ValueError(f"Market {condition_id} not found")

        # Calculate shares from bet amount
        shares = bet_usdc / max(price, 1e-6)

        order_args = OrderArgs(
            token_id=condition_id,
            price=price,
            size=shares,
            side=side.value,
        )
        signed_order = client.create_and_sign_order(order_args)
        resp = client.post_order(signed_order, OrderType.GTC)

        tx_hash = resp.get("orderID", "") or resp.get("transactionHash", "")
        logger.info(
            "Order placed: %s %s on %s — tx %s",
            side.value, f"${bet_usdc:.2f}", condition_id, tx_hash,
        )

        return Trade(
            market_id=condition_id,
            question=market.question,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=shares,
            status=TradeStatus.PLACED,
            tx_hash=tx_hash,
        )

    except ImportError:
        logger.error("py-clob-client not installed — falling back to dry run")
        shares = bet_usdc / max(price, 1e-6)
        return Trade(
            market_id=condition_id,
            question=condition_id,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=shares,
            status=TradeStatus.PLACED,
            tx_hash=f"fallback_{uuid.uuid4().hex[:12]}",
        )
    except Exception as e:
        logger.error("Trade execution failed: %s", e)
        raise


async def check_settlement(trade: Trade) -> Optional[dict]:
    """Check if a placed trade has been settled."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            market = await get_market_by_id(trade.market_id)
            if market and market.time_to_resolution_days <= 0:
                # Market has resolved — check final outcome
                # In real implementation, query on-chain settlement
                return {"resolved": True, "market": market}
    except Exception as e:
        logger.debug("Settlement check failed: %s", e)
    return None
