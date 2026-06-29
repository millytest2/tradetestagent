"""
Polymarket US integration — the CFTC-regulated, US-legal API.

This is the endpoint that WORKS for US users (api.polymarket.us), launched
Feb 2026. Unlike the international CLOB (clob.polymarket.com) which geoblocks
US orders with a 403, this one is built for US residents.

Setup (one time):
  1. Open the Polymarket app, finish KYC (you already did this to deposit)
  2. Go to polymarket.us/developer → generate API keys
  3. You get a KEY ID (uuid) and a SECRET KEY (base64 Ed25519 private key)
  4. Put them in .env:
       POLYMARKET_KEY_ID=...
       POLYMARKET_SECRET_KEY=...

Market data (scan/research) still comes from the Gamma API (read-only, never
geoblocked). Only ORDER EXECUTION routes through here.

Docs: https://docs.polymarket.us   SDK: pip install polymarket-us
"""

from __future__ import annotations

import logging
import uuid

from config import settings
from core.models import MarketSide, Trade, TradeStatus

logger = logging.getLogger(__name__)


def _client():
    """Build an authenticated Polymarket US SDK client."""
    from polymarket_us import PolymarketUS

    # Accept either the dedicated US field names OR the generic Polymarket
    # API creds (same UUID key + base64 secret work for both).
    key_id = settings.polymarket_key_id or settings.polymarket_api_key
    secret_key = settings.polymarket_secret_key or settings.polymarket_api_secret
    if not key_id or not secret_key:
        raise ValueError(
            "No Polymarket US credentials. Set POLYMARKET_KEY_ID + "
            "POLYMARKET_SECRET_KEY (or POLYMARKET_API_KEY + POLYMARKET_API_SECRET) "
            "in .env. Generate them at polymarket.us/developer."
        )
    return PolymarketUS(key_id=key_id, secret_key=secret_key)


async def get_balance() -> float:
    """Return USD balance on the Polymarket US account."""
    try:
        client = _client()
        # Balance lives under the account/portfolio resource in the SDK
        bal = client.account.balance()
        client.close()
        return float(getattr(bal, "available", None) or bal.get("available", 0))
    except Exception as e:
        logger.warning("Polymarket US balance fetch failed: %s", e)
        return settings.bankroll_usdc


async def place_trade(
    condition_id: str,
    side: MarketSide,
    bet_usdc: float,
    price: float,
    dry_run: bool = True,
    slug: str = "",
) -> Trade:
    """
    Place a limit order on Polymarket US via the official SDK.

    The US API is slug-based (not condition_id), so `slug` must be provided —
    the risk agent passes market.slug through.
    """
    if not slug:
        raise ValueError(
            f"Polymarket US needs a market slug (condition_id={condition_id}). "
            "Market.slug was empty — check the Gamma scan parsed it."
        )

    # Buy the side we believe in. The US API expresses direction via intent.
    intent = "ORDER_INTENT_BUY_LONG" if side == MarketSide.YES else "ORDER_INTENT_BUY_SHORT"
    # contracts: each ~$1 max payout; quantity = dollars / price
    quantity = max(1, int(bet_usdc / max(price, 0.01)))
    price_str = f"{max(0.01, min(0.99, price)):.2f}"
    client_oid = str(uuid.uuid4())

    if dry_run:
        logger.info(
            "[DRY RUN] Polymarket US: %s %s | %d @ $%s (~$%.2f)",
            intent, slug, quantity, price_str, bet_usdc,
        )
        return Trade(
            market_id=condition_id,
            question=slug,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=float(quantity),
            status=TradeStatus.PLACED,
            tx_hash=f"pmus_dry_{client_oid[:12]}",
        )

    try:
        client = _client()
        order = client.orders.create({
            "marketSlug": slug,
            "intent": intent,
            "type": "ORDER_TYPE_LIMIT",
            "price": {"value": price_str, "currency": "USD"},
            "quantity": quantity,
            "tif": "TIME_IN_FORCE_GOOD_TILL_CANCEL",
        })
        order_id = (
            getattr(order, "id", None)
            or (order.get("id") if isinstance(order, dict) else None)
            or client_oid
        )
        client.close()
        logger.info(
            "Polymarket US order placed: %s %s %d @ $%s — id=%s",
            intent, slug, quantity, price_str, order_id,
        )
        return Trade(
            market_id=condition_id,
            question=slug,
            side=side,
            entry_price=price,
            bet_usdc=bet_usdc,
            shares=float(quantity),
            status=TradeStatus.PLACED,
            tx_hash=str(order_id),
        )
    except Exception as e:
        logger.error("Polymarket US order failed: %s", e)
        raise
