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


def _public_client():
    """Read-only client (no auth needed for market data)."""
    from polymarket_us import PolymarketUS
    return PolymarketUS()


def _parse_us_market(raw: dict):
    """Parse a Polymarket US market dict into our Market model."""
    import json as _json
    from datetime import datetime, timezone
    from core.models import Market

    try:
        slug = raw.get("slug") or ""
        if not slug:
            return None

        outcomes = raw.get("outcomes")
        prices = raw.get("outcomePrices")
        if isinstance(outcomes, str):
            try: outcomes = _json.loads(outcomes)
            except Exception: outcomes = []
        if isinstance(prices, str):
            try: prices = _json.loads(prices)
            except Exception: prices = []
        outcomes = outcomes or []
        prices = prices or []

        # Two-outcome market → treat outcome[0] as "YES", outcome[1] as "NO"
        yes_price = float(prices[0]) if len(prices) >= 1 else 0.5
        no_price  = float(prices[1]) if len(prices) >= 2 else (1.0 - yes_price)

        end_date = raw.get("endDate") or ""
        days_left = 999.0
        if end_date:
            try:
                dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                days_left = (dt - datetime.now(timezone.utc)).total_seconds() / 86400
            except Exception:
                pass

        # Build a readable question that includes the outcome labels
        question = raw.get("question") or slug
        if len(outcomes) >= 2:
            question = f"{question} — {outcomes[0]} (YES) vs {outcomes[1]} (NO)"

        return Market(
            condition_id=slug,                       # slug doubles as id for US
            slug=slug,
            question=question,
            description=raw.get("description", "")[:500],
            end_date_iso=end_date,
            liquidity_usdc=float(raw.get("liquidity", 0) or 5000),  # US API may omit
            volume_24h_usdc=float(raw.get("volume24hr", raw.get("volume", 0)) or 1000),
            yes_price=max(0.01, min(0.99, yes_price)),
            no_price=max(0.01, min(0.99, no_price)),
            spread=abs(no_price - (1.0 - yes_price)),
            price_change_24h=0.0,
            time_to_resolution_days=max(0.0, days_left),
            tags=[raw.get("category", "polymarket_us")],
            yes_token_id="",
            no_token_id="",
        )
    except Exception as e:
        logger.debug("Failed to parse US market %s: %s", raw.get("slug"), e)
        return None


PMUS_GATEWAY = "https://gateway.polymarket.us/v1/markets"


async def get_active_markets(limit: int = 200):
    """
    Fetch OPEN Polymarket US markets via the public gateway.

    The `closed=false` filter is REQUIRED — without it the API returns
    resolved markets (closed does NOT default to false in practice).
    Paginates to gather a wide pool of open markets.
    """
    import httpx

    # Sports markets number in the thousands and would crowd out everything
    # else if we only paginated the default list. So we ALSO pull the
    # non-sports categories explicitly (crypto, politics, economics, weather)
    # and merge — guaranteeing econ/Bitcoin/politics/temperature markets are in
    # the pool, not buried past the fetch window.
    CATEGORIES = ["crypto", "politics", "economics", "weather", "pop-culture", "science"]
    raw_markets = []
    seen_slugs = set()

    async def _pull(client, params, cap):
        out, offset = [], 0
        while len(out) < cap:
            try:
                resp = await client.get(PMUS_GATEWAY, params={**params, "limit": 100, "offset": offset})
                if resp.status_code != 200:
                    break
                page = resp.json().get("markets", []) if isinstance(resp.json(), dict) else []
            except Exception:
                break
            if not page:
                break
            out.extend(page)
            if len(page) < 100:
                break
            offset += 100
        return out

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            # 1) explicit non-sports categories first (so they're guaranteed in)
            for cat in CATEGORIES:
                for m in await _pull(client, {"closed": "false", "category": cat}, 200):
                    sl = m.get("slug")
                    if sl and sl not in seen_slugs:
                        seen_slugs.add(sl); raw_markets.append(m)
            # 2) then the general pool to fill out the rest (incl. sports)
            for m in await _pull(client, {"closed": "false"}, 800):
                sl = m.get("slug")
                if sl and sl not in seen_slugs:
                    seen_slugs.add(sl); raw_markets.append(m)
    except Exception as e:
        logger.error("Polymarket US market list failed: %s", e)
        return []

    markets = []
    for raw in raw_markets:
        if not isinstance(raw, dict):
            continue
        if raw.get("closed") or raw.get("archived"):
            continue
        m = _parse_us_market(raw)
        if m:
            markets.append(m)

    logger.info("Fetched %d open Polymarket US markets", len(markets))
    return markets


async def get_whale_signal(market, threshold_usd: float = 1500.0) -> float:
    """
    Detect whale order-book pressure on a Polymarket US market.

    Returns +1.0 (big money stacked on the buy/YES side) to -1.0 (big money on
    the sell/NO side), 0.0 if no meaningful whale activity or the book can't be
    read. Best-effort: tries the gateway order-book endpoints and degrades
    gracefully (same 0.0 the bot used before) if the shape differs.
    """
    import httpx

    slug = getattr(market, "slug", "") or getattr(market, "condition_id", "")
    if not slug:
        return 0.0

    book = None
    for path in (f"/v1/markets/{slug}/book",
                 f"/v1/markets/{slug}/orderbook",
                 f"/v1/markets/{slug}/bbo"):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get("https://gateway.polymarket.us" + path)
                if r.status_code == 200:
                    book = r.json()
                    break
        except Exception:
            continue
    if not isinstance(book, dict):
        return 0.0

    try:
        # Order-book responses vary; accept the common key names
        bids = book.get("bids") or book.get("buys") or book.get("buy") or []
        asks = book.get("asks") or book.get("sells") or book.get("sell") or []
        # Some APIs nest under "orderbook"
        if not bids and not asks and isinstance(book.get("orderbook"), dict):
            ob = book["orderbook"]
            bids = ob.get("bids", []); asks = ob.get("asks", [])

        def _whale_notional(orders) -> float:
            total = 0.0
            for o in orders:
                if not isinstance(o, dict):
                    continue
                price = float(o.get("price", 0) or 0)
                size = float(o.get("size", o.get("quantity", 0)) or 0)
                notional = price * size
                if notional >= threshold_usd:
                    total += notional
            return total

        bid_usd = _whale_notional(bids)
        ask_usd = _whale_notional(asks)
        total = bid_usd + ask_usd
        if total < threshold_usd:
            return 0.0
        imbalance = (bid_usd - ask_usd) / total
        logger.info(
            "PM-US whale: %+.2f (bids=$%.0f asks=$%.0f) for %s",
            imbalance, bid_usd, ask_usd, slug,
        )
        return float(max(-1.0, min(1.0, imbalance)))
    except Exception as e:
        logger.debug("PM-US whale signal failed for %s: %s", slug, e)
        return 0.0


async def get_current_price(slug: str, side: str):
    """Current market price (0..1) for our side of a market, or None."""
    import httpx, json as _json
    if not slug:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"https://gateway.polymarket.us/v1/markets/{slug}")
            if r.status_code != 200:
                return None
            data = r.json()
        m = data.get("market") if isinstance(data, dict) and "market" in data else data
        prices = m.get("outcomePrices") if isinstance(m, dict) else None
        if isinstance(prices, str):
            prices = _json.loads(prices)
        if not prices or len(prices) < 2:
            return None
        yes_p = float(prices[0])
        return yes_p if side == "YES" else (1.0 - yes_p)
    except Exception as e:
        logger.debug("PM-US price fetch failed for %s: %s", slug, e)
        return None


async def close_position(slug: str, side: MarketSide, shares: float, price: float,
                         dry_run: bool = True) -> bool:
    """
    SELL/exit an open position on Polymarket US. Returns True on success.
    Best-effort: uses the SDK close-position / sell-intent path.
    """
    if dry_run:
        logger.info("[DRY RUN] Close %s %s (%.0f shares @ %.2f)", side.value, slug, shares, price)
        return True
    try:
        client = _client()
        # Sell the side we hold: opposite intent of the buy
        intent = "ORDER_INTENT_SELL_LONG" if side == MarketSide.YES else "ORDER_INTENT_SELL_SHORT"
        try:
            client.orders.close_position({"marketSlug": slug})
        except Exception:
            # Fallback: explicit sell limit order
            client.orders.create({
                "marketSlug": slug,
                "intent": intent,
                "type": "ORDER_TYPE_LIMIT",
                "price": {"value": f"{max(0.01, min(0.99, price)):.2f}", "currency": "USD"},
                "quantity": int(shares),
                "tif": "TIME_IN_FORCE_GOOD_TILL_CANCEL",
            })
        client.close()
        logger.info("Closed position %s %s (%.0f shares)", side.value, slug, shares)
        return True
    except Exception as e:
        logger.error("PM-US close_position failed for %s: %s", slug, e)
        return False


async def check_settlement(slug: str):
    """
    Check if a Polymarket US market has resolved.
    Returns 'yes' if outcome[0] (our YES) won, 'no' if outcome[1] (NO) won,
    or None if still open / unknown. Resolved markets report outcomePrices
    like ["1","0"] (YES won) or ["0","1"] (NO won).
    """
    import httpx, json as _json
    if not slug:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"https://gateway.polymarket.us/v1/markets/{slug}")
            if r.status_code != 200:
                return None
            data = r.json()
        m = data.get("market") if isinstance(data, dict) and "market" in data else data
        if not isinstance(m, dict):
            return None
        if not (m.get("closed") or m.get("archived")):
            return None
        prices = m.get("outcomePrices")
        if isinstance(prices, str):
            prices = _json.loads(prices)
        if not prices or len(prices) < 2:
            return None
        if float(prices[0]) >= 0.99:
            return "yes"
        if float(prices[1]) >= 0.99:
            return "no"
        return None
    except Exception as e:
        logger.debug("PM-US settlement check failed for %s: %s", slug, e)
        return None


# Keys that mean "spendable cash" vs "total account equity (incl. open
# positions)". The exchange's /v1/account/balances payload can carry either or
# both; we separate them so sizing uses genuinely spendable cash, not equity
# that's tied up in unsettled positions.
_CASH_KEYS = ("available", "available_balance", "availablebalance", "cash",
              "buying_power", "buyingpower", "withdrawable", "free", "usdc", "usd")
_TOTAL_KEYS = ("total", "equity", "portfolio_value", "portfoliovalue",
               "net_liquidation", "networth", "total_balance", "value", "balance")


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_balance_fields(bal) -> tuple[float | None, float | None]:
    """Return (total_equity, cash_available) from any SDK/REST balances shape.

    Either element may be None if the payload doesn't distinguish them.
    """
    if bal is None:
        return None, None

    # A list/tuple of per-asset balances → find the first entry that yields
    # numbers (typically the USD/USDC line).
    if isinstance(bal, (list, tuple)):
        for entry in bal:
            t, c = _extract_balance_fields(entry)
            if t is not None or c is not None:
                return t, c
        return None, None

    # Normalise to a lowercase-keyed dict.
    if isinstance(bal, dict):
        d = {str(k).lower(): v for k, v in bal.items()}
    else:
        d = {}
        for k in _CASH_KEYS + _TOTAL_KEYS + ("balances", "data", "result"):
            v = getattr(bal, k, None)
            if v is not None:
                d[k] = v

    if not d:
        n = _num(bal)                # bare number → treat as cash
        return (None, n)

    # Unwrap a nested container (e.g. {"data": [...]}, {"balances": {...}}).
    for nest in ("balances", "data", "result"):
        if nest in d and isinstance(d[nest], (list, dict)):
            t, c = _extract_balance_fields(d[nest])
            if t is not None or c is not None:
                return t, c

    total = next((_num(d[k]) for k in _TOTAL_KEYS if k in d and _num(d[k]) is not None), None)
    cash = next((_num(d[k]) for k in _CASH_KEYS if k in d and _num(d[k]) is not None), None)
    return total, cash


async def get_account_balances() -> dict:
    """Fetch the account balances and split into total equity vs spendable cash.

    Returns {'total': float|None, 'cash': float|None}. The raw payload is logged
    once per call so the exact exchange shape is visible in the run logs.
    """
    raw = None
    # 1) SDK probe.
    try:
        client = _client()
        for holder_name in ("account", "portfolio", "wallet"):
            holder = getattr(client, holder_name, None)
            if holder is None:
                continue
            for getter in ("balances", "balance", "get_balances", "get_balance", "get"):
                fn = getattr(holder, getter, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        continue
            if raw is not None:
                break
        try:
            client.close()
        except Exception:
            pass
    except Exception as e:
        logger.debug("SDK balances probe failed: %s", e)

    # 2) Raw authenticated REST fallback.
    if raw is None:
        try:
            raw = await _rest_balances_raw()
        except Exception as e:
            logger.debug("REST balances probe failed: %s", e)

    if raw is None:
        return {"total": None, "cash": None}

    # Surface the real shape so cash-vs-total can be verified from logs.
    try:
        logger.info("Raw balances payload: %s", str(raw)[:400])
    except Exception:
        pass

    total, cash = _extract_balance_fields(raw)
    logger.info("Balances parsed — total=%s cash=%s",
                f"${total:.2f}" if total is not None else "n/a",
                f"${cash:.2f}" if cash is not None else "n/a")
    return {"total": total, "cash": cash}


async def get_balance() -> float:
    """Spendable USDC on the Polymarket US account.

    Prefers exchange-reported cash, falls back to total equity, then to the
    configured bankroll. Callers that need the total/cash split should use
    get_account_balances() directly.
    """
    b = await get_account_balances()
    for v in (b.get("cash"), b.get("total")):
        if v is not None and v > 0:
            return float(v)
    logger.debug("Live balance unavailable — using configured bankroll $%.2f",
                 settings.bankroll_usdc)
    return settings.bankroll_usdc


async def _rest_get_raw(endpoints: tuple[str, ...]):
    """Best-effort authenticated GET returning the first non-null JSON payload
    from the given endpoints, using the SDK's own signed HTTP session."""
    client = _client()
    session = (getattr(client, "session", None)
               or getattr(client, "_session", None)
               or getattr(client, "http", None))
    try:
        for ep in endpoints:
            for caller in ("get", "request"):
                fn = getattr(session, caller, None) if session else None
                if not callable(fn):
                    continue
                try:
                    resp = fn(ep) if caller == "get" else fn("GET", ep)
                    data = resp.json() if hasattr(resp, "json") else resp
                    if data is not None:
                        return data
                except Exception:
                    continue
    finally:
        try:
            client.close()
        except Exception:
            pass
    return None


async def _rest_balances_raw():
    """Best-effort authenticated GET returning the raw balances payload."""
    return await _rest_get_raw((
        "/v1/account/balances", "/v1/account/balance", "/v1/balances",
        "/v1/balance", "/v1/portfolio", "/v1/account",
    ))


def _extract_position_slugs(raw) -> set[str]:
    """Pull the set of market slugs/ids we currently hold from a positions
    payload of any shape. Only counts entries with a non-zero size."""
    out: set[str] = set()
    if raw is None:
        return out

    items = raw
    if isinstance(raw, dict):
        for k in ("positions", "data", "result", "items", "holdings"):
            if isinstance(raw.get(k), list):
                items = raw[k]
                break
        else:
            items = [raw]
    if not isinstance(items, (list, tuple)):
        return out

    for it in items:
        if isinstance(it, dict):
            d = {str(k).lower(): v for k, v in it.items()}
        else:
            d = {str(k).lower(): getattr(it, k)
                 for k in dir(it) if not k.startswith("_")}
        # Require a non-zero size/quantity to count as an open holding.
        size = None
        for sk in ("size", "quantity", "shares", "netquantity", "net_quantity",
                   "position", "amount", "balance"):
            if sk in d and _num(d[sk]) is not None:
                size = _num(d[sk])
                break
        if size is not None and abs(size) < 1e-9:
            continue
        for k in ("marketslug", "market_slug", "slug", "market", "market_id",
                  "marketid", "conditionid", "condition_id", "ticker", "symbol"):
            if d.get(k):
                out.add(str(d[k]))
                break
    return out


async def get_open_positions() -> set[str]:
    """Return the set of market slugs/ids we currently hold on Polymarket US.

    This is the SOURCE OF TRUTH for dedup — it reflects the real account, so it
    prevents re-buying a market we already hold even if the local trade DB (the
    GitHub cache) was lost or a prior cycle didn't persist.
    """
    raw = None
    # 1) SDK probe.
    try:
        client = _client()
        for holder_name in ("positions", "account", "portfolio"):
            holder = getattr(client, holder_name, None)
            if holder is None:
                continue
            for getter in ("positions", "list", "get_positions", "open_positions", "get"):
                fn = getattr(holder, getter, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        continue
            if raw is not None:
                break
        try:
            client.close()
        except Exception:
            pass
    except Exception as e:
        logger.debug("SDK positions probe failed: %s", e)

    # 2) REST fallback.
    if raw is None:
        try:
            raw = await _rest_get_raw((
                "/v1/account/positions", "/v1/positions",
                "/v1/account/portfolio", "/v1/portfolio/positions",
            ))
        except Exception as e:
            logger.debug("REST positions probe failed: %s", e)

    slugs = _extract_position_slugs(raw)
    if slugs:
        logger.info("Exchange reports %d open position(s): %s",
                    len(slugs), sorted(slugs)[:10])
    return slugs


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
