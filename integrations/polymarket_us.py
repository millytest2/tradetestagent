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


def _first_num(*vals, default: float) -> float:
    """First value that is not None, coerced to float; else `default`.
    A real 0 is kept (not treated as missing)."""
    for v in vals:
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return float(default)


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
            # Default ONLY when the field is truly absent (None) — a real 0 must
            # stay 0 so the liquidity/volume quality filters can reject it.
            liquidity_usdc=(float(raw["liquidity"]) if raw.get("liquidity") is not None else 5000.0),
            volume_24h_usdc=_first_num(raw.get("volume24hr"), raw.get("volume"), default=1000.0),
            yes_price=max(0.01, min(0.99, yes_price)),
            no_price=max(0.01, min(0.99, no_price)),
            spread=abs(no_price - (1.0 - yes_price)),
            price_change_24h=0.0,
            # Keep negative days for already-expired markets so the scan's
            # "never trade expired" guard (time_to_resolution_days < 0) can fire.
            time_to_resolution_days=days_left,
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
    slug = getattr(market, "slug", "") or getattr(market, "condition_id", "")
    if not slug:
        return 0.0

    md = await _fetch_book(slug)
    if md is None:
        return 0.0

    try:
        bids = md.get("bids") or []
        asks = md.get("offers") or md.get("asks") or []

        def _whale_notional(orders) -> float:
            total = 0.0
            for o in orders:
                if not isinstance(o, dict):
                    continue
                price = _px(o.get("px") if o.get("px") is not None else o.get("price")) or 0.0
                size = _num(o.get("qty") if o.get("qty") is not None
                            else o.get("size", o.get("quantity"))) or 0.0
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


def _px(v) -> float | None:
    """Coerce a gateway price — either a bare number or {'value': '0.78',
    'currency': 'USD'} — to a float. None if absent/unparseable."""
    if isinstance(v, dict):
        v = v.get("value")
    return _num(v)


async def _fetch_book(slug: str) -> dict | None:
    """Fetch and unwrap the order book for a market. The gateway nests
    everything under 'marketData': {marketSlug, bids, offers, state, stats}."""
    import httpx
    if not slug:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"https://gateway.polymarket.us/v1/markets/{slug}/book")
            if r.status_code != 200:
                return None
            book = r.json()
    except Exception as e:
        logger.debug("PM-US book fetch failed for %s: %s", slug, e)
        return None
    if not isinstance(book, dict):
        return None
    md = book.get("marketData")
    return md if isinstance(md, dict) else book


def _book_best(orders, highest: bool) -> float | None:
    ps = []
    for o in orders or []:
        if isinstance(o, dict):
            p = _px(o.get("px") if o.get("px") is not None else o.get("price"))
            if p is not None:
                ps.append(p)
    if not ps:
        return None
    return max(ps) if highest else min(ps)


async def get_current_price(slug: str, side: str):
    """Current market price (0..1) for our side of a market, or None.

    Reads the gateway book: marketData.bids/offers (prices are {'value': str}
    objects), falling back to stats.lastTradePx/closePx, and — for an EXPIRED
    market — the settlementPx (1.0 or 0.0), so resolved positions mark to their
    true final value.
    """
    md = await _fetch_book(slug)
    if md is None:
        return None

    stats = md.get("stats") if isinstance(md.get("stats"), dict) else {}
    state = str(md.get("state") or "")

    yes_p = None
    if "EXPIRED" in state:
        yes_p = _px(stats.get("settlementPx"))   # authoritative final value
    if yes_p is None:
        best_bid = _book_best(md.get("bids"), True)
        best_ask = _book_best(md.get("offers") or md.get("asks"), False)
        if best_bid is not None and best_ask is not None:
            yes_p = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            yes_p = best_bid
        elif best_ask is not None:
            yes_p = best_ask
    if yes_p is None:
        yes_p = _px(stats.get("lastTradePx"))
    if yes_p is None:
        yes_p = _px(stats.get("closePx"))
    if yes_p is None:
        return None
    yes_p = max(0.0, min(1.0, yes_p))
    return yes_p if side == "YES" else (1.0 - yes_p)


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
            # Fallback: a MARKETABLE sell (priced a few cents through the quote so
            # it fills immediately) — a resting limit at mid can sit unfilled while
            # the caller wrongly books the position as closed.
            sell_price = max(0.01, min(0.99, price - 0.03))
            client.orders.create({
                "marketSlug": slug,
                "intent": intent,
                "type": "ORDER_TYPE_LIMIT",
                "price": {"value": f"{sell_price:.2f}", "currency": "USD"},
                "quantity": int(shares),
                "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
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
    Returns 'yes'/'no' for which OUTCOME won (market-data path), 'won'/'lost'
    for OUR position's result (authenticated positions path), or None if still
    open / unknown.
    """
    import httpx, json as _json
    if not slug:
        return None

    # 0) BOOK (authoritative): an EXPIRED market carries its settlementPx —
    #    1.0 means outcome[0] (YES side) won, 0.0 means it lost.
    try:
        md = await _fetch_book(slug)
        if md is not None and "EXPIRED" in str(md.get("state") or ""):
            stats = md.get("stats") if isinstance(md.get("stats"), dict) else {}
            sp = _px(stats.get("settlementPx"))
            if sp is not None:
                logger.info("Settlement via book for %s: settlementPx=%.4f", slug, sp)
                if sp >= 0.99:
                    return "yes"
                if sp <= 0.01:
                    return "no"
    except Exception as e:
        logger.debug("PM-US settlement book check failed for %s: %s", slug, e)

    # 1) Market lookup via the LIST endpoint (the single-market GET
    #    /v1/markets/<slug> returns 404 on this gateway, so query instead).
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for params in ({"slug": slug}, {"slugs": slug}, {"search": slug}):
                try:
                    r = await client.get(PMUS_GATEWAY, params=params)
                    if r.status_code != 200:
                        continue
                    body = r.json()
                    mkts = body.get("markets", []) if isinstance(body, dict) else []
                    m = next((x for x in mkts if isinstance(x, dict)
                              and x.get("slug") == slug), None)
                    if m is None:
                        continue
                    if not (m.get("closed") or m.get("archived")):
                        return None   # found and still open → not resolved
                    prices = m.get("outcomePrices")
                    if isinstance(prices, str):
                        prices = _json.loads(prices)
                    if prices and len(prices) >= 2:
                        if float(prices[0]) >= 0.99:
                            return "yes"
                        if float(prices[1]) >= 0.99:
                            return "no"
                    return None
                except Exception:
                    continue
    except Exception as e:
        logger.debug("PM-US settlement market lookup failed for %s: %s", slug, e)

    # 2) Authenticated positions payload: an 'expired' position has resolved.
    #    'realized' > 0 means the payout was credited → our side WON.
    try:
        raw = await _rest_get_raw((
            "/v1/portfolio/positions", "/v1/account/positions", "/v1/positions",
        ))
        container = raw.get("positions") if isinstance(raw, dict) else None
        entry = container.get(slug) if isinstance(container, dict) else None
        if isinstance(entry, dict) and entry.get("expired"):
            logger.info("Settlement via positions payload for %s: %s",
                        slug, str(entry).replace("\n", " ")[:300])
            realized = None
            rz = entry.get("realized")
            if isinstance(rz, dict):
                realized = _num(rz.get("value"))
            elif rz is not None:
                realized = _num(rz)
            if realized is not None:
                return "won" if realized > 0 else "lost"
    except Exception as e:
        logger.debug("PM-US settlement positions check failed for %s: %s", slug, e)
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
    payload. Handles the Polymarket US shape {'positions': {<slug>: {...}}} —
    a DICT keyed by slug — as well as a plain list of position objects. Only
    counts entries with a non-zero net position."""
    out: set[str] = set()
    if raw is None:
        return out

    # Unwrap the container.
    container = raw
    if isinstance(raw, dict):
        for k in ("positions", "data", "result", "items", "holdings"):
            if k in raw and isinstance(raw[k], (list, dict)):
                container = raw[k]
                break

    # Normalise to a list of (key_slug, entry) pairs.
    pairs = []
    if isinstance(container, dict):
        # Keyed by slug (the KEY is the market slug).
        pairs = [(str(k), v) for k, v in container.items()]
    elif isinstance(container, (list, tuple)):
        pairs = [(None, v) for v in container]
    else:
        return out

    for key_slug, entry in pairs:
        if isinstance(entry, dict):
            d = {str(k).lower(): v for k, v in entry.items()}
        elif key_slug is not None:
            # value isn't a dict but the key is a slug — count it
            out.add(key_slug)
            continue
        else:
            continue

        # Non-zero net position required.
        size = None
        for sk in ("netposition", "net_position", "size", "quantity", "shares",
                   "netquantity", "net_quantity", "position", "qtybought", "amount"):
            if sk in d and _num(d[sk]) is not None:
                size = _num(d[sk])
                break
        if size is not None and abs(size) < 1e-9:
            continue

        # Slug: the dict key, else nested marketMetadata.slug, else common keys.
        slug = key_slug
        if not slug:
            meta = d.get("marketmetadata") or {}
            if isinstance(meta, dict):
                slug = meta.get("slug") or meta.get("Slug")
        if not slug:
            for k in ("marketslug", "market_slug", "slug", "market", "market_id",
                      "marketid", "conditionid", "condition_id", "ticker", "symbol"):
                if d.get(k):
                    slug = d[k]
                    break
        if slug:
            out.add(str(slug))
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

    # 2) REST fallback. /v1/portfolio/positions is the one that responds 200 on
    #    this gateway, so try it first.
    if raw is None:
        try:
            raw = await _rest_get_raw((
                "/v1/portfolio/positions", "/v1/account/positions",
                "/v1/positions", "/v1/account/portfolio",
            ))
        except Exception as e:
            logger.debug("REST positions probe failed: %s", e)

    # Surface the real shape so parsing can be verified from the logs. Strip
    # embedded newlines (the exchange's 'outcome' field contains one, which was
    # truncating the logged line) and show more of the payload.
    if raw is not None:
        try:
            flat = str(raw).replace("\n", " ").replace("\r", " ")
            logger.info("Raw positions payload: %s", flat[:1500])
        except Exception:
            pass

    slugs = _extract_position_slugs(raw)
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
    exec_price = max(0.01, min(0.99, price))
    price_str = f"{exec_price:.2f}"
    # contracts: each ~$1 max payout; quantity = dollars / price
    quantity = max(1, int(bet_usdc / max(exec_price, 0.01)))
    # ACTUAL dollars committed = contracts × price (differs from the intended
    # bet_usdc because of the integer-contract rounding). Record the real cost so
    # committed-capital accounting and the wallet guard stay accurate.
    actual_cost = round(quantity * exec_price, 2)
    client_oid = str(uuid.uuid4())

    if dry_run:
        logger.info(
            "[DRY RUN] Polymarket US: %s %s | %d @ $%s (~$%.2f)",
            intent, slug, quantity, price_str, actual_cost,
        )
        return Trade(
            market_id=condition_id,
            question=slug,
            side=side,
            entry_price=exec_price,
            bet_usdc=actual_cost,
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
            entry_price=exec_price,
            bet_usdc=actual_cost,
            shares=float(quantity),
            status=TradeStatus.PLACED,
            tx_hash=str(order_id),
        )
    except Exception as e:
        logger.error("Polymarket US order failed: %s", e)
        raise
