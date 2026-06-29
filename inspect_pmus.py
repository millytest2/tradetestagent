"""
Diagnostic: find which query returns OPEN Polymarket US markets.
Read-only — no order placed, no money moves.

Run:  python inspect_pmus.py
Then paste the whole output back.
"""
import json
import httpx

BASE = "https://gateway.polymarket.us/v1/markets"

def show(label, params):
    print(f"\n===== {label}  params={params} =====")
    try:
        r = httpx.get(BASE, params=params, timeout=20)
        print("HTTP", r.status_code)
        data = r.json()
        markets = data.get("markets", data if isinstance(data, list) else [])
        print("returned:", len(markets), "markets")
        open_ones = [m for m in markets
                     if isinstance(m, dict) and not m.get("closed") and not m.get("archived")]
        print("OPEN (not closed/archived):", len(open_ones))
        for m in open_ones[:5]:
            print(f"   slug={m.get('slug')!r:50} closed={m.get('closed')} "
                  f"active={m.get('active')} prices={m.get('outcomePrices')} "
                  f"cat={m.get('category')}")
        # if none open, show what the first few look like
        if not open_ones and markets:
            print("   (no open ones — sample of what came back:)")
            for m in markets[:3]:
                if isinstance(m, dict):
                    print(f"   slug={m.get('slug')!r} closed={m.get('closed')} active={m.get('active')}")
    except Exception as e:
        print("ERROR:", e)

show("no filter", {})
show("active only", {"active": "true"})
show("active + closed=false", {"active": "true", "closed": "false"})
show("closed=false only", {"closed": "false"})
show("limit 500", {"limit": 500})
show("active + limit 500", {"active": "true", "limit": 500})

print("\nDone — copy everything above and paste it back.")
