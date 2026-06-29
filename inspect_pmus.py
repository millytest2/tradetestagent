"""
Diagnostic: dump the real Polymarket US market structure so we can map
slugs correctly. Public read — no order is placed, no money moves.

Run:  python inspect_pmus.py
Then paste the output back.
"""
import json
from polymarket_us import PolymarketUS

print("Connecting to Polymarket US...")
client = PolymarketUS()

result = client.markets.list()
print("RESULT TYPE:", type(result).__name__)

# The result is a dict — show its top-level keys
if isinstance(result, dict):
    print("TOP-LEVEL KEYS:", list(result.keys()))
    # Find the list of markets inside
    markets = None
    for k, v in result.items():
        if isinstance(v, list):
            print(f"  -> key '{k}' holds a list of {len(v)} items")
            if markets is None:
                markets = v
                markets_key = k
    if markets is None:
        print("No list found. Full dict (truncated):")
        print(json.dumps(result, default=str)[:2000])
else:
    markets = list(result)

if markets:
    print(f"\nUsing {len(markets)} markets. First 5:\n")
    for i, m in enumerate(markets[:5]):
        print(f"========== MARKET {i} ==========")
        if isinstance(m, dict):
            for k, v in m.items():
                print(f"  {k}: {repr(v)[:140]}")
        elif hasattr(m, "model_dump"):
            for k, v in m.model_dump().items():
                print(f"  {k}: {repr(v)[:140]}")
        else:
            print("  (raw):", repr(m)[:400])
        print()

client.close()
print("Done — copy everything above and paste it back.")
