"""
Diagnostic: dump the real Polymarket US market structure so we can map
slugs correctly. Public read — no order is placed, no money moves.

Run:  python inspect_pmus.py
Then paste the output back.
"""
from polymarket_us import PolymarketUS

print("Connecting to Polymarket US...")
client = PolymarketUS()

result = client.markets.list()
print("RESULT TYPE:", type(result).__name__)

# Find the iterable of markets inside the response
items = result
for attr in ("data", "items", "markets", "results"):
    if hasattr(result, attr):
        items = getattr(result, attr)
        print(f"(markets live under .{attr})")
        break

try:
    items = list(items)
except Exception:
    items = [items]

print(f"TOTAL MARKETS: {len(items)}\n")

for i, m in enumerate(items[:5]):
    print(f"========== MARKET {i} ==========")
    print("type:", type(m).__name__)
    # Dump fields whichever way the object is shaped
    d = None
    if isinstance(m, dict):
        d = m
    elif hasattr(m, "model_dump"):
        d = m.model_dump()
    elif hasattr(m, "__dict__"):
        d = m.__dict__
    if d:
        for k, v in d.items():
            print(f"  {k}: {repr(v)[:140]}")
    else:
        print("  (raw):", repr(m)[:300])
    print()

client.close()
print("Done — copy everything above and paste it back.")
