"""
Polymarket Key Setup
════════════════════
Derives and saves CLOB API credentials from your private key.
Run this ONCE before first live trading:

  python setup_keys.py

It will:
  1. Derive your wallet address from the private key
  2. Create (or re-derive) Polymarket CLOB API credentials
  3. Write them into .env
  4. Check your USDC balance on Polygon
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

from eth_account import Account
from rich.console import Console
from rich.panel import Panel

console = Console()


def derive_address(private_key: str) -> str:
    acct = Account.from_key(private_key)
    return acct.address


def setup_clob_credentials(private_key: str) -> dict:
    """
    Derive Polymarket CLOB API key/secret/passphrase from private key.
    These are deterministically derived — the same key always gives the same creds.
    """
    try:
        from py_clob_client.client import ClobClient

        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=private_key,
            signature_type=2,
        )
        creds = client.create_or_derive_api_creds()
        return {
            "api_key":        creds.api_key,
            "api_secret":     creds.api_secret,
            "api_passphrase": creds.api_passphrase,
        }
    except Exception as e:
        console.print(f"[yellow]Could not derive CLOB creds (network needed): {e}[/yellow]")
        return {}


def update_env(updates: dict, env_path: str = ".env") -> None:
    """Update specific keys in the .env file without touching others."""
    with open(env_path, "r") as f:
        content = f.read()

    for key, value in updates.items():
        pattern = rf"^{key}=.*$"
        replacement = f"{key}={value}"
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content += f"\n{key}={value}"

    with open(env_path, "w") as f:
        f.write(content)


async def check_polygon_balance(address: str) -> float:
    """Check USDC balance on Polygon mainnet."""
    try:
        import httpx
        # Polygon USDC contract: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
        USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        # Use public RPC
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": USDC_CONTRACT,
                "data": f"0x70a08231000000000000000000000000{address[2:].lower().zfill(64)}"
            }, "latest"],
            "id": 1,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post("https://polygon-rpc.com", json=payload)
            result = r.json().get("result", "0x0")
            raw = int(result, 16)
            return raw / 1_000_000   # USDC has 6 decimals
    except Exception as e:
        console.print(f"[dim]Balance check failed (network): {e}[/dim]")
        return -1.0


def main() -> None:
    pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")

    # Try reading from .env if not in environment
    if not pk:
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("POLYMARKET_PRIVATE_KEY="):
                        pk = line.split("=", 1)[1].strip()
                        break
        except FileNotFoundError:
            pass

    if not pk:
        console.print("[red]ERROR: POLYMARKET_PRIVATE_KEY not found in .env[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]Polymarket Key Setup[/bold cyan]",
        border_style="cyan",
    ))

    # 1. Derive wallet address
    address = derive_address(pk)
    console.print(f"  Wallet address : [bold]{address}[/bold]")
    console.print(f"  Polygon scan   : https://polygonscan.com/address/{address}")

    # 2. Check balance
    balance = asyncio.run(check_polygon_balance(address))
    if balance >= 0:
        color = "green" if balance > 0 else "yellow"
        console.print(f"  USDC balance   : [{color}]${balance:,.2f}[/{color}]")
        if balance == 0:
            console.print(
                "  [yellow]⚠  Wallet has no USDC. "
                "Bridge USDC to Polygon before live trading.[/yellow]"
            )
    else:
        console.print("  USDC balance   : [dim]network unavailable[/dim]")

    # 3. Derive CLOB credentials
    console.print("\n  Deriving CLOB API credentials from private key...")
    creds = setup_clob_credentials(pk)

    if creds:
        update_env({
            "POLYMARKET_API_KEY":        creds["api_key"],
            "POLYMARKET_API_SECRET":     creds["api_secret"],
            "POLYMARKET_API_PASSPHRASE": creds["api_passphrase"],
        })
        console.print(
            f"  API key        : [green]{creds['api_key'][:12]}…[/green]  (written to .env)"
        )
        console.print("  [green]✓ CLOB credentials saved.[/green]")
    else:
        console.print(
            "  [yellow]CLOB creds need live network — run again with Polymarket reachable.[/yellow]"
        )

    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Bridge USDC to Polygon if balance is 0\n"
        "  2. Add your ANTHROPIC_API_KEY to .env\n"
        "  3. Run: [cyan]python main.py --run-once[/cyan]   (dry-run, no real money)\n"
        "  4. Run: [cyan]python main.py --live[/cyan]       (LIVE — real USDC)\n"
        "  5. Run: [cyan]streamlit run dashboard.py[/cyan]  (live dashboard)"
    )
    console.print(
        "\n[red bold]⚠  Security reminder:[/red bold] "
        "You shared this private key in chat — rotate it to a fresh wallet "
        "before depositing significant funds."
    )


if __name__ == "__main__":
    main()
