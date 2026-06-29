#!/bin/bash
# One-shot setup and launch for the Polymarket trading bot
# Run this once on your Mac: bash start.sh

set -e
cd "$(dirname "$0")"

echo "=== Prediction Market Bot Setup ==="

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "Python3 not found. Install it from https://python.org/downloads then re-run."
    exit 1
fi
echo "✓ Python $(python3 --version)"

# 2. Create virtualenv if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "Installing dependencies (this takes ~60s the first time)..."
pip install -q --upgrade pip
pip install -q \
    pydantic-settings \
    pydantic \
    sqlalchemy \
    httpx \
    tenacity \
    rich \
    vaderSentiment \
    py-clob-client \
    eth-account \
    anthropic \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    joblib \
    praw \
    pytrends \
    feedparser \
    cryptography

echo "✓ Dependencies installed"

# 4. Check .env
if [ ! -f ".env" ]; then
    echo ""
    echo "ERROR: .env file not found."
    echo "Create a .env file with your Polymarket private key:"
    echo ""
    echo "  POLYMARKET_PRIVATE_KEY=0x..."
    echo "  LIVE_EXCHANGE=polymarket"
    echo "  BANKROLL_USDC=100.0"
    echo "  NOTIFY_EMAIL=your@email.com"
    echo ""
    exit 1
fi
echo "✓ .env found"

# 5. Init database
python3 -c "from core.database import init_db; init_db(); print('✓ Database ready')"

# 6. Launch
echo ""
echo "=== Starting bot (live trading) ==="
echo "Press Ctrl+C to stop."
echo ""
python3 main.py --live
