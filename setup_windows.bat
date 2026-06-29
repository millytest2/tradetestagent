@echo off
title Prediction Market Bot Setup
echo ================================================
echo  Prediction Market Bot - Windows Setup
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo.
    echo  1. Go to https://python.org/downloads
    echo  2. Download Python 3.11 or newer
    echo  3. Run the installer - CHECK the box that says "Add Python to PATH"
    echo  4. Come back and double-click this file again
    echo.
    pause
    exit /b 1
)
echo [OK] Python found

:: Check .env file
if not exist ".env" (
    echo.
    echo ERROR: .env file not found in this folder.
    echo.
    echo Creating it now with your credentials...
    (
        echo POLYMARKET_PRIVATE_KEY=0x7ca04a28f733a1ba12264b31c8d1662cc232b796295defd218f8f77a36a72f95
        echo POLYMARKET_API_KEY=3178e964-f72a-4ff5-bc78-1dc68aa3b45a
        echo POLYMARKET_API_SECRET=GMlK4WCVziT7bWwlP2YqFOtdJGQrZp1qJZ2jXEM/LEOFy+BKLno9TjDG/CmmRKJcNvWuMISXHcTqgANYcYrnaw==
        echo LIVE_EXCHANGE=polymarket
        echo BANKROLL_USDC=100.0
        echo NOTIFY_EMAIL=milestipton10@gmail.com
        echo AB_TESTING_ENABLED=true
        echo NOTIFY_SMTP_HOST=smtp.gmail.com
        echo NOTIFY_SMTP_PORT=587
    ) > .env
    echo [OK] .env created with your credentials
)

:: Create virtual environment if needed
if not exist ".venv" (
    echo.
    echo Creating virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
)

:: Activate and install
echo.
echo Installing dependencies (takes 2-3 minutes first time)...
call .venv\Scripts\activate.bat
pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency install failed. Check your internet connection.
    pause
    exit /b 1
)
echo [OK] All dependencies installed

:: Init database
python -c "from core.database import init_db; init_db()" 2>nul
echo [OK] Database ready

echo.
echo ================================================
echo  Starting bot in LIVE trading mode...
echo  Your $100 on Polymarket is now active.
echo  Press Ctrl+C to stop.
echo ================================================
echo.
python main.py --live
pause
