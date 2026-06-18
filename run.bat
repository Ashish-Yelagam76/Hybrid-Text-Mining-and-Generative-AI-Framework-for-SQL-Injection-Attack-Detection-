@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist ".env" (
    echo.
    echo Copy .env.example to .env and add your GEMINI_API_KEY for AI analysis.
    copy .env.example .env >nul
)

echo.
echo Starting SQL Injection Detection app...
python main.py

pause
