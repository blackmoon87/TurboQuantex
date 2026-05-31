@echo off
setlocal enabledelayedexpansion
echo ====================================================
echo             TURBOQUANTEX ENGINE SETUP
echo ====================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python was not found on your system.
    echo.
    echo Please install Python 3.8+ using one of the following methods:
    echo 1. Windows Package Manager (Run in cmd):
    echo    winget install Python.Python.3.11
    echo 2. Download manually from:
    echo    https://www.python.org/downloads/
    echo.
    echo [!] IMPORTANT: Make sure to check the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [+] Python installation detected.
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [-] Error: Python version 3.8 or higher is required.
    python --version
    pause
    exit /b 1
)

:: Create Virtual Environment
if not exist venv (
    echo [*] Creating virtual environment (venv)...
    python -m venv venv
) else (
    echo [+] Virtual environment (venv) already exists.
)

:: Activate Virtual Environment & Install Dependencies
echo [*] Activating virtual environment and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

if exist requirements.txt (
    echo [*] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [!] Warning: requirements.txt not found. Installing core packages manually...
    pip install sentence-transformers flask numpy torch
)

echo.
echo ====================================================
echo             SETUP COMPLETED SUCCESSFULLY!
echo ====================================================
echo.
echo To start the background embedding daemon:
echo    python app.py
echo.
echo To run vector codebase searches:
echo    python turbo_code.py search --query "search term"
echo.
pause
