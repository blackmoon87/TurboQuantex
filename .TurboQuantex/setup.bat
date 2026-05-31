@echo off
echo ====================================================
echo             TURBOQUANTEX ENGINE SETUP
echo ====================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 goto python_ok

echo [!] Python was not found on your system.
echo.
echo Please install Python 3.8+ using one of the following methods:
echo 1. Windows Package Manager:
echo    winget install Python.Python.3.11
echo 2. Download manually from:
echo    https://www.python.org/downloads/
echo.
echo [!] IMPORTANT: Make sure to check the box "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:python_ok
echo [+] Python installation detected.
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% equ 0 goto python_version_ok

echo [-] Error: Python version 3.8 or higher is required.
python --version
pause
exit /b 1

:python_version_ok
:: Create Virtual Environment
if exist "%~dp0venv" goto venv_exists
echo [*] Creating virtual environment (venv)...
python -m venv "%~dp0venv"
goto venv_done

:venv_exists
echo [+] Virtual environment (venv) already exists.

:venv_done
:: Activate Virtual Environment & Install Dependencies
echo [*] Activating virtual environment and upgrading pip...
call "%~dp0venv\Scripts\activate.bat"
python -m pip install --upgrade pip

if not exist "%~dp0requirements.txt" goto manual_install
echo [*] Installing dependencies from requirements.txt...
pip install -r "%~dp0requirements.txt"
goto setup_done

:manual_install
echo [!] Warning: requirements.txt not found. Installing core packages manually...
pip install onnxruntime tokenizers flask numpy

:setup_done
echo.
echo ====================================================
echo             SETUP COMPLETED SUCCESSFULLY!
echo ====================================================
echo.
echo To start the background embedding daemon:
echo    python .TurboQuantex\app.py
echo.
echo To run vector codebase searches:
echo    python .TurboQuantex\tq.py search --query "search term"
echo.
pause
