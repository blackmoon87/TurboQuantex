#!/bin/bash

echo "===================================================="
echo "             TURBOQUANTEX ENGINE SETUP (Unix)"
echo "===================================================="
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 was not found on your system."
    echo
    echo "Please install Python 3.8+ using your package manager:"
    echo "  - macOS (using Homebrew):"
    echo "    brew install python"
    echo "  - Linux (Ubuntu/Debian):"
    echo "    sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo
    exit 1
fi

echo "[+] Python 3 installation detected."
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[-] Error: Python version 3.8 or higher is required."
    python3 --version
    exit 1
fi

# Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment (venv)..."
    python3 -m venv venv
else
    echo "[+] Virtual environment (venv) already exists."
fi

# Activate Virtual Environment & Install Dependencies
echo "[*] Activating virtual environment and upgrading pip..."
source venv/bin/activate
python3 -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "[*] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "[!] Warning: requirements.txt not found. Installing core packages manually..."
    pip install sentence-transformers flask numpy torch
fi

echo
echo "===================================================="
echo "             SETUP COMPLETED SUCCESSFULLY!"
echo "===================================================="
echo
echo "To start the background embedding daemon:"
echo "   python3 app.py"
echo
echo "To run vector codebase searches:"
echo "   python3 turbo_code.py search --query \"search term\""
echo
