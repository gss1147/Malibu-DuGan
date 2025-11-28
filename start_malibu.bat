@echo off
title MALIBU DUGAN AI - 100% FUNCTIONAL
cd /d X:\Malibu_DuGan

echo.
echo ===================================================
echo     MALIBU DUGAN AI v4.0 - FULL SYSTEM
echo ===================================================
echo.

:: Set environment
set PYTHONPATH=X:\Malibu_DuGan;X:\Malibu_DuGan\AI_Python
set KMP_DUPLICATE_LIB_OK=TRUE

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Quick dependency check
echo Checking dependencies...
python -c "import numpy, torch, sklearn, PyQt5" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install numpy==1.24.3 torch==2.0.1+cpu torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu scikit-learn PyQt5 pyyaml pillow
)

:: Start the AI system
echo.
echo Starting Malibu DuGan AI...
echo.
python main.py

:: If main.py fails, show error
if errorlevel 1 (
    echo.
    echo Application exited with errors.
    echo.
)

pause