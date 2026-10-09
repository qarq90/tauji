@echo off
title Tauji Launcher
cd /d "%~dp0"

echo ========================================
echo   Starting Tauji...
echo ========================================
echo.

python -m streamlit run app.py

echo.
echo App stopped. Press any key to close.
pause >nul