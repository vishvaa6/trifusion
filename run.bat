@echo off
title AI Waste Segregation - 5-Door Smart Routing System
cd /d "%~dp0"
echo =================================================================
echo       AI INDUSTRIAL WASTE SEGREGATION - 5-DOOR ROUTING SYSTEM
echo =================================================================
echo Launching with Webcam detection and Virtual Conveyor Simulation...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
