@echo off
title AI Waste Segregation - Virtual Conveyor Simulation Mode
cd /d "%~dp0"
echo =================================================================
echo       AI INDUSTRIAL WASTE SEGREGATION - VIRTUAL SIMULATION
echo =================================================================
echo Launching directly into Virtual Conveyor Simulation mode...
python main.py --simulate
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
