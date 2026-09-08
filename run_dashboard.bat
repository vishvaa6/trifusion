@echo off
title AI Waste Segregation - Web SCADA Dashboard
cd /d "%~dp0"
echo =================================================================
echo    AI INDUSTRIAL WASTE SEGREGATION - WEB SCADA DASHBOARD
echo =================================================================
echo Starting Web Dashboard server at http://localhost:5000 ...
start http://localhost:5000
python dashboard.py
pause
