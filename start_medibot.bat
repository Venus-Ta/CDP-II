@echo off
echo Starting MediBot...
cd /d "%~dp0"
start "MediBot Server" cmd /k "python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak > nul
start http://localhost:8000