@echo off
:: ============================================================
::  HackNEX 2026 -- On-Device Conversational Stack (HNX26EPS08)
::  MASTER STARTUP SCRIPT  --  Double-click to launch full project
:: ============================================================
title HackNEX 2026 - EdgeVoice Master Launcher
color 0A
chcp 65001 >nul
setlocal EnableDelayedExpansion
cls

echo.
echo  ====================================================================
echo    HackNEX 2026 -- EdgeVoice AI Master Launcher
echo    HNX26EPS08   Offline Edge AI   100%% CPU   0 Cloud Calls
echo  ====================================================================
echo.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BACKEND_DIR=%ROOT%\hacknex-edge"
set "FRONTEND_DIR=%ROOT%\person 4\ai\frontend"
set "SKIP_FRONTEND=0"

:: === 1. Check Python ===
echo  [1/5] Checking Python...
python --version >nul 2>nul
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.11+ and retry.
    pause
    exit /b 1
)
echo         Python OK.

:: === 2. Check Node.js ===
echo.
echo  [2/5] Checking Node.js...
node --version >nul 2>nul
if errorlevel 1 (
    echo  [WARNING] Node.js not found. Skipping frontend.
    set "SKIP_FRONTEND=1"
)
if "!SKIP_FRONTEND!"=="0" echo         Node.js OK.

:: === 3. npm install (if needed) ===
echo.
if "!SKIP_FRONTEND!"=="1" goto :skip_npm
echo  [3/5] Checking frontend node_modules...
if not exist "!FRONTEND_DIR!\node_modules" (
    echo         Running npm install (first time only^)...
    cmd /c "cd /d "!FRONTEND_DIR!" && npm.cmd install"
) else (
    echo         node_modules already present.
)
goto :start_backend

:skip_npm
echo  [3/5] Skipped (no Node.js).

:start_backend
:: === 4. Start Backend ===
echo.
echo  [4/5] Starting EdgeVoice Backend (FastAPI + Dashboard) on port 8000...
start "EdgeVoice Backend" cmd /k "cd /d "!BACKEND_DIR!" && color 0B && chcp 65001 >nul && title EdgeVoice Backend [http://localhost:8000] && python -u main.py"
echo         Waiting 4 seconds for backend to initialize...
timeout /t 4 /nobreak >nul

:: === 5. Start Frontend ===
echo.
if "!SKIP_FRONTEND!"=="1" goto :skip_frontend
echo  [5/5] Starting Vite React Frontend on port 5173...
start "EdgeVoice Frontend" cmd /k "cd /d "!FRONTEND_DIR!" && color 0D && chcp 65001 >nul && title EdgeVoice Frontend [http://localhost:5173] && npm.cmd run dev"
timeout /t 3 /nobreak >nul
goto :done

:skip_frontend
echo  [5/5] Skipped frontend (Node.js not found).

:done
echo.
echo  ====================================================================
echo   All services launched!
echo   Backend  (FastAPI Dashboard + WS) : http://localhost:8000
if "!SKIP_FRONTEND!"=="0" (
    echo   Frontend (React / Vite UI^)        : http://localhost:5173
)
echo  ====================================================================
echo.

start "" "http://localhost:8000"
if "!SKIP_FRONTEND!"=="0" (
    timeout /t 2 /nobreak >nul
    start "" "http://localhost:5173"
)

echo  Press any key to close this launcher window.
echo  (Backend and Frontend will keep running in their own windows.)
pause >nul
exit /b 0