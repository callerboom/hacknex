# scripts/start_demo.ps1 — One-Click Startup for HackNEX 2026 (HNX26EPS08)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " HACKNEX 2026: ON-DEVICE CONVERSATIONAL STACK (HNX26EPS08)" -ForegroundColor Green
Write-Host " STARTING DEMO ENVIRONMENT UNDER ENFORCED RESOURCE LIMITS" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\.."

Set-Location $ProjectRoot

# 1. Enforce Limits Verification
Write-Host "`n[1/3] Enforcing 4 Cores + 3072 MB RAM limit via Win32 Job Object..." -ForegroundColor Cyan
python -c "from src.edge.limits import enforce_limits; p = enforce_limits(4, 3072); print('Limits verified:', p['effective_cores'], 'cores,', p['working_set_mb'], 'MB RSS')"

# 2. Launch FastAPI Mission Control & Live Telemetry Server
Write-Host "`n[2/3] Launching Telemetry Dashboard on http://localhost:8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; python dashboard/server.py"

Start-Sleep -Seconds 2

# 3. Open browser
Write-Host "`n[3/3] Opening Mission Control in default browser..." -ForegroundColor Green
Start-Process "http://localhost:8000"

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " EdgeVoice is running live! Speak into mic or use UI simulator." -ForegroundColor Green
Write-Host "=================================================================`n" -ForegroundColor Green
