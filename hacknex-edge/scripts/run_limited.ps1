# scripts/run_limited.ps1 — Runs Voice Pipeline with Enforced Resource Limits
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\.."
Set-Location $ProjectRoot

Write-Host "Enforcing hardware limits (4 CPU Cores, 3072 MB RAM)..." -ForegroundColor Cyan
python -c "
from src.edge.limits import enforce_limits
p = enforce_limits(4, 3072)
print('Active cores:', p['effective_cores'], 'RAM cap:', p['declared_ram_cap_mb'], 'MB, Status:', p['status'])
"

Write-Host "`nStarting EdgeVoice Pipeline... Press Ctrl+C to stop.`n" -ForegroundColor Green
python -c "
from src.edge.pipeline import build_pipeline
pipe = build_pipeline()
pipe.run_until_quit()
"
