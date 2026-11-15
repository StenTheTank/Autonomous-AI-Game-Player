param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "   AI Game Player — Autonomous Vision Agent v2.0     " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

if ($Install) {
    Write-Host "`n[1/2] Installing backend dependencies & Playwright Chromium..." -ForegroundColor Yellow
    Push-Location $BackendDir
    pip install -r requirements.txt
    playwright install chromium
    Pop-Location

    Write-Host "`n[2/2] Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    npm install
    Pop-Location
}

Write-Host "`nLaunching FastAPI + WebSocket Engine (http://localhost:8000)..." -ForegroundColor Green
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Set-Location '$BackendDir'; python main.py"

Write-Host "Launching React Mission Control UI (http://localhost:5173)..." -ForegroundColor Green
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Set-Location '$FrontendDir'; npm run dev"

Write-Host ""
Write-Host "All services launched!" -ForegroundColor Cyan
Write-Host "  • Dashboard UI:        http://localhost:5173"
Write-Host "  • Built-in Demo Game:  http://localhost:8000/demo/crystal-quest"
Write-Host "  • Mouse Demo Game:     http://localhost:8000/demo/sequence-strike"
Write-Host ""
Write-Host "Tip: Run './start.ps1 -Install' on a fresh machine to install packages & Chromium." -ForegroundColor DarkGray
