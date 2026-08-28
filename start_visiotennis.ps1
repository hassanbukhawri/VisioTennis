# start_visiotennis.ps1 -- one-shot dev startup routine (2026-08-28).
#
# Run this once per sitting instead of remembering the backend/Ollama/Funnel
# sequence by hand. Idempotent: safe to re-run if some pieces are already up
# (each step checks before acting, never starts a second backend/Ollama).
#
# What it does, in order:
#   1. Confirms Ollama is up and qwen3:14b is loaded (starts Ollama if not).
#   2. Confirms the backend is up on port 8000 (starts it if not).
#   3. Confirms Tailscale Funnel is pointed at port 8000 (repoints if not --
#      this is the exact port-mismatch class of bug that caused the 502s
#      logged in PROJECT_LOG.md 2026-08-07/08-21/08-28).
#   4. Verifies the public Funnel URL actually responds, then prints it --
#      that's the value to paste into mobile/lib/config.js's BACKEND_URL.

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
Set-Location $repoRoot

$funnelUrl = "https://visiotennis-trial.tail458008.ts.net"
$ollamaExe = "C:\Users\PC\AppData\Local\Programs\Ollama\ollama.exe"

# TRIAL_PASSCODE (2026-08-28) -- backend/app.py now requires this from the
# environment, not hardcoded (see backend/.env.example for why). Load it from
# the untracked backend/.env here so the backend process started below
# inherits it, without ever writing the real value into this tracked script.
$envFile = Join-Path $repoRoot "backend\.env"
if (-not (Test-Path $envFile)) {
    Write-Host "ERROR: backend\.env not found. Copy backend\.env.example to backend\.env and fill in TRIAL_PASSCODE first." -ForegroundColor Red
    exit 1
}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}
if (-not $env:TRIAL_PASSCODE) {
    Write-Host "ERROR: TRIAL_PASSCODE not set after loading backend\.env. Check the file's contents." -ForegroundColor Red
    exit 1
}

function Test-HttpOk {
    # Treats 401 (auth-gated but reachable) as success, not just 2xx.
    param([string]$Uri, [int]$TimeoutSec = 5)
    try {
        Invoke-WebRequest -Uri $Uri -TimeoutSec $TimeoutSec -UseBasicParsing | Out-Null
        return $true
    } catch {
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode.value__ -eq 401) { return $true }
        return $false
    }
}

Write-Host "== VisioTennis startup ==" -ForegroundColor Cyan

# --- 1. Ollama ---
Write-Host "[1/4] Ollama..." -ForegroundColor Yellow
$ollamaUp = $false
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $ollamaUp = $true
} catch { $ollamaUp = $false }

if (-not $ollamaUp) {
    Write-Host "  Not responding -- starting it..." -ForegroundColor Yellow
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    try {
        $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
        $ollamaUp = $true
    } catch {
        Write-Host "  ERROR: Ollama still not responding. Start it manually and re-run this script." -ForegroundColor Red
        exit 1
    }
}
$hasModel = $tags.models | Where-Object { $_.name -eq "qwen3:14b" }
if ($hasModel) {
    Write-Host "  OK -- qwen3:14b loaded." -ForegroundColor Green
} else {
    $names = if ($tags.models) { ($tags.models | ForEach-Object { $_.name }) -join ", " } else { "(none)" }
    Write-Host "  WARNING: Ollama is up but qwen3:14b is not loaded (found: $names)." -ForegroundColor Red
}

# --- 2. Backend ---
Write-Host "[2/4] Backend (port 8000)..." -ForegroundColor Yellow
if (Test-HttpOk -Uri "http://localhost:8000/docs") {
    Write-Host "  OK -- already running." -ForegroundColor Green
} else {
    Write-Host "  Not running -- starting it..." -ForegroundColor Yellow
    $log = Join-Path $repoRoot "backend_startup.log"
    Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $repoRoot -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    Start-Sleep -Seconds 6
    if (Test-HttpOk -Uri "http://localhost:8000/docs") {
        Write-Host "  OK -- started." -ForegroundColor Green
    } else {
        Write-Host "  ERROR: backend still not responding after starting. Check $log / $log.err" -ForegroundColor Red
        exit 1
    }
}

# --- 3. Tailscale Funnel ---
Write-Host "[3/4] Tailscale Funnel..." -ForegroundColor Yellow
$funnelStatus = & tailscale funnel status 2>&1 | Out-String
if ($funnelStatus -match "127\.0\.0\.1:8000") {
    Write-Host "  OK -- already pointed at port 8000." -ForegroundColor Green
} else {
    Write-Host "  Not pointed at 8000 (or off) -- fixing..." -ForegroundColor Yellow
    & tailscale funnel --bg --yes 8000 2>&1 | Out-Null
    Start-Sleep -Seconds 2
    $funnelStatus = & tailscale funnel status 2>&1 | Out-String
    if ($funnelStatus -match "127\.0\.0\.1:8000") {
        Write-Host "  OK -- repointed." -ForegroundColor Green
    } else {
        Write-Host "  ERROR: could not confirm Funnel is pointed at 8000. Run 'tailscale funnel status' manually." -ForegroundColor Red
        exit 1
    }
}

# --- 4. Verify public URL ---
Write-Host "[4/4] Verifying public URL..." -ForegroundColor Yellow
if (Test-HttpOk -Uri "$funnelUrl/docs" -TimeoutSec 10) {
    Write-Host "  OK -- reachable." -ForegroundColor Green
} else {
    Write-Host "  WARNING: public URL did not respond as expected. Backend/Funnel may need a moment, or check your Tailscale connection." -ForegroundColor Red
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Backend URL (paste into mobile/lib/config.js):" -ForegroundColor Cyan
Write-Host " $funnelUrl" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor Cyan
