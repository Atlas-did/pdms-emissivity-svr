# network-diag.ps1 ? diagnose & fix GitHub access issues
# Usage: .\network-diag.ps1          (diagnose only)
#        .\network-diag.ps1 -Fix     (auto-fix common problems)

param([switch]$Fix)

$ErrorActionPreference = "Continue"

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Network Diagnostic Tool"                 -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

$issues = @()
$ok = @()

# ?? 1. System Proxy ??
Write-Host "[1] System Proxy" -ForegroundColor Yellow
$proxy = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -ErrorAction SilentlyContinue
$proxyOn = $proxy.ProxyEnable -eq 1
$proxyServer = $proxy.ProxyServer
Write-Host "    ProxyEnable = $proxyOn"
Write-Host "    ProxyServer = '$proxyServer'"

if (-not $proxyOn) {
    $issues += "System proxy is OFF ? browser goes direct, GFW will block GitHub"
    if ($Fix) {
        Set-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -Name ProxyEnable -Value 1 -Type DWord -Force
        Write-Host "    -> FIXED: ProxyEnable = 1" -ForegroundColor Green
    }
} else {
    $ok += "System proxy ON"
}
if (-not $proxyServer -or $proxyServer -notmatch "31181") {
    $issues += "ProxyServer not pointing to 127.0.0.1:31181"
    if ($Fix) {
        Set-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -Name ProxyServer -Value "127.0.0.1:31181" -Type String -Force
        Write-Host "    -> FIXED: ProxyServer = 127.0.0.1:31181" -ForegroundColor Green
    }
} else {
    $ok += "ProxyServer correct (127.0.0.1:31181)"
}

# ?? 2. Registry env vars (setx pollution) ??
Write-Host "`n[2] Registry Environment Variables" -ForegroundColor Yellow
$http = [Environment]::GetEnvironmentVariable("HTTP_PROXY", "User")
$https = [Environment]::GetEnvironmentVariable("HTTPS_PROXY", "User")
if ($http) { Write-Host "    HTTP_PROXY  = $http  <- DANGEROUS (persistent)" -ForegroundColor Red; $issues += "HTTP_PROXY set in registry: $http" }
if ($https) { Write-Host "    HTTPS_PROXY = $https <- DANGEROUS (persistent)" -ForegroundColor Red; $issues += "HTTPS_PROXY set in registry: $https" }
if (-not $http -and -not $https) { $ok += "No registry proxy pollution"; Write-Host "    (clean)" -ForegroundColor Green }

# ?? 3. dev-sidecar process ??
Write-Host "`n[3] dev-sidecar Process" -ForegroundColor Yellow
$dcProcs = @(Get-Process -Name "dev-sidecar" -ErrorAction SilentlyContinue)
if ($dcProcs.Count -eq 0) {
    $issues += "dev-sidecar NOT running"
    Write-Host "    NOT RUNNING" -ForegroundColor Red
} elseif ($dcProcs.Count -gt 2) {
    $issues += "dev-sidecar has $($dcProcs.Count) instances (should be 1-2)"
    Write-Host "    $($dcProcs.Count) instances (PID: $($dcProcs.Id -join ', '))" -ForegroundColor Yellow
    if ($Fix) {
        Stop-Process -Name "dev-sidecar" -Force -ErrorAction SilentlyContinue
        Start-Sleep 1
        Write-Host "    -> Killed all dev-sidecar processes. Restart from system tray." -ForegroundColor Green
    }
} else {
    $ok += "dev-sidecar running ($($dcProcs.Count) instances)"
    Write-Host "    RUNNING (PID: $($dcProcs.Id -join ', '))" -ForegroundColor Green
}

# ?? 4. Port 31181 ??
Write-Host "`n[4] Port 31181" -ForegroundColor Yellow
$port = netstat -ano | Select-String "31181.*LISTENING"
if ($port) {
    $ok += "Port 31181 listening"
    Write-Host "    LISTENING" -ForegroundColor Green
} else {
    $issues += "Port 31181 NOT listening ? dev-sidecar proxy dead"
    Write-Host "    NOT LISTENING" -ForegroundColor Red
}

# ?? 5. codex-relay ??
Write-Host "`n[5] codex-relay (port 4444)" -ForegroundColor Yellow
$relay = netstat -ano | Select-String "4444.*LISTENING"
if ($relay) {
    $ok += "codex-relay on :4444"
    Write-Host "    LISTENING" -ForegroundColor Green
} else {
    $issues += "codex-relay NOT running on :4444 ? Codex will fail"
    Write-Host "    NOT LISTENING" -ForegroundColor Red
}

# ?? 6. Connectivity tests ??
Write-Host "`n[6] Connectivity Tests" -ForegroundColor Yellow
$tests = @(
    @{name="Direct Baidu";        url="https://www.baidu.com";        proxy=$false},
    @{name="Via proxy Baidu";     url="https://www.baidu.com";        proxy=$true},
    @{name="Via proxy GitHub";    url="https://github.com";           proxy=$true},
    @{name="Via proxy gh-raw";    url="https://raw.githubusercontent.com"; proxy=$true}
)

foreach ($t in $tests) {
    $parms = @{ Uri = $t.url; TimeoutSec = 5; UseBasicParsing = $true; ErrorAction = "SilentlyContinue" }
    if ($t.proxy) { $parms.Proxy = "http://127.0.0.1:31181" }
    try {
        $r = Invoke-WebRequest @parms
        Write-Host "    [OK] $($t.name): HTTP $($r.StatusCode)" -ForegroundColor Green
        $ok += "$($t.name) OK"
    } catch {
        $msg = $_.Exception.Message -replace '\n.*', ''
        Write-Host "    [FAIL] $($t.name): $msg" -ForegroundColor Red
        $issues += "$($t.name): $msg"
    }
}

# ?? 7. Script injection file ??
Write-Host "`n[7] Script Injection File" -ForegroundColor Yellow
$scriptPath = "D:\Software_download_link\DEVsidecar\dev-sidecar\resources\extra\scripts\github.script"
if (Test-Path $scriptPath) {
    $len = (Get-Item $scriptPath).Length
    if ($len -gt 10) {
        $issues += "github.script is $len bytes ? script injection ACTIVE (causes garbled text!)"
        Write-Host "    SIZE: $len bytes <- DANGER: will inject code into GitHub responses" -ForegroundColor Red
        if ($Fix) {
            Stop-Process -Name "dev-sidecar" -Force -ErrorAction SilentlyContinue
            Start-Sleep 1
            @() | Set-Content $scriptPath -Force -ErrorAction SilentlyContinue
            Write-Host "    -> FIXED: cleared to 0 bytes (restart dev-sidecar to take effect)" -ForegroundColor Green
        }
    } else {
        $ok += "github.script empty/small ($len bytes)"
        Write-Host "    $len bytes (safe)" -ForegroundColor Green
    }
} else {
    $ok += "No github.script file"
    Write-Host "    (file does not exist)" -ForegroundColor Green
}

# ?? 8. overwall check ??
Write-Host "`n[8] overwall Status" -ForegroundColor Yellow
# Try to detect overwall via dev-sidecar API
try {
    $status = Invoke-RestMethod -Uri "http://127.0.0.1:31181/api/overwall" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "    overwall: $status" -ForegroundColor $(if ($status -match "true|on|1") { "Green" } else { "Red" })
} catch {
    Write-Host "    Cannot query ? check dev-sidecar tray icon -> overwall is ON?" -ForegroundColor Yellow
    Write-Host "    If overwall is OFF, GitHub WILL be blocked by GFW." -ForegroundColor Red
    $issues += "Cannot confirm overwall status ? check dev-sidecar tray icon"
}

# ?? Summary ??
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OK:      $($ok.Count) items" -ForegroundColor Green
$ok | ForEach-Object { Write-Host "    [OK] $_" -ForegroundColor Green }
Write-Host "  ISSUES:  $($issues.Count) items" -ForegroundColor Red
$issues | ForEach-Object { Write-Host "    [!!] $_" -ForegroundColor Red }

if ($issues.Count -eq 0) {
    Write-Host "`n  All checks passed!" -ForegroundColor Green
} else {
    Write-Host "`n  Run with -Fix to auto-fix: .\network-diag.ps1 -Fix" -ForegroundColor Yellow
    Write-Host "  Then restart dev-sidecar from system tray and ensure OVERWALL = ON" -ForegroundColor Yellow
}
