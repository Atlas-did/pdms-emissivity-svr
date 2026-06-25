# codex-relay watchdog ? auto-restart on crash, run as hidden background process
# Usage:
#   powershell -WindowStyle Hidden -File start-codex-relay.ps1 -Daemon   (background)
#   .\start-codex-relay.ps1                                               (visible)

param([switch]$Daemon)

$ErrorActionPreference = "Stop"
$relay = Join-Path $env:USERPROFILE ".cargo\bin\codex-relay.exe"

if (-not (Test-Path $relay)) {
    Write-Host "[FATAL] codex-relay not found at $relay"
    exit 1
}

# Read API key from a local env file (never hardcode keys in scripts)
# Look for config in script directory first, then user profile
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configFile = Join-Path $scriptDir "..\.codex-relay.env"
if (-not (Test-Path $configFile)) {
    $configFile = Join-Path $env:USERPROFILE ".codex-relay.env"
}
if (-not (Test-Path $configFile)) {
    Write-Host "[FATAL] Config not found: .codex-relay.env (looked in script dir and ~)"
    Write-Host "Create it with these 2 lines:"
    Write-Host "  CODEX_RELAY_UPSTREAM=https://api.deepseek.com/v1"
    Write-Host "  CODEX_RELAY_API_KEY=sk-xxxxxxxx"
    exit 1
}
Get-Content $configFile | Where-Object { $_ -match '^\s*[A-Z_]+\s*=' -and $_ -notmatch '^\s*#' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
}
Write-Host "[OK] Loaded $configFile"

if ($Daemon) {
    $pid | Out-File (Join-Path $env:TEMP "codex-relay.pid") -NoNewline
    Write-Host "[DAEMON] PID=$pid, crash-log: $env:TEMP\codex-relay-crash.log"
}

$restartDelay = 3
$maxCrashes = 10
$crashWindowSec = 60
$crashTimestamps = @()

while ($true) {
    # Remove old crash records outside the window
    $now = Get-Date
    $crashTimestamps = @($crashTimestamps | Where-Object { $now - $_ | ForEach-Object TotalSeconds -lt $crashWindowSec })
    if ($crashTimestamps.Count -ge $maxCrashes) {
        Write-Host "[FATAL] $maxCrashes crashes in ${crashWindowSec}s, giving up"
        exit 1
    }

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[START] $ts ? codex-relay :4444"
    $proc = Start-Process -FilePath $relay -NoNewWindow -PassThru
    $proc.WaitForExit()
    $crashTimestamps += Get-Date
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[CRASH] $ts (exit=$($proc.ExitCode)), restart in ${restartDelay}s"
    Start-Sleep $restartDelay
}
