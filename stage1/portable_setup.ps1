param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

# This script lives in <ROOT>/stage1; ROOT is its parent
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "[portable_setup] Root: $Root"

# Create venv if missing
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
  Write-Host "[portable_setup] Creating venv at $VenvDir"
  & $PythonExe -m venv $VenvDir
}

Write-Host "[portable_setup] Upgrading pip"
& $VenvPython -m pip install -U pip

$Req = Join-Path $PSScriptRoot "portable_requirements.txt"
Write-Host "[portable_setup] Installing requirements from $Req"
& $VenvPython -m pip install -r $Req

Write-Host "[portable_setup] Done. Next: run .\stage1\portable_run.ps1"
