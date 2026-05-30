param(
  [ValidateSet('inspect','simulate','train','evaluate','full')]
  [string]$Mode = "inspect",

  [string]$ConfigFile = "stage1/simulation_data/config.json"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  throw "Missing venv python: $VenvPython. Run .\stage1\portable_setup.ps1 first."
}

Write-Host "[portable_run] Root: $Root"
Write-Host "[portable_run] Mode: $Mode"
Write-Host "[portable_run] Config: $ConfigFile"

$Stage1 = Join-Path $Root "stage1\main_pdms_svr_bandscan_full.py"
$Stage2 = Join-Path $Root "stage2\分区图片1.0.py"

function Invoke-Stage1([string]$Action) {
  & $VenvPython $Stage1 --action $Action --config-file $ConfigFile
  if ($LASTEXITCODE -ne 0) { throw "Stage1 action failed: $Action" }
}

if ($Mode -eq "inspect") {
  Invoke-Stage1 "inspect_regions"
  exit 0
}

if ($Mode -eq "simulate") {
  Invoke-Stage1 "simulate"
  exit 0
}

if ($Mode -eq "train") {
  Invoke-Stage1 "train"
  exit 0
}

if ($Mode -eq "evaluate") {
  Invoke-Stage1 "evaluate"
  exit 0
}

# full
Invoke-Stage1 "simulate"
Invoke-Stage1 "train"
Invoke-Stage1 "evaluate"

Write-Host "[portable_run] Running Stage2 figures..."
& $VenvPython $Stage2
if ($LASTEXITCODE -ne 0) { throw "Stage2 failed" }

Write-Host "[portable_run] Done. Outputs: .\stage1\simulation_data\ and .\figures\"
