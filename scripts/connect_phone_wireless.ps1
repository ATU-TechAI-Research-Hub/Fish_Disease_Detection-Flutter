# Pair wireless debugging, connect, and forward backend port to the phone.
# Get Pair/Connect hosts from: Settings → Developer options → Wireless debugging
param(
  [Parameter(Mandatory = $true)]
  [string] $PairHost,
  [Parameter(Mandatory = $true)]
  [string] $PairCode,
  [Parameter(Mandatory = $true)]
  [string] $ConnectHost
)

$ErrorActionPreference = "Stop"
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
if (-not (Test-Path $adb)) { throw "adb not found at $adb" }

& $adb kill-server | Out-Null
& $adb start-server | Out-Null

Write-Host "Pairing $PairHost ..."
& $adb pair $PairHost $PairCode
if ($LASTEXITCODE -ne 0) { throw "adb pair failed" }

Write-Host "Connecting $ConnectHost ..."
& $adb connect $ConnectHost
if ($LASTEXITCODE -ne 0) { throw "adb connect failed" }

& $adb reverse tcp:8000 tcp:8000
& $adb devices -l
Write-Host "Backend tunnel: phone http://127.0.0.1:8000 -> PC port 8000"
