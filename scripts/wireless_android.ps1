# Pair and connect a physical Android device over Wi-Fi, then run AquaScan.
# Usage:
#   .\scripts\wireless_android.ps1 -PairHost "192.168.1.10:37123" -PairCode "123456" -ConnectHost "192.168.1.10:5555"
param(
  [Parameter(Mandatory = $true)]
  [string] $PairHost,
  [Parameter(Mandatory = $true)]
  [string] $PairCode,
  [Parameter(Mandatory = $true)]
  [string] $ConnectHost,
  [string] $BackendUrl = "http://172.18.135.26:8000"
)

$ErrorActionPreference = "Stop"
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
if (-not (Test-Path $adb)) {
  throw "adb not found at $adb. Install Android SDK platform-tools."
}

& $adb kill-server | Out-Null
& $adb start-server | Out-Null

Write-Host "Pairing with $PairHost ..."
& $adb pair $PairHost $PairCode
if ($LASTEXITCODE -ne 0) { throw "adb pair failed (exit $LASTEXITCODE)" }

Write-Host "Connecting to $ConnectHost ..."
& $adb connect $ConnectHost
if ($LASTEXITCODE -ne 0) { throw "adb connect failed (exit $LASTEXITCODE)" }

& $adb devices -l
$deviceLine = (& $adb devices) | Where-Object { $_ -match "device$" -and $_ -notmatch "List of" } | Select-Object -First 1
if (-not $deviceLine) { throw "No device in 'device' state after connect." }
$deviceId = ($deviceLine -split "\s+")[0]
Write-Host "Using device: $deviceId"

Set-Location (Join-Path $PSScriptRoot "..")
flutter pub get
flutter run -d $deviceId --dart-define=AQUASCAN_BACKEND_URL=$BackendUrl
