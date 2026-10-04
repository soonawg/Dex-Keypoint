param(
    [int]$Camera = 0,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Windows camera sender is not set up. Run .\setup_camera_sender.ps1 first."
}

& $python .\camera_sender.py --camera $Camera --host $HostAddress --port $Port
exit $LASTEXITCODE
