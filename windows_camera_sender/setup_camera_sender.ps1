$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.10 or newer and make it available as 'python'."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Windows camera sender environment."
    }
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in the camera sender environment."
}

& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Windows camera sender dependencies."
}

Write-Host "Setup complete. Start with: .\run_camera_sender.ps1"
