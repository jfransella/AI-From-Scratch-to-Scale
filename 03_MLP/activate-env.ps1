# Cross-platform virtual environment activation script for PowerShell
Write-Host "Activating MLP virtual environment..." -ForegroundColor Green

if ($IsWindows -or $env:OS -eq "Windows_NT") {
    # Windows
    & ".\.venv\Scripts\Activate.ps1"
} else {
    # macOS/Linux
    & "source .venv/bin/activate"
}

Write-Host "Virtual environment activated!" -ForegroundColor Green
