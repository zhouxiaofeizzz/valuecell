# PowerShell script to prepare Python environments

$ErrorActionPreference = "Stop"

function Write-Highlight($message) { Write-Host $message -ForegroundColor Blue }
function Write-Success($message) { Write-Host $message -ForegroundColor Green }
function Write-Warn($message) { Write-Host $message -ForegroundColor Yellow }
function Write-Err($message) { Write-Host $message -ForegroundColor Red }
function Highlight-Command($command) { Write-Highlight "Running: $command" }

function Get-PlaywrightBrowserRevision($browserName) {
    $browsersJsonPath = Join-Path (Get-Location) ".venv\Lib\site-packages\playwright\driver\package\browsers.json"
    if (-not (Test-Path $browsersJsonPath)) {
        return $null
    }

    $browsersJson = Get-Content $browsersJsonPath -Raw | ConvertFrom-Json
    $browser = $browsersJson.browsers | Where-Object { $_.name -eq $browserName } | Select-Object -First 1
    if ($null -eq $browser) {
        return $null
    }

    return $browser.revision
}

function Test-PlaywrightChromiumInstalled {
    $chromiumRevision = Get-PlaywrightBrowserRevision "chromium"
    $headlessRevision = Get-PlaywrightBrowserRevision "chromium-headless-shell"
    if (-not $chromiumRevision -or -not $headlessRevision) {
        return $false
    }

    $browserRoot = if ($env:PLAYWRIGHT_BROWSERS_PATH) {
        $env:PLAYWRIGHT_BROWSERS_PATH
    } else {
        Join-Path $env:LOCALAPPDATA "ms-playwright"
    }

    $chromiumDir = Join-Path $browserRoot "chromium-$chromiumRevision"
    $headlessDir = Join-Path $browserRoot "chromium_headless_shell-$headlessRevision"

    return (
        (Test-Path (Join-Path $chromiumDir "INSTALLATION_COMPLETE")) -and
        (Test-Path (Join-Path $chromiumDir "chrome-win64\chrome.exe")) -and
        (Test-Path (Join-Path $headlessDir "INSTALLATION_COMPLETE")) -and
        (Test-Path (Join-Path $headlessDir "chrome-headless-shell-win64\chrome-headless-shell.exe"))
    )
}

function Ensure-PlaywrightChromium {
    if (Test-PlaywrightChromiumInstalled) {
        Write-Success "Playwright Chromium is already installed, skipping browser download."
        return
    }

    Highlight-Command "uv run playwright install --with-deps chromium"
    uv run playwright install --with-deps chromium
}

# Check current directory and switch to python if needed
$currentPath = Get-Location
if ((Test-Path "python") -and (Test-Path "python\pyproject.toml") -and (Test-Path ".gitignore")) {
    Write-Warn "Detected project root. Switching to python directory..."
    Set-Location "python"
} elseif (-not (Test-Path "pyproject.toml")) {
    Write-Err "Error: This script must be run from the project python directory or project root. You are in $currentPath"
    exit 1
}

# Final check if in python directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Err "Error: Failed to switch to python directory. You are in $(Get-Location)"
    exit 1
}

# Check if uv is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Err "Error: 'uv' command not found. Please install 'uv' from https://docs.astral.sh/uv/"
    exit 1
}

Write-Highlight "=========================================="
Write-Highlight "Starting environment preparation..."
Write-Highlight "=========================================="

# Prepare main environment
Write-Success "Project root confirmed. Preparing environments..."

Write-Warn "Setting up main Python environment..."
if (-not (Test-Path ".venv")) {
    Highlight-Command "uv venv --python 3.12"
    uv venv --python 3.12
} else {
    Write-Warn ".venv already exists, skipping venv creation."
}
Highlight-Command "uv sync --group dev"
uv sync --group dev
Ensure-PlaywrightChromium
Write-Success "Main environment setup complete."

Write-Success "=========================================="
Write-Success "All environments are set up."
Write-Success "=========================================="

