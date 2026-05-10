<#
install.ps1 - one-shot setup for cull on Windows (PowerShell-native variant).

What it does:
  1. Verifies Python 3.10+ and git are on PATH.
  2. Creates a .venv\ if missing.
  3. Upgrades pip + installs requirements.txt - including gallery-dl
     pulled from Codeberg via the git+https:// pin in requirements.txt.
  4. Downloads Playwright's Chromium (one-time, used by the X scraper).
  5. Copies .env.example to .env on first run; never overwrites an
     existing .env.

Re-run safely: every step is idempotent. Doesn't launch anything;
run .\launch.bat (or boot via the venv) after install completes.

If PowerShell blocks execution, run once with the bypass:
  powershell -ExecutionPolicy Bypass -File .\install.ps1
#>
[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Write-Step { param([string]$Msg) Write-Host "[install] $Msg" }
function Write-Err  { param([string]$Msg) Write-Host "ERROR: $Msg" -ForegroundColor Red }

Write-Step "cull setup - installing into $ScriptDir"

# ── Required tools ─────────────────────────────────────────────────────────
$pyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCmd = @("py", "-3")
} elseif (Get-Command $Python -ErrorAction SilentlyContinue) {
    $pyCmd = @($Python)
} else {
    Write-Err "Python not found on PATH."
    Write-Host "  Install Python 3.10+ from python.org and retry."
    exit 1
}

$pyVersion = & $pyCmd[0] @($pyCmd[1..($pyCmd.Length - 1)] + @("-c", "import sys; print('%d.%d' % sys.version_info[:2])"))
$pyParts = $pyVersion.Split(".")
$pyMajor = [int]$pyParts[0]; $pyMinor = [int]$pyParts[1]
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 10)) {
    Write-Err "Python 3.10+ required, found $pyVersion"
    exit 1
}
Write-Step "Python $pyVersion OK"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Err "git not found on PATH."
    Write-Host "  cull installs gallery-dl from codeberg.org via 'pip install git+...',"
    Write-Host "  which requires the git CLI. Install Git for Windows and retry."
    exit 1
}
$gitVersion = (git --version).Split(" ")[2]
Write-Step "git $gitVersion OK"

# ── Virtualenv ─────────────────────────────────────────────────────────────
$VenvPath = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Step "creating virtualenv in .venv"
    & $pyCmd[0] @($pyCmd[1..($pyCmd.Length - 1)] + @("-m", "venv", $VenvPath))
} else {
    Write-Step "reusing existing .venv"
}

# Use the venv's pip/python directly. Avoids `Activate.ps1`'s execution-policy
# prompt and the side-effect of leaving the venv active in the user's session.
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

# ── Bootstrap pip ──────────────────────────────────────────────────────────
Write-Step "upgrading pip / setuptools / wheel"
& $VenvPython -m pip install --upgrade pip setuptools wheel | Out-Null

# ── Requirements (idempotent via hash file) ────────────────────────────────
$ReqPath = Join-Path $ScriptDir "requirements.txt"
$HashFile = Join-Path $VenvPath ".requirements.hash"
$NewHash = (Get-FileHash -Algorithm SHA256 $ReqPath).Hash
$OldHash = if (Test-Path $HashFile) { (Get-Content $HashFile -Raw).Trim() } else { "" }

if ($NewHash -ne $OldHash) {
    Write-Step "installing requirements (this includes gallery-dl from codeberg)"
    & $VenvPip install -r $ReqPath
    if ($LASTEXITCODE -ne 0) {
        Write-Err "pip install failed. See the output above."
        exit 1
    }
    Set-Content -Path $HashFile -Value $NewHash
} else {
    Write-Step "requirements already installed (hash unchanged)"
}

# ── Playwright Chromium (one-time) ─────────────────────────────────────────
if ($env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD -ne "1") {
    $PlaywrightFlag = Join-Path $VenvPath ".playwright_chromium_installed"
    if (-not (Test-Path $PlaywrightFlag)) {
        Write-Step "downloading Chromium for Playwright (one-time, ~120 MB)"
        & $VenvPython -m playwright install chromium
        if ($LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Path $PlaywrightFlag -Force | Out-Null
        } else {
            Write-Host "WARN: Playwright Chromium install failed. The X scraper won't work" -ForegroundColor Yellow
            Write-Host "      until you run 'python -m playwright install chromium' manually." -ForegroundColor Yellow
        }
    } else {
        Write-Step "Playwright Chromium already installed"
    }
}

# ── .env seed ──────────────────────────────────────────────────────────────
$EnvPath = Join-Path $ScriptDir ".env"
$EnvExample = Join-Path $ScriptDir ".env.example"
if (-not (Test-Path $EnvPath)) {
    Copy-Item $EnvExample $EnvPath
    Write-Step "created .env from .env.example"
    Write-Step ">>> edit .env to add your API keys (Civitai, Groq, Discord, etc.) <<<"
} else {
    Write-Step "reusing existing .env"
}

Write-Host ""
Write-Step "done. Next steps:"
Write-Host "  1. (optional) edit .env to add API keys"
Write-Host "  2. .\launch.bat    # boots the dashboard at http://localhost:5000"
