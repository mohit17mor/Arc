$ErrorActionPreference = "Stop"

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    return $null
}

$pythonCmd = Get-PythonCommand
if (-not $pythonCmd) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        $reply = Read-Host "Python 3 is required. Install it with winget now? [Y/n]"
        if ($reply -match '^[Nn]$') {
            throw "Python is required to install Arc."
        }
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        $pythonCmd = Get-PythonCommand
    }
}

if (-not $pythonCmd) {
    throw "Python 3.10+ is required to install Arc."
}

$packageUrl = $env:ARC_INSTALL_WHEEL_URL
if (-not $packageUrl) {
    $packageUrl = "https://github.com/mohit17mor/Arc/archive/refs/heads/main.zip"
}

$tmpRoot = Join-Path $env:TEMP ("arc-bootstrap-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmpRoot | Out-Null
try {
    & $pythonCmd -m venv (Join-Path $tmpRoot "bootstrap-venv")
    $bootstrapPython = Join-Path $tmpRoot "bootstrap-venv\Scripts\python.exe"
    $bootstrapPip = Join-Path $tmpRoot "bootstrap-venv\Scripts\pip.exe"
    & $bootstrapPip install --upgrade pip | Out-Null
    & $bootstrapPip install $packageUrl
    & $bootstrapPython -m arc.install.bootstrap --wheel-url $packageUrl

    $arcBin = Join-Path $HOME ".arc\bin"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) {
        $userPath = ""
    }
    if (-not ($userPath.Split(";") -contains $arcBin)) {
        $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $arcBin } else { "$arcBin;$userPath" }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "Added $arcBin to your user PATH. Open a new terminal before running arc."
    }
    else {
        Write-Host "Arc installer finished. Run: arc init"
    }

    # Optional: TTS setup hint
    Write-Host ""
    Write-Host "Optional — Voice output (TTS):"
    Write-Host "  pip install arc-agent[tts]"
    Write-Host "  Then download Kokoro model files (~300MB, one-time):"
    Write-Host "    mkdir $HOME\.arc\models\kokoro"
    Write-Host "    cd $HOME\.arc\models\kokoro"
    Write-Host "    Invoke-WebRequest -Uri https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -OutFile kokoro-v1.0.onnx"
    Write-Host "    Invoke-WebRequest -Uri https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -OutFile voices-v1.0.bin"
    Write-Host "  Or skip this — Arc falls back to Windows built-in speech."
}
finally {
    Remove-Item -Recurse -Force $tmpRoot -ErrorAction SilentlyContinue
}
