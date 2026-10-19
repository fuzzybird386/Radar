# Create the conda env "dl" (Python 3.12) on Windows PowerShell.
# Installs Miniconda into $env:USERPROFILE\miniconda3 when conda is missing.
# macOS and Linux: setup_dl_env.sh
# PyTorch is the default pip wheel for Windows, not a CUDA build.
param(
    [string]$EnvName = "dl"
)

$ErrorActionPreference = "Stop"
$MinicondaPrefix = if ($env:MINICONDA_PREFIX) { $env:MINICONDA_PREFIX } else { Join-Path $env:USERPROFILE "miniconda3" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Import-Conda {
    $hook = Join-Path $MinicondaPrefix "shell\condabin\conda-hook.ps1"
    if (Test-Path $hook) {
        . $hook
        return $true
    }
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        return $true
    }
    return $false
}

function Install-Miniconda {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    $name = switch ($arch) {
        "X64" { "Miniconda3-latest-Windows-x86_64.exe" }
        "Arm64" { "Miniconda3-latest-Windows-arm64.exe" }
        default { throw "unsupported Windows architecture: $arch" }
    }
    $url = "https://repo.anaconda.com/miniconda/$name"
    $work = Join-Path ([System.IO.Path]::GetTempPath()) ("miniconda-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $work | Out-Null
    $installer = Join-Path $work $name
    Write-Host "installing Miniconda to $MinicondaPrefix"
    Invoke-WebRequest -Uri $url -OutFile $installer
    # /D= must be the last argument and must not be wrapped in extra quotes.
    $arguments = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$MinicondaPrefix"
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
    Remove-Item -Recurse -Force $work
    if ($process.ExitCode -ne 0) {
        throw "Miniconda installer exited with code $($process.ExitCode)"
    }
    $hook = Join-Path $MinicondaPrefix "shell\condabin\conda-hook.ps1"
    . $hook
    conda init powershell | Out-Host
    conda init cmd.exe | Out-Host
}

$Requirements = Join-Path $ScriptDir "requirements-dl.txt"
if (-not (Test-Path $Requirements)) {
    throw "missing $Requirements. Put requirements-dl.txt next to this script, then run it again."
}

if (-not (Import-Conda)) {
    Install-Miniconda
}

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
$existing = conda env list | ForEach-Object { ($_ -split "\s+")[0] }
if ($existing -contains $EnvName) {
    Write-Host "conda env '$EnvName' already exists, installing packages"
} else {
    conda create -y -n $EnvName python=3.12
}
conda activate $EnvName
python -m pip install -U pip
python -m pip install -r $Requirements
# d2l 1.0.3 pins jupyter==1.0.0, numpy==1.23.5, scipy==1.10.1,
# pandas==2.0.3, and matplotlib==3.7.2. Those pins conflict with
# requirements-dl.txt, and numpy 1.23.5 does not install on Python 3.12.
python -m pip install --no-deps "d2l==1.0.3"
python -m ipykernel install --user --name $EnvName --display-name "Python ($EnvName)"
python -c "import numpy, scipy, torch, mmwave, d2l; print('ok', numpy.__version__, torch.__version__)"
