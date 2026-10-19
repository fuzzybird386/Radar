#!/usr/bin/env bash
# Create the conda env "dl" (Python 3.12) on macOS, Linux, or Windows Git Bash.
# Installs Miniconda into $HOME/miniconda3 when conda is missing.
# Native Windows PowerShell: setup_dl_env.ps1
# PyTorch is the default pip wheel for the current OS, not a CUDA build.
set -euo pipefail

ENV_NAME="${1:-dl}"
MINICONDA_PREFIX="${MINICONDA_PREFIX:-$HOME/miniconda3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

os_family() {
  case "$(uname -s)" in
    Darwin) echo darwin ;;
    Linux) echo linux ;;
    MINGW* | MSYS* | CYGWIN*) echo windows ;;
    *) echo unknown ;;
  esac
}

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  local base
  for base in "$MINICONDA_PREFIX" "$HOME/anaconda3" "$HOME/miniforge3"; do
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      return 0
    fi
  done
  return 1
}

miniconda_name() {
  local family="$1" arch="$2"
  case "$family-$arch" in
    darwin-arm64) echo "Miniconda3-latest-MacOSX-arm64.sh" ;;
    darwin-x86_64) echo "Miniconda3-latest-MacOSX-x86_64.sh" ;;
    linux-x86_64) echo "Miniconda3-latest-Linux-x86_64.sh" ;;
    linux-aarch64 | linux-arm64) echo "Miniconda3-latest-Linux-aarch64.sh" ;;
    windows-x86_64 | windows-amd64) echo "Miniconda3-latest-Windows-x86_64.exe" ;;
    windows-aarch64 | windows-arm64) echo "Miniconda3-latest-Windows-arm64.exe" ;;
    *)
      echo "unsupported platform: $family $arch" >&2
      exit 1
      ;;
  esac
}

install_miniconda() {
  local family arch name url work installer
  family="$(os_family)"
  arch="$(uname -m)"
  name="$(miniconda_name "$family" "$arch")"
  url="https://repo.anaconda.com/miniconda/${name}"
  work="$(mktemp -d)"
  installer="$work/$name"
  echo "installing Miniconda to $MINICONDA_PREFIX"
  curl -fsSL "$url" -o "$installer"
  if [[ "$family" == "windows" ]]; then
    install_miniconda_windows "$installer"
  else
    chmod +x "$installer"
    # Execute via shebang. `bash installer.sh` makes macOS bash 3.2 look sourced.
    "$installer" -b -p "$MINICONDA_PREFIX"
  fi
  rm -rf "$work"
  # shellcheck disable=SC1091
  source "$MINICONDA_PREFIX/etc/profile.d/conda.sh"
  conda init bash
  if [[ "$(basename "${SHELL:-bash}")" == "zsh" ]]; then
    conda init zsh
  fi
  if [[ "$family" == "windows" ]]; then
    conda init powershell || true
    conda init cmd.exe || true
  fi
}

install_miniconda_windows() {
  local installer="$1" win_installer win_prefix
  command -v cygpath >/dev/null 2>&1 || {
    echo "Windows Git Bash needs cygpath. Or run setup_dl_env.ps1 in PowerShell." >&2
    exit 1
  }
  command -v powershell.exe >/dev/null 2>&1 || {
    echo "powershell.exe is required to run the Miniconda installer." >&2
    exit 1
  }
  win_installer="$(cygpath -w "$installer")"
  win_prefix="$(cygpath -w "$MINICONDA_PREFIX")"
  # /D= must be the last argument and is not quoted inside the installer.
  powershell.exe -NoProfile -Command \
    "Start-Process -FilePath '$win_installer' -ArgumentList '/InstallationType=JustMe','/AddToPath=0','/RegisterPython=0','/S','/D=$win_prefix' -Wait"
}

REQUIREMENTS="$SCRIPT_DIR/requirements-dl.txt"
if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "missing $REQUIREMENTS" >&2
  echo "put requirements-dl.txt next to this script, then run it again" >&2
  exit 1
fi

if ! load_conda; then
  install_miniconda
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already exists, installing packages"
else
  conda create -y -n "$ENV_NAME" python=3.12
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install -U pip
python -m pip install -r "$REQUIREMENTS"
# d2l 1.0.3 pins jupyter==1.0.0, numpy==1.23.5, scipy==1.10.1,
# pandas==2.0.3, and matplotlib==3.7.2. Those pins conflict with
# requirements-dl.txt, and numpy 1.23.5 does not install on Python 3.12.
python -m pip install --no-deps "d2l==1.0.3"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
python -c "import numpy, scipy, torch, mmwave, d2l; print('ok', numpy.__version__, torch.__version__)"
