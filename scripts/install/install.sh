#!/bin/sh
set -eu

if command -v python3 >/dev/null 2>&1; then
  ARC_INSTALL_PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  ARC_INSTALL_PYTHON=python
else
  if command -v brew >/dev/null 2>&1; then
    printf "Python 3 is required. Install it with Homebrew now? [Y/n] "
    read -r reply
    case "${reply:-y}" in
      n|N) exit 1 ;;
      *) brew install python ;;
    esac
    ARC_INSTALL_PYTHON=python3
  elif command -v apt-get >/dev/null 2>&1; then
    printf "Python 3 is required. Install it with apt now? [Y/n] "
    read -r reply
    case "${reply:-y}" in
      n|N) exit 1 ;;
      *)
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv
        ;;
    esac
    ARC_INSTALL_PYTHON=python3
  else
    echo "Arc installer requires Python 3.10+ or a supported package manager bootstrap." >&2
    exit 1
  fi
fi

ARC_INSTALL_PACKAGE_URL="${ARC_INSTALL_WHEEL_URL:-https://github.com/mohit17mor/Arc/archive/refs/heads/main.zip}"
ARC_TMPDIR="$(mktemp -d)"
trap 'rm -rf "$ARC_TMPDIR"' EXIT

"$ARC_INSTALL_PYTHON" -m venv "$ARC_TMPDIR/bootstrap-venv"
ARC_BOOTSTRAP_PYTHON="$ARC_TMPDIR/bootstrap-venv/bin/python"
ARC_BOOTSTRAP_PIP="$ARC_TMPDIR/bootstrap-venv/bin/pip"

"$ARC_BOOTSTRAP_PIP" install --upgrade pip >/dev/null
"$ARC_BOOTSTRAP_PIP" install "$ARC_INSTALL_PACKAGE_URL"
"$ARC_BOOTSTRAP_PYTHON" -m arc.install.bootstrap --wheel-url "$ARC_INSTALL_PACKAGE_URL"

ARC_BIN_DIR="${HOME}/.arc/bin"
if ! printf "%s" "${PATH}" | grep -q "${ARC_BIN_DIR}"; then
  ARC_PROFILE="${HOME}/.profile"
  if [ -n "${SHELL:-}" ] && printf "%s" "${SHELL}" | grep -q "zsh"; then
    ARC_PROFILE="${HOME}/.zprofile"
  fi
  touch "$ARC_PROFILE"
  if ! grep -q 'export PATH="$HOME/.arc/bin:$PATH"' "$ARC_PROFILE" 2>/dev/null; then
    printf '\nexport PATH="$HOME/.arc/bin:$PATH"\n' >> "$ARC_PROFILE"
  fi
  echo "Added ~/.arc/bin to PATH via ${ARC_PROFILE}. Open a new terminal before running arc."
else
  echo "Arc installer finished. Run: arc init"
fi

# Optional: TTS setup hint
echo ""
echo "Optional — Voice output (TTS):"
echo "  pip install arc-agent[tts]"
echo "  Then download Kokoro model files (~300MB, one-time):"
echo "    mkdir -p ~/.arc/models/kokoro"
echo "    cd ~/.arc/models/kokoro"
echo "    curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
echo "    curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
echo "  Or skip this — Arc falls back to system speech (say/espeak-ng)."
