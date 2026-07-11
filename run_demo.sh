#!/usr/bin/env bash
set -e
echo "==================================================="
echo "    Acoustic Drone Detection - Setup & Run"
echo "==================================================="

# Create the virtual environment if it does not exist yet.
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Gate setup on whether the key deps actually import, NOT on a sentinel file.
# A half-finished previous install (e.g. a .venv with no torch) self-heals.
setup_ok() {
    python - <<'PY' 2>/dev/null
import importlib
for m in ("torch", "torchaudio", "transformers", "hear21passt", "flask",
          "librosa", "xgboost", "lightgbm"):
    importlib.import_module(m)
PY
}

if ! setup_ok; then
    echo "[2/5] Installing PyTorch (CPU, MKL-optimised) from the official index..."
    # Explicit CPU wheels: MKL/AVX2-optimised (fast on CPU) and avoid the
    # ~2 GB CUDA download that the default PyPI torch wheel pulls in.
    pip install --quiet --upgrade pip
    pip install --quiet --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio

    echo "[3/5] Installing the remaining dependencies..."
    pip install --quiet -r requirements.txt

    if ! setup_ok; then
        echo "[ERROR] Setup failed: some dependencies still do not import."
        echo "   Try:  source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    touch .setup_done
    echo "Setup finished successfully."
else
    echo "[2/5] Environment already set up. Skipping installation..."
fi

echo "[4/5] Checking and downloading neural network weights..."
python download_weights.py

echo "[5/5] Starting the web application on http://127.0.0.1:5000 ..."
if   which xdg-open > /dev/null 2>&1; then (sleep 2; xdg-open http://127.0.0.1:5000) &
elif which open     > /dev/null 2>&1; then (sleep 2; open     http://127.0.0.1:5000) &
fi

python demo.py
