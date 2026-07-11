@echo off
echo ===================================================
echo     Acoustic Drone Detection - Setup ^& Run
echo ===================================================

IF NOT EXIST ".venv" (
    echo [1/5] Creating Python Virtual Environment...
    python -m venv .venv
)

call .venv\Scripts\activate

REM Gate setup on whether the key deps import, not on a sentinel file, so a
REM half-finished previous install (e.g. .venv with no torch) self-heals.
python -c "import torch,torchaudio,transformers,hear21passt,flask,librosa,xgboost,lightgbm" 2>NUL
IF %ERRORLEVEL%==0 GOTO SKIP_SETUP

echo [2/5] Installing PyTorch (CPU, MKL-optimised) from the official index...
python -m pip install --quiet --upgrade pip
pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch torchaudio

echo [3/5] Installing the remaining dependencies...
pip install -r requirements.txt --quiet

python -c "import torch,torchaudio,transformers,hear21passt,flask,librosa,xgboost,lightgbm" 2>NUL
IF NOT %ERRORLEVEL%==0 (
    echo Setup failed: some dependencies still do not import.
    echo Try:  call .venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)
echo done > .setup_done
echo Setup finished successfully!
GOTO RUN_APP

:SKIP_SETUP
echo [2/5] Environment already set up. Skipping installation...

:RUN_APP
echo [4/5] Checking and downloading neural network weights...
python download_weights.py

echo [5/5] Starting the Web Application on http://127.0.0.1:5000 ...
start http://127.0.0.1:5000
python demo.py

pause
