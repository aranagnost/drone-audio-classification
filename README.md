# Acoustic Drone Detection & Motor-Count Classification

Official PyTorch implementation for the thesis: **"Drone Type Recognition through Audio Data Processing"**.

This project presents a two-stage hierarchical machine learning pipeline that detects Unmanned Aerial Vehicles (UAVs) in noisy environments and classifies them by motor configuration (2, 4, 6, or 8 motors).

## Overview

Acoustic detection provides a passive alternative to traditional radar and visual tracking. This pipeline addresses the problem in two stages:

1. **Stage 1:** Drone vs. No-Drone binary classification using an Audio Spectrogram Transformer (AST). Test macro-F1: 0.9977.
2. **Stage 2:** A soft-cascade ensemble of four models (AST, PaSST, XGBoost, LightGBM) for 4-way motor-count classification (2/4/6/8), resolving the harmonic overlap between structurally similar configurations, notably hexacopters vs. octocopters. Test macro-F1: 0.6787.

## Dataset

62,535 annotated two-second clips (12,327 drone, 50,208 non-drone) across 10 distractor subtypes (wind, power tools, airplanes, and others). Splits are URL-disjoint at the source-recording level to prevent leakage.

* [Link to Kaggle Dataset] *(to be published)*

## Repository Structure

* `data/`: Custom PyTorch Dataset classes and data loaders.
* `models/`: Architecture definitions (AST, PaSST, CNN baselines).
* `training/`: Training routines and metrics.
* `eval/`: Pipeline evaluation and ensemble cascade logic (`eval_end_to_end.py`).
* `artifacts/`: Serialized tree models and pre-computed prediction CSVs.

## Usage

### 1. Interactive Demo (Recommended)
Run the local web interface to test your own audio files. The startup scripts will automatically set up a virtual environment, install dependencies, download the necessary PyTorch weights from GitHub Releases (~1GB), and open the app in your browser.

* **Windows:** Double-click `run_demo.bat`
* **Linux/Mac:** Run `bash run_demo.sh`

### 2. Manual Evaluation (Research)
To manually install dependencies and reproduce the test-set macro-F1 across the ensemble:
```bash
pip install -r requirements.txt
python eval/eval_end_to_end.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
