#!/usr/bin/env python3
"""Download the model weights from the GitHub release into the artifacts/ tree.
Skips files that are already present and look complete.

The demo runs on the files below: the AST stage-1 detector, the AST/PaSST
stage-2 backbones, and the two gradient-boosted tree models (XGBoost 2 s +
LightGBM 10 s) that complete the stage-2 cascade. The .pt checkpoints are large
(~330 MB each); the .joblib tree models are ~1 MB each.
"""
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

RELEASE_URL_BASE = "https://github.com/aranagnost/drone-audio-classification/releases/download/v1.0"

# (release asset name, local destination path, minimum valid size in bytes).
# A Release is a flat list of assets, so the two best_model.joblib files are
# published under distinct asset names and mapped back to their nested dirs here.
ASSETS = [
    ("stage1_ast_v1.pt",                 "artifacts/checkpoints/stage1_ast_v1.pt",    100_000_000),
    ("stage2_ast_v7.pt",                 "artifacts/checkpoints/stage2_ast_v7.pt",    100_000_000),
    ("stage2_passt_v1.pt",               "artifacts/checkpoints/stage2_passt_v1.pt",  100_000_000),
    ("xgb_stage2_best_model.joblib",     "artifacts/xgb_stage2/best_model.joblib",        500_000),
    ("xgb_stage2_10s_best_model.joblib", "artifacts/xgb_stage2_10s/best_model.joblib",    500_000),
]


def download_file(url, dest, min_bytes):
    size_hint = "~330MB" if min_bytes >= 100_000_000 else "~1MB"
    print(f"Downloading {dest.name} ({size_hint}, this may take a while)...")
    # Download to a temp file first, then move into place, so an interrupted
    # download never leaves a truncated file that passes the size check.
    tmp = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
        tmp = Path(tmp_name)
        os.close(fd)
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < min_bytes:
            raise IOError(f"downloaded file is only {tmp.stat().st_size} bytes")
        tmp.replace(dest)
        print(f"Downloaded {dest.name}")
        return True
    except Exception as e:
        print(f"Failed to download {dest.name}. Error: {e}")
        print("   Please check your internet connection or the Release URL.")
        if tmp is not None and tmp.exists():
            tmp.unlink()
        return False


def main():
    failed = []
    for asset_name, dest_str, min_bytes in ASSETS:
        dest = Path(dest_str)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"{dest.name} already exists.")
            continue
        url = f"{RELEASE_URL_BASE}/{asset_name}"
        if not download_file(url, dest, min_bytes):
            failed.append(dest.name)

    if failed:
        print(f"\n{len(failed)} file(s) could not be downloaded: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
