#!/usr/bin/env python3
"""
inference_app.py

Local Flask web app for testing trained drone audio classification models.
Upload audio, pick a model, see per-class probabilities for each segment.

Usage:
    python inference_app.py              # default port 5001
    python inference_app.py --port 8080  # custom port

Then open http://localhost:5001 in your browser.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from flask import Flask, Response, jsonify, request, send_file
from pydub import AudioSegment
from pydub.effects import normalize

# ──────────────────── paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"

# ──────────────────── audio config defaults (overridden by checkpoint cfg) ───
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 64
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 160
DEFAULT_WIN_LENGTH = 400
DEFAULT_F_MIN = 20

SEG_LEN_MS = 2000
STEP_MS = 1500

app = Flask(__name__)

# ──────────────────── model registry ─────────────────────────────────────────
# Populated at startup by scanning models/*/model.json
MODEL_REGISTRY: dict[str, dict] = {}

# Temp dir for serving segment audio; cleared on each new prediction
_temp_dir: Path | None = None


def discover_models():
    """Scan models/*/model.json and build the registry."""
    for model_json_path in sorted(MODELS_DIR.glob("*/model.json")):
        model_id = model_json_path.parent.name
        try:
            with open(model_json_path) as f:
                spec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Skipping {model_json_path}: {e}")
            continue

        # Validate checkpoints exist
        stages = spec.get("stages", [])
        valid = True
        for stage in stages:
            ckpt_path = CHECKPOINTS_DIR / stage["checkpoint"]
            if not ckpt_path.exists():
                print(f"[WARN] Skipping {model_id}: checkpoint not found: {ckpt_path}")
                valid = False
                break
        if not valid:
            continue

        # Load model class from model.py in the same directory
        model_py = model_json_path.parent / "model.py"
        if not model_py.exists():
            print(f"[WARN] Skipping {model_id}: model.py not found")
            continue

        module_name = f"models.{model_id}.model"
        module_spec = importlib.util.spec_from_file_location(module_name, str(model_py))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        model_class_name = spec.get("model_class", "SmallCNN")
        model_class = getattr(module, model_class_name, None)
        if model_class is None:
            print(f"[WARN] Skipping {model_id}: class {model_class_name} not found in model.py")
            continue

        # Load checkpoints and build stage info
        loaded_stages = []
        for stage in stages:
            ckpt_path = CHECKPOINTS_DIR / stage["checkpoint"]
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Determine num_classes from checkpoint weights
            num_classes = ckpt["model_state"]["classifier.weight"].shape[0]

            # Build model instance
            model = model_class(num_classes=num_classes)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            # Extract audio config from checkpoint
            cfg = ckpt.get("cfg", {})

            loaded_stages.append({
                "name": stage["name"],
                "labels": stage["labels"],
                "run_if": stage.get("run_if"),
                "model": model,
                "cfg": cfg,
            })

        MODEL_REGISTRY[model_id] = {
            "id": model_id,
            "name": spec["name"],
            "stages": loaded_stages,
        }
        print(f"[OK] Registered model: {model_id} ({spec['name']}) with {len(loaded_stages)} stages")


def wav_to_logmel(wav_path: Path, cfg: dict) -> torch.Tensor:
    """Load a WAV segment and return a (1, 1, n_mels, time) log-mel tensor."""
    sample_rate = cfg.get("sample_rate", DEFAULT_SAMPLE_RATE)
    n_mels = cfg.get("n_mels", DEFAULT_N_MELS)
    n_fft = cfg.get("n_fft", DEFAULT_N_FFT)
    hop_length = cfg.get("hop_length", DEFAULT_HOP_LENGTH)
    win_length = cfg.get("win_length", DEFAULT_WIN_LENGTH)
    f_min = cfg.get("f_min", DEFAULT_F_MIN)
    clip_samples = int(sample_rate * (SEG_LEN_MS / 1000))

    waveform, sr = torchaudio.load(str(wav_path))
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] < clip_samples:
        waveform = F.pad(waveform, (0, clip_samples - waveform.shape[1]))
    else:
        waveform = waveform[:, :clip_samples]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=f_min,
        n_mels=n_mels,
    )
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-9)
    return log_mel.unsqueeze(0)


def segment_audio(audio_path: Path, out_dir: Path) -> list[dict]:
    """Segment audio into overlapping 2s clips. Returns list of {path, start_ms, end_ms}."""
    audio = AudioSegment.from_file(str(audio_path))
    audio = normalize(audio)
    audio = audio.set_channels(1).set_frame_rate(DEFAULT_SAMPLE_RATE)

    duration_ms = len(audio)
    segments = []
    idx = 0
    start = 0

    while start + SEG_LEN_MS <= duration_ms:
        seg = audio[start : start + SEG_LEN_MS]
        seg_path = out_dir / f"seg_{idx:04d}.wav"
        seg.export(str(seg_path), format="wav")
        segments.append({
            "index": idx,
            "start_ms": start,
            "end_ms": start + SEG_LEN_MS,
            "path": seg_path,
        })
        idx += 1
        start += STEP_MS

    # Handle trailing audio: if there's remaining audio after the last full segment
    if start < duration_ms and duration_ms - start >= SEG_LEN_MS // 2:
        seg = audio[duration_ms - SEG_LEN_MS : duration_ms]
        if len(seg) == SEG_LEN_MS:
            seg_path = out_dir / f"seg_{idx:04d}.wav"
            seg.export(str(seg_path), format="wav")
            segments.append({
                "index": idx,
                "start_ms": duration_ms - SEG_LEN_MS,
                "end_ms": duration_ms,
                "path": seg_path,
            })

    return segments, duration_ms


def run_pipeline(model_entry: dict, segments: list[dict]) -> list[dict]:
    """Run the multi-stage pipeline on all segments. Returns per-segment results."""
    stages = model_entry["stages"]
    results = []

    for seg in segments:
        seg_results = []
        for stage_idx, stage in enumerate(stages):
            # Check run_if condition
            run_if = stage.get("run_if")
            gate_prob = 1.0
            if run_if is not None:
                prev_stage = run_if["stage"]
                required_class = run_if["class"]
                threshold = run_if.get("threshold", 0.5)
                # Find the previous stage's result
                if prev_stage < len(seg_results):
                    prev_result = seg_results[prev_stage]
                    prev_labels = stages[prev_stage]["labels"]
                    req_idx = prev_labels.index(required_class)
                    gate_prob = prev_result["probs"][req_idx]
                    if gate_prob < threshold:
                        continue  # Skip this stage
                else:
                    continue  # Previous stage didn't run

            # Run inference
            logmel = wav_to_logmel(seg["path"], stage["cfg"])
            with torch.no_grad():
                logits = stage["model"](logmel)
                probs = F.softmax(logits, dim=1).squeeze(0).tolist()

            result_entry = {
                "stage": stage_idx,
                "probs": [round(p, 4) for p in probs],
            }
            # Attach gate probability for conditional stages
            if run_if is not None:
                result_entry["gate"] = round(gate_prob, 4)

            seg_results.append(result_entry)

        results.append({
            "index": seg["index"],
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "results": seg_results,
        })

    return results


def compute_aggregate(model_entry: dict, segment_results: list[dict]) -> list[dict]:
    """Compute (weighted) mean probabilities per stage across segments."""
    stages = model_entry["stages"]
    aggregate = []

    for stage_idx, stage in enumerate(stages):
        # Collect all probs (and gate weights) for this stage
        all_probs = []
        all_weights = []
        for seg in segment_results:
            for r in seg["results"]:
                if r["stage"] == stage_idx:
                    all_probs.append(r["probs"])
                    # Use gate probability as weight for conditional stages
                    all_weights.append(r.get("gate", 1.0))

        if not all_probs:
            aggregate.append({
                "stage": stage_idx,
                "mean_probs": [0.0] * len(stage["labels"]),
                "count": 0,
                "note": "no segments ran this stage",
            })
            continue

        n_classes = len(all_probs[0])
        total_weight = sum(all_weights)

        if stage.get("run_if") is not None and total_weight > 0:
            # Weighted average: segments with higher P(drone) contribute more
            mean = [
                round(sum(all_probs[i][c] * all_weights[i] for i in range(len(all_probs))) / total_weight, 4)
                for c in range(n_classes)
            ]
            run_if = stage["run_if"]
            threshold = run_if.get("threshold", 0.5)
            entry = {
                "stage": stage_idx,
                "mean_probs": mean,
                "count": len(all_probs),
                "note": f"weighted by P({run_if['class']}), threshold {threshold}",
            }
        else:
            # Simple average for unconditional stages
            n = len(all_probs)
            mean = [round(sum(p[c] for p in all_probs) / n, 4) for c in range(n_classes)]
            entry = {"stage": stage_idx, "mean_probs": mean, "count": n}

        aggregate.append(entry)

    return aggregate


# ──────────────────── API endpoints ──────────────────────────────────────────
@app.route("/api/models")
def api_models():
    models = []
    for mid, m in MODEL_REGISTRY.items():
        models.append({
            "id": mid,
            "name": m["name"],
            "stages": [{"name": s["name"], "labels": s["labels"]} for s in m["stages"]],
        })
    return jsonify(models)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global _temp_dir

    model_id = request.form.get("model")
    if not model_id or model_id not in MODEL_REGISTRY:
        return jsonify({"error": "Invalid model"}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    model_entry = MODEL_REGISTRY[model_id]

    # Clean up previous temp dir
    if _temp_dir and Path(_temp_dir).exists():
        shutil.rmtree(_temp_dir, ignore_errors=True)

    # Save upload to temp
    tmp_base = Path(tempfile.mkdtemp(prefix="drone_inference_"))
    upload_path = tmp_base / "upload_raw"
    file.save(str(upload_path))

    # Convert to WAV if needed (via ffmpeg)
    wav_path = tmp_base / "upload.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(upload_path), "-ar", str(DEFAULT_SAMPLE_RATE),
             "-ac", "1", str(wav_path)],
            capture_output=True, check=True, timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try loading directly with pydub
        try:
            audio = AudioSegment.from_file(str(upload_path))
            audio.export(str(wav_path), format="wav")
        except Exception as e:
            shutil.rmtree(tmp_base, ignore_errors=True)
            return jsonify({"error": f"Could not read audio file: {e}"}), 400

    # Segment
    seg_dir = tmp_base / "segments"
    seg_dir.mkdir()
    try:
        segments, duration_ms = segment_audio(wav_path, seg_dir)
    except Exception as e:
        shutil.rmtree(tmp_base, ignore_errors=True)
        return jsonify({"error": f"Segmentation failed: {e}"}), 400

    if not segments:
        shutil.rmtree(tmp_base, ignore_errors=True)
        return jsonify({"error": "Audio too short (need at least 2 seconds)"}), 400

    # Run pipeline
    segment_results = run_pipeline(model_entry, segments)

    # Compute aggregate
    aggregate = compute_aggregate(model_entry, segment_results)

    # Generate token for segment audio serving
    token = uuid.uuid4().hex[:12]
    serve_dir = tmp_base / "serve" / token
    serve_dir.mkdir(parents=True)
    for seg in segments:
        shutil.copy2(str(seg["path"]), str(serve_dir / f"{seg['index']}.wav"))

    _temp_dir = tmp_base

    return jsonify({
        "model_id": model_id,
        "model_name": model_entry["name"],
        "token": token,
        "duration_ms": duration_ms,
        "num_segments": len(segments),
        "stages": [{"name": s["name"], "labels": s["labels"]} for s in model_entry["stages"]],
        "segments": segment_results,
        "aggregate": aggregate,
    })


@app.route("/api/segment_audio/<token>/<int:idx>")
def api_segment_audio(token, idx):
    if not _temp_dir:
        return Response("No prediction data", status=404)
    wav = Path(_temp_dir) / "serve" / token / f"{idx}.wav"
    if not wav.exists():
        return Response("Segment not found", status=404)
    return send_file(str(wav), mimetype="audio/wav")


# ──────────────────── frontend (single-page inline HTML) ─────────────────────
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drone Audio Classifier</title>
<style>
:root {
  --bg: #1a1a2e;
  --surface: #16213e;
  --surface2: #0f3460;
  --accent: #e94560;
  --accent2: #533483;
  --text: #eee;
  --text-dim: #999;
  --green: #4caf50;
  --red: #f44336;
  --orange: #ff9800;
  --blue: #2196f3;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}

/* ── header ── */
.header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 14px 24px;
  background: var(--surface);
  border-bottom: 1px solid var(--surface2);
}
.header h1 {
  font-size: 18px;
  color: var(--accent);
  margin-right: auto;
}

/* ── controls row ── */
.controls-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 24px;
  background: var(--surface);
  border-bottom: 1px solid var(--surface2);
  flex-wrap: wrap;
}
.controls-row label {
  font-size: 12px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.controls-row select,
.controls-row input[type="file"] {
  background: var(--surface2);
  color: var(--text);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 14px;
}
.controls-row select { min-width: 200px; }
.file-input-wrap {
  position: relative;
  display: inline-block;
}
.file-input-wrap input[type="file"] {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}
.file-input-label {
  display: inline-block;
  padding: 8px 16px;
  background: var(--surface2);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px;
  font-size: 14px;
  color: var(--text);
  cursor: pointer;
  transition: border-color 0.15s;
}
.file-input-label:hover { border-color: var(--accent); }
.file-name {
  font-size: 13px;
  color: var(--text-dim);
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.btn {
  padding: 8px 20px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
}
.btn:hover { opacity: 0.85; }
.btn:disabled { opacity: 0.35; cursor: not-allowed; }
.btn-primary {
  background: var(--accent);
  color: #fff;
}
.btn-secondary {
  background: var(--surface2);
  color: var(--text);
  border: 1px solid rgba(255,255,255,0.15);
}

/* ── spinner overlay ── */
.spinner-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  z-index: 100;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 16px;
}
.spinner-overlay.active { display: flex; }
.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(255,255,255,0.2);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-text { font-size: 14px; color: var(--text-dim); }

/* ── results container ── */
.results {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}
.results.hidden { display: none; }
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 80px 24px;
  color: var(--text-dim);
  font-size: 16px;
}

/* ── aggregate section ── */
.aggregate-section {
  margin-bottom: 32px;
}
.aggregate-section h2 {
  font-size: 16px;
  margin-bottom: 16px;
  color: var(--text);
}
.stage-card {
  background: var(--surface);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 12px;
}
.stage-card h3 {
  font-size: 14px;
  color: var(--text-dim);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.stage-card h3 .note {
  font-size: 11px;
  font-weight: 400;
  color: var(--text-dim);
  font-style: italic;
}
.stage-card .count-badge {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 10px;
  background: var(--surface2);
  color: var(--text-dim);
}
.prob-bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.prob-bar-label {
  font-size: 13px;
  min-width: 90px;
  text-align: right;
  color: var(--text);
}
.prob-bar-track {
  flex: 1;
  height: 22px;
  background: var(--surface2);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}
.prob-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
  min-width: 2px;
}
.prob-bar-value {
  font-size: 12px;
  min-width: 50px;
  color: var(--text-dim);
  font-family: monospace;
}

/* ── segment table ── */
.segment-section h2 {
  font-size: 16px;
  margin-bottom: 12px;
}
.segment-table-wrap {
  overflow-x: auto;
}
.segment-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.segment-table th {
  text-align: left;
  padding: 8px 10px;
  background: var(--surface);
  color: var(--text-dim);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid var(--surface2);
  position: sticky;
  top: 0;
  z-index: 1;
}
.segment-table td {
  padding: 6px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  vertical-align: middle;
}
.segment-table tr:hover td {
  background: rgba(255,255,255,0.02);
}
.seg-time {
  font-family: monospace;
  font-size: 12px;
  color: var(--text-dim);
  white-space: nowrap;
}
.play-btn {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.2);
  background: var(--surface2);
  color: var(--text);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  transition: all 0.15s;
  padding: 0;
}
.play-btn:hover { border-color: var(--accent); color: var(--accent); }
.play-btn.playing { background: var(--accent); border-color: var(--accent); color: #fff; }
.mini-bar-wrap {
  display: flex;
  gap: 2px;
  align-items: flex-end;
  height: 24px;
}
.mini-bar {
  width: 18px;
  border-radius: 2px 2px 0 0;
  position: relative;
  cursor: default;
}
.mini-bar-tooltip {
  display: none;
  position: absolute;
  bottom: calc(100% + 4px);
  left: 50%;
  transform: translateX(-50%);
  background: #222;
  color: #fff;
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 4px;
  white-space: nowrap;
  z-index: 10;
  pointer-events: none;
}
.mini-bar:hover .mini-bar-tooltip { display: block; }
.skipped-label {
  color: var(--text-dim);
  font-size: 12px;
  font-style: italic;
}
.gate-badge {
  font-size: 10px;
  color: var(--text-dim);
  margin-left: 4px;
  opacity: 0.7;
  white-space: nowrap;
}

/* ── colors for probability bars ── */
.color-0 { background: #2196f3; }
.color-1 { background: #e94560; }
.color-2 { background: #4caf50; }
.color-3 { background: #ff9800; }
.color-4 { background: #9c27b0; }
.color-5 { background: #00bcd4; }
.color-6 { background: #795548; }
.color-7 { background: #607d8b; }

/* ── toast ── */
.toast {
  position: fixed;
  bottom: 24px;
  right: 24px;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  color: #fff;
  z-index: 1000;
  opacity: 0;
  transition: opacity 0.3s;
  pointer-events: none;
}
.toast.show { opacity: 1; }
.toast.error { background: var(--red); }
</style>
</head>
<body>

<div class="header">
  <h1>Drone Audio Classifier</h1>
</div>

<div class="controls-row">
  <div>
    <label>Audio file</label><br>
    <div class="file-input-wrap">
      <span class="file-input-label" id="file-label">Choose file...</span>
      <input type="file" id="file-input" accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a,.webm">
    </div>
    <span class="file-name" id="file-name"></span>
  </div>
  <div>
    <label>Model</label><br>
    <select id="model-select"></select>
  </div>
  <button class="btn btn-primary" id="btn-classify" disabled>Classify</button>
</div>

<div class="spinner-overlay" id="spinner">
  <div class="spinner"></div>
  <div class="spinner-text">Running inference...</div>
</div>

<div class="empty-state" id="empty-state">Upload an audio file and select a model to begin</div>

<div class="results hidden" id="results">
  <div class="aggregate-section" id="aggregate-section"></div>
  <div class="segment-section">
    <h2 id="segment-heading">Per-Segment Results</h2>
    <div class="segment-table-wrap">
      <table class="segment-table" id="segment-table">
        <thead id="segment-thead"></thead>
        <tbody id="segment-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<audio id="hidden-audio" style="display:none"></audio>

<script>
(function() {
  // ── state ──
  let models = [];
  let currentResult = null;
  let playingIdx = -1;

  const fileInput = document.getElementById('file-input');
  const fileLabel = document.getElementById('file-label');
  const fileName = document.getElementById('file-name');
  const modelSelect = document.getElementById('model-select');
  const btnClassify = document.getElementById('btn-classify');
  const spinner = document.getElementById('spinner');
  const emptyState = document.getElementById('empty-state');
  const resultsEl = document.getElementById('results');
  const aggregateSection = document.getElementById('aggregate-section');
  const segmentThead = document.getElementById('segment-thead');
  const segmentTbody = document.getElementById('segment-tbody');
  const segmentHeading = document.getElementById('segment-heading');
  const hiddenAudio = document.getElementById('hidden-audio');

  const STAGE_COLORS = ['#2196f3', '#e94560', '#4caf50', '#ff9800', '#9c27b0', '#00bcd4', '#795548', '#607d8b'];

  function toast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = 'toast show error';
    setTimeout(() => el.className = 'toast', 3000);
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function fmtTime(ms) {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const ss = s % 60;
    return m + ':' + String(ss).padStart(2, '0');
  }

  function pct(v) { return (v * 100).toFixed(1) + '%'; }

  // ── load models ──
  async function loadModels() {
    try {
      const resp = await fetch('/api/models');
      models = await resp.json();
    } catch (e) {
      toast('Failed to load models');
      return;
    }
    modelSelect.innerHTML = '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name;
      modelSelect.appendChild(opt);
    });
    updateClassifyBtn();
  }

  // ── file input ──
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      fileName.textContent = fileInput.files[0].name;
      fileLabel.textContent = 'Change file...';
    } else {
      fileName.textContent = '';
      fileLabel.textContent = 'Choose file...';
    }
    updateClassifyBtn();
  });

  function updateClassifyBtn() {
    btnClassify.disabled = !(fileInput.files.length > 0 && models.length > 0);
  }

  // ── classify ──
  btnClassify.addEventListener('click', doClassify);

  async function doClassify() {
    if (!fileInput.files.length || !models.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', modelSelect.value);

    spinner.classList.add('active');
    btnClassify.disabled = true;

    try {
      const resp = await fetch('/api/predict', { method: 'POST', body: formData });
      const data = await resp.json();
      if (data.error) {
        toast(data.error);
        return;
      }
      currentResult = data;
      renderResults(data);
    } catch (e) {
      toast('Inference failed: ' + e.message);
    } finally {
      spinner.classList.remove('active');
      updateClassifyBtn();
    }
  }

  // ── render results ──
  function renderResults(data) {
    emptyState.style.display = 'none';
    resultsEl.classList.remove('hidden');

    renderAggregate(data);
    renderSegments(data);
  }

  function barColor(stageIdx, classIdx) {
    // Use stage-specific color palette
    const palettes = [
      ['#2196f3', '#e94560'],                                    // stage 0: blue/red
      ['#4caf50', '#ff9800', '#9c27b0', '#00bcd4', '#795548'],  // stage 1+
    ];
    const pal = stageIdx === 0 ? palettes[0] : palettes[1];
    return pal[classIdx % pal.length];
  }

  function renderAggregate(data) {
    let html = '<h2>Aggregate Probabilities</h2>';
    data.aggregate.forEach((agg, si) => {
      const stage = data.stages[agg.stage];
      const note = agg.note ? ' <span class="note">(' + esc(agg.note) + ')</span>' : '';
      const countBadge = '<span class="count-badge">' + agg.count + ' segments</span>';
      html += '<div class="stage-card">';
      html += '<h3>' + esc(stage.name) + note + ' ' + countBadge + '</h3>';

      if (agg.count === 0) {
        html += '<div style="color:var(--text-dim);font-style:italic;font-size:13px;">No segments ran this stage</div>';
      } else {
        stage.labels.forEach((label, ci) => {
          const prob = agg.mean_probs[ci];
          const color = barColor(agg.stage, ci);
          html += '<div class="prob-bar-row">';
          html += '<span class="prob-bar-label">' + esc(label) + '</span>';
          html += '<div class="prob-bar-track"><div class="prob-bar-fill" style="width:' + (prob * 100) + '%;background:' + color + '"></div></div>';
          html += '<span class="prob-bar-value">' + pct(prob) + '</span>';
          html += '</div>';
        });
      }
      html += '</div>';
    });
    aggregateSection.innerHTML = html;
  }

  function renderSegments(data) {
    segmentHeading.textContent = 'Per-Segment Results (' + data.num_segments + ' segments, ' + fmtTime(data.duration_ms) + ' total)';

    // Build header
    let thHtml = '<tr><th>#</th><th>Time</th><th></th>';
    data.stages.forEach(s => {
      thHtml += '<th>' + esc(s.name) + '</th>';
    });
    thHtml += '</tr>';
    segmentThead.innerHTML = thHtml;

    // Build rows
    let tbHtml = '';
    data.segments.forEach(seg => {
      tbHtml += '<tr>';
      tbHtml += '<td>' + seg.index + '</td>';
      tbHtml += '<td class="seg-time">' + fmtTime(seg.start_ms) + ' - ' + fmtTime(seg.end_ms) + '</td>';
      tbHtml += '<td><button class="play-btn" data-idx="' + seg.index + '">&#9654;</button></td>';

      // For each stage, render mini bars or "skipped"
      data.stages.forEach((stage, si) => {
        const result = seg.results.find(r => r.stage === si);
        if (!result) {
          tbHtml += '<td><span class="skipped-label">&mdash;</span></td>';
          return;
        }
        tbHtml += '<td><div class="mini-bar-wrap">';
        result.probs.forEach((p, ci) => {
          const color = barColor(si, ci);
          const h = Math.max(2, Math.round(p * 24));
          tbHtml += '<div class="mini-bar" style="height:' + h + 'px;background:' + color + '">';
          tbHtml += '<div class="mini-bar-tooltip">' + esc(stage.labels[ci]) + ': ' + pct(p) + '</div>';
          tbHtml += '</div>';
        });
        tbHtml += '</div>';
        // Show gate weight for conditional stages
        if (result.gate !== undefined && result.gate < 1.0) {
          tbHtml += '<span class="gate-badge">w=' + result.gate.toFixed(2) + '</span>';
        }
        tbHtml += '</td>';
      });
      tbHtml += '</tr>';
    });
    segmentTbody.innerHTML = tbHtml;

    // Bind play buttons
    segmentTbody.querySelectorAll('.play-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx);
        playSegment(idx, btn);
      });
    });
  }

  // ── audio playback ──
  function playSegment(idx, btn) {
    // Stop current if playing same
    if (playingIdx === idx && !hiddenAudio.paused) {
      hiddenAudio.pause();
      btn.classList.remove('playing');
      btn.innerHTML = '&#9654;';
      playingIdx = -1;
      return;
    }

    // Reset all play buttons
    segmentTbody.querySelectorAll('.play-btn').forEach(b => {
      b.classList.remove('playing');
      b.innerHTML = '&#9654;';
    });

    hiddenAudio.src = '/api/segment_audio/' + currentResult.token + '/' + idx;
    hiddenAudio.play().catch(() => {});
    btn.classList.add('playing');
    btn.innerHTML = '&#9646;&#9646;';
    playingIdx = idx;

    hiddenAudio.onended = () => {
      btn.classList.remove('playing');
      btn.innerHTML = '&#9654;';
      playingIdx = -1;
    };
  }

  // ── init ──
  loadModels();
})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# ──────────────────── main ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone audio inference web app")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Discovering models in {MODELS_DIR}...")
    discover_models()

    if not MODEL_REGISTRY:
        print("[WARN] No models found! Check models/*/model.json and artifacts/checkpoints/")

    print(f"\nStarting inference app at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
