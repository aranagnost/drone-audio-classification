#!/usr/bin/env python3
"""
Local inference demo for acoustic drone detection.
Serves a small Flask web UI to test the trained PyTorch models.

Usage:
    python demo.py
"""

import os

# Keep numba's JIT enabled here. The batch feature scripts disable it to avoid a
# librosa segfault inside forked workers, but the demo is single-process and
# disabling it breaks librosa's guvectorize kernels. Force it on before any numba
# import so the feature modules' own setdefault becomes a no-op.
os.environ["NUMBA_DISABLE_JIT"] = "0"

import uuid
import tempfile
import shutil
import warnings
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
from flask import Flask, Response, jsonify, request, send_file
from pydub import AudioSegment
from pydub.effects import normalize
from transformers import ASTFeatureExtractor

warnings.filterwarnings("ignore")

# Silence chatty third-party logging (transformers, hear21passt, Flask) so the
# console only shows our own status lines.
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@contextmanager
def _quiet():
    """Suppress stdout/stderr from libraries that print directly (e.g. hear21passt)."""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "checkpoints")
XGB_DIR = os.path.join(PROJECT_ROOT, "artifacts", "xgb_stage2")
LGBM_DIR = os.path.join(PROJECT_ROOT, "artifacts", "xgb_stage2_10s")

# --- Constants ---
SEG_LEN_MS = 2000
STEP_MS = 1500

# Accepted upload length: 6 s minimum for enough context and segments to confirm
# a drone, 30 s maximum to keep CPU processing time reasonable.
MIN_INPUT_SECONDS = 6
MAX_INPUT_SECONDS = 30
S2_LABELS = ["2_motors", "4_motors", "6_motors", "8_motors"]
STAGE_B_LABELS = ["4_motors", "6_motors", "8_motors"]

# A clip is flagged as a drone only if at least PRESENCE_MIN_CONSEC consecutive
# segments exceed PRESENCE_THRESHOLD. Requiring a sustained run rejects isolated
# false positives (e.g. lawnmowers) while still catching a partial-clip drone.
PRESENCE_THRESHOLD = 0.5
PRESENCE_MIN_CONSEC = 3   # about 4.5 s of sustained detection

# Models required by each pipeline, used to reject a request when a checkpoint
# is missing from the release.
REQUIRED_MODELS = {
    "ast_v7":          ["s2_ast_v7"],
    "passt":           ["s2_passt"],
    "end_to_end_ast":  ["s1_ast", "s2_ast_v7", "s2_passt"],
}

# Calibration temperatures for probability blending.
T_AST = 1.117
T_PASST = 0.970
T_XGB = 1.113
T_LGBM = 1.450

app = Flask(__name__)
_temp_dir = None

# --- Global Models Registry ---
MODELS = {}
EXTRACTOR = None
FEAT_NAMES_2S = None
FEAT_NAMES_10S = None
_tree_warned = False


def _load_feature_names(path):
    """Read a feature_names.txt (one column name per line) or return None."""
    try:
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return None

def load_models():
    """Load PyTorch models into RAM."""
    global EXTRACTOR
    print("Loading feature extractor...")
    with _quiet():
        EXTRACTOR = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    def clean_state_dict(state_dict):
        """Removes 'module.' or other common prefixes from state_dict keys if they exist."""
        clean_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v
        return clean_dict

    def load_checkpoint(Cls, filename, num_classes, device):
        """Build the model with the head the checkpoint used and load its weights,
        asserting a fully-matched load. Auto-detects the 2-layer MLP head from the
        state dict. Returns None if the file is absent, so a partial release
        disables only the pipelines that need the missing weights.
        """
        path = os.path.join(CHECKPOINTS_DIR, filename)
        if not os.path.exists(path):
            print(f"[skip] {filename} not found in {CHECKPOINTS_DIR}, "
                  f"pipelines needing it will be disabled.")
            return None
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        sd = clean_state_dict(sd)
        mlp_head = any("classifier.fc1" in k or "classifier.fc2" in k for k in sd)

        # Building a PaSST model dumps the whole module tree + banner via print().
        with _quiet():
            model = Cls(num_classes=num_classes, dropout=0.0, mlp_head=mlp_head).to(device)
        res = model.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(
                f"{filename}: checkpoint did not match model "
                f"({len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys). "
                f"This usually means the installed transformers version renamed the AST "
                f"backbone. Pin transformers<5 (see requirements.txt). "
                f"First missing: {res.missing_keys[:3]}  First unexpected: {res.unexpected_keys[:3]}"
            )
        model.eval()
        return model

    print("Loading PyTorch checkpoints...")
    try:
        from models.ast_models import ASTClassifier
        from models.passt_models import PaSSTClassifier

        device = "cuda" if torch.cuda.is_available() else "cpu"

        specs = [
            ("s1_ast",    ASTClassifier,   "stage1_ast_v1.pt",   2),
            ("s2_ast_v7", ASTClassifier,   "stage2_ast_v7.pt",   4),
            ("s2_passt",  PaSSTClassifier, "stage2_passt_v1.pt", 4),
        ]
        for key, Cls, filename, num_classes in specs:
            model = load_checkpoint(Cls, filename, num_classes, device)
            if model is not None:
                MODELS[key] = model

        # hear21passt prints a wall of debug on its first forward pass (gated by
        # a module-level `first_RUN` flag). Flip it off so inference stays quiet.
        try:
            import hear21passt.models.passt as _passt_mod
            _passt_mod.first_RUN = False
        except Exception:
            pass

        loaded = [k for k, _, _, _ in specs if k in MODELS]
        print(f"Neural networks loaded: {loaded}")
    except Exception as e:
        print(f"Error loading PyTorch models: {e}")

    # Try loading Tree models gracefully
    global FEAT_NAMES_2S, FEAT_NAMES_10S
    try:
        import joblib
        MODELS["xgb_2s"] = joblib.load(os.path.join(XGB_DIR, "best_model.joblib"))
        MODELS["lgbm_10s"] = joblib.load(os.path.join(LGBM_DIR, "best_model.joblib"))
        FEAT_NAMES_2S = _load_feature_names(os.path.join(XGB_DIR, "feature_names.txt"))
        FEAT_NAMES_10S = _load_feature_names(os.path.join(LGBM_DIR, "feature_names.txt"))
        print("Tree models loaded successfully.")
    except Exception as e:
        print(f"Tree models skipped (missing dependencies or files): {e}")

# --- Feature Preparation ---
def extract_10s_window(audio_path, center_ms):
    """Extract a 10s window around the center, loop if too short to mimic dataset behavior."""
    audio = AudioSegment.from_file(audio_path)
    start_ms = max(0, center_ms - 5000)
    end_ms = min(len(audio), center_ms + 5000)
    slice_audio = audio[start_ms:end_ms]
    
    # Loop to fill 10s if necessary
    while len(slice_audio) > 0 and len(slice_audio) < 10000:
        slice_audio += slice_audio
    
    tmp_wav = tempfile.mktemp(suffix=".wav")
    slice_audio[:10000].export(tmp_wav, format="wav")
    return tmp_wav

def _load_waveform(wav_path):
    """Load an audio file to a (channels, samples) float32 tensor and sample rate.

    Uses soundfile because recent torchaudio delegates decoding to the optional
    torchcodec package; torchaudio is kept only for resampling.
    """
    import soundfile as sf
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # (samples, channels)
    wav = torch.from_numpy(data.T).contiguous()                    # (channels, samples)
    return wav, sr

def prep_ast_input(wav_path):
    """16kHz Log-Mel for AST."""
    wav, sr = _load_waveform(wav_path)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    inputs = EXTRACTOR(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    return inputs.input_values

def prep_passt_input(wav_path):
    """32kHz raw waveform for PaSST."""
    wav, sr = _load_waveform(wav_path)
    if sr != 32000: wav = torchaudio.functional.resample(wav, sr, 32000)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    # Pad if strictly needed, but 10s window handles it
    return wav

def apply_temperature(probs, T):
    probs = np.clip(probs, 1e-12, 1.0)
    log_probs = np.log(probs) / T
    log_probs -= log_probs.max()
    e = np.exp(log_probs)
    return e / e.sum()

# --- Inference Pipelines ---
def run_segment(wav_file, start_ms, end_ms, mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    center_ms = start_ms + 1000
    
    # Extract 2s and 10s wav files for this segment
    tmp_2s = tempfile.mktemp(suffix=".wav")
    AudioSegment.from_file(wav_file)[start_ms:end_ms].export(tmp_2s, format="wav")
    tmp_10s = extract_10s_window(wav_file, center_ms)

    result = {"time": f"{start_ms/1000:.1f}s - {end_ms/1000:.1f}s"}

    with torch.no_grad():
        if mode == "ast_v7":
            x = prep_ast_input(tmp_10s).to(device)
            probs = F.softmax(MODELS["s2_ast_v7"](x), dim=1).squeeze(0).cpu().numpy()
            pred = S2_LABELS[probs.argmax()]
            result.update({"prediction": pred, "probs": {S2_LABELS[i]: float(probs[i]) for i in range(4)}, "type": "AST Standalone"})

        elif mode == "passt":
            x = prep_passt_input(tmp_10s).to(device)
            probs = F.softmax(MODELS["s2_passt"](x), dim=1).squeeze(0).cpu().numpy()
            pred = S2_LABELS[probs.argmax()]
            result.update({"prediction": pred, "probs": {S2_LABELS[i]: float(probs[i]) for i in range(4)}, "type": "PaSST Standalone"})

        elif mode.startswith("end_to_end"):
            # Stage 1 binary detection. The stage-1 model was trained with 10 s
            # of context, so we feed the 10 s window, not the 2 s slice: feeding
            # 2 s is out-of-distribution and makes stage 1 over-predict "drone".
            x_s1 = prep_ast_input(tmp_10s).to(device)
            p_s1 = F.softmax(MODELS["s1_ast"](x_s1), dim=1).squeeze(0).cpu().numpy()
            stage1_type = "Stage 1 Rejection (AST)"

            # Attach the stage-1 decision and confidence to every result so the
            # UI can always show how sure we are the segment is a drone.
            p_drone = float(p_s1[1])
            is_drone = p_drone >= 0.5
            result["stage1"] = {
                "label": "drone" if is_drone else "no_drone",
                "confidence": p_drone if is_drone else float(p_s1[0]),
                "p_drone": p_drone,
            }

            if not is_drone:
                result.update({"prediction": "No Drone Detected", "type": stage1_type, "confidence": float(p_s1[0])})
            else:
                # 2. Stage A: 2-Motors Gate (AST + PaSST avg)
                x_ast = prep_ast_input(tmp_10s).to(device)
                p_ast_raw = F.softmax(MODELS["s2_ast_v7"](x_ast), dim=1).squeeze(0).cpu().numpy()
                p_ast_cal = apply_temperature(p_ast_raw, T_AST)
                
                x_passt = prep_passt_input(tmp_10s).to(device)
                p_passt_raw = F.softmax(MODELS["s2_passt"](x_passt), dim=1).squeeze(0).cpu().numpy()
                p_passt_cal = apply_temperature(p_passt_raw, T_PASST)
                
                p_2_avg = (p_ast_cal[0] + p_passt_cal[0]) / 2.0
                
                if p_2_avg > 0.5:
                    # Stage A gate fired: report the averaged AST/PaSST probabilities.
                    raw_4way = (p_ast_cal + p_passt_cal) / 2.0
                    result.update({"prediction": "2_motors", "type": "Stage A Gate", "probs": {S2_LABELS[i]: float(raw_4way[i]) for i in range(4)}})
                else:
                    # 3. Stage B: Blend {4, 6, 8}
                    ast_b = p_ast_cal[1:] / p_ast_cal[1:].sum()
                    passt_b = p_passt_cal[1:] / p_passt_cal[1:].sum()
                    
                    has_trees = False
                    if "xgb_2s" in MODELS and "lgbm_10s" in MODELS:
                        try:
                            from data.extract_features import extract_for_file as ex_2s
                            from data.extract_features_10s import extract_for_file as ex_10s
                            f_2s = ex_2s(tmp_2s, FEAT_NAMES_2S)
                            f_10s = ex_10s(tmp_10s, FEAT_NAMES_10S)
                            p_xgb = apply_temperature(MODELS["xgb_2s"].predict_proba(f_2s)[0], T_XGB)
                            p_lgbm = apply_temperature(MODELS["lgbm_10s"].predict_proba(f_10s)[0], T_LGBM)
                            has_trees = True
                        except Exception as e:
                            global _tree_warned
                            if not _tree_warned:
                                import traceback
                                print(f"[WARN] Tree feature extraction failed, "
                                      f"falling back to DL-only Stage B: {e}")
                                traceback.print_exc()
                                _tree_warned = True
                    
                    if has_trees:
                        blend = 0.0 * ast_b + 0.9 * passt_b + 0.0 * p_xgb + 0.1 * p_lgbm
                        blend_type = "Stage B (4-Way Full Cascade)"
                    else:
                        blend = 0.0 * ast_b + 1.0 * passt_b 
                        blend_type = "Stage B (Deep Learning Fallback)"

                    pred = STAGE_B_LABELS[blend.argmax()]
                    
                    # Build a single 4-way distribution: split the remaining
                    # (1 - p_2_avg) across the {4,6,8} classes by the blend.
                    full_probs = [float(p_2_avg)] + [float((1.0 - p_2_avg) * b) for b in blend]
                    result.update({"prediction": pred, "type": blend_type, "probs": {S2_LABELS[i]: float(full_probs[i]) for i in range(4)}})

    os.remove(tmp_2s)
    os.remove(tmp_10s)
    return result

def compute_aggregate(segments):
    """Combine the per-segment results into a single verdict for the whole clip.

    Detection (end-to-end modes): flag a drone when a sustained run of segments
    crosses the threshold, and report peak/mean p_drone and how many fired.
    Motor count: a p_drone-weighted mean of the 4-way probabilities, or an
    unweighted mean for the standalone AST/PaSST modes that have no stage 1.
    """
    agg = {"n_segments": len(segments)}
    if not segments:
        return agg

    # Detection (end-to-end modes only): flag a drone if any sustained run of
    # consecutive segments crosses the threshold.
    p_drones = [s["stage1"]["p_drone"] for s in segments if "stage1" in s]
    if p_drones:
        flags = [p >= PRESENCE_THRESHOLD for p in p_drones]
        n_flagged = sum(flags)
        longest_run = _run = 0
        for f in flags:
            _run = _run + 1 if f else 0
            longest_run = max(longest_run, _run)
        is_drone = longest_run >= PRESENCE_MIN_CONSEC
        peak = max(p_drones)
        agg["detection"] = {
            "verdict": "drone" if is_drone else "no_drone",
            # Confidence = strongest detection when a drone is found; otherwise
            # how far the strongest segment stayed below "drone".
            "confidence": peak if is_drone else 1.0 - peak,
            "peak_p_drone": peak,
            "mean_p_drone": sum(p_drones) / len(p_drones),
            "n_segments_drone": n_flagged,
            "longest_run": longest_run,
            "frac_segments_drone": n_flagged / len(p_drones),
        }

    # --- Motor count (gate-weighted mean of 4-way probs) ---
    if p_drones:
        # Only segments that passed stage 1 carry a 4-way "probs"; weight by p_drone.
        weighted = [(s["probs"], s["stage1"]["p_drone"])
                    for s in segments if "probs" in s and s.get("stage1", {}).get("p_drone", 0) >= 0.5]
    else:
        # Standalone modes: every segment has probs, weight them equally.
        weighted = [(s["probs"], 1.0) for s in segments if "probs" in s]

    if weighted:
        total_w = sum(w for _, w in weighted)
        mean_probs = {lab: sum(p[lab] * w for p, w in weighted) / total_w for lab in S2_LABELS}
        agg["motor"] = {
            "verdict": max(mean_probs, key=mean_probs.get),
            "probs": mean_probs,
            "n_segments": len(weighted),
            "weighted": bool(p_drones),
        }

    return agg

# --- API Endpoints ---
@app.route("/api/predict", methods=["POST"])
def api_predict():
    global _temp_dir
    mode = request.form.get("model")
    file = request.files.get("file")
    if not file: return jsonify({"error": "No file uploaded"}), 400

    if mode not in REQUIRED_MODELS:
        return jsonify({"error": f"Unknown pipeline '{mode}'"}), 400
    missing = [m for m in REQUIRED_MODELS[mode] if m not in MODELS]
    if missing:
        return jsonify({"error": f"Pipeline '{mode}' is unavailable, missing model(s): "
                                 f"{', '.join(missing)}. See download_weights.py."}), 400

    if _temp_dir and os.path.exists(_temp_dir):
        shutil.rmtree(_temp_dir, ignore_errors=True)

    _temp_dir = tempfile.mkdtemp(prefix="drone_demo_")
    wav_path = os.path.join(_temp_dir, "upload.wav")
    file.save(wav_path)

    # Segment
    audio = normalize(AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(16000))
    audio.export(wav_path, format="wav")
    duration_ms = len(audio)
    
    if duration_ms < MIN_INPUT_SECONDS * 1000:
        return jsonify({"error": f"Audio is too short ({duration_ms/1000:.1f}s). "
                                 f"Please upload at least {MIN_INPUT_SECONDS} seconds. The models "
                                 f"need ~10s of context and the detector needs a few seconds to confirm."}), 400
    if duration_ms > MAX_INPUT_SECONDS * 1000:
        return jsonify({"error": f"Audio is too long ({duration_ms/1000:.1f}s). "
                                 f"Please upload at most {MAX_INPUT_SECONDS} seconds "
                                 f"(trim the clip to the part you want to analyse)."}), 400

    segments = []
    start = 0
    idx = 0
    
    serve_dir = os.path.join(_temp_dir, "serve")
    os.makedirs(serve_dir)

    while start + SEG_LEN_MS <= duration_ms:
        res = run_segment(wav_path, start, start + SEG_LEN_MS, mode)
        res["idx"] = idx
        segments.append(res)
        
        # Save slice for frontend playback
        audio[start:start + SEG_LEN_MS].export(os.path.join(serve_dir, f"{idx}.wav"), format="wav")
        
        start += STEP_MS
        idx += 1

    aggregate = compute_aggregate(segments)
    return jsonify({"segments": segments, "duration_s": duration_ms / 1000, "aggregate": aggregate})

@app.route("/api/audio/<int:idx>")
def api_audio(idx):
    if not _temp_dir: return Response("Not found", 404)
    return send_file(os.path.join(_temp_dir, "serve", f"{idx}.wav"), mimetype="audio/wav")

# --- Frontend HTML ---
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drone Audio Inference</title>
<style>
  :root { --bg: #0f172a; --surface: #1e293b; --accent: #3b82f6; --text: #f8fafc; --text-dim: #94a3b8; }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }
  .container { max-width: 900px; margin: 0 auto; }
  .card { background: var(--surface); padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
  h1 { margin-top: 0; color: var(--accent); }
  select, input[type="file"], button { padding: 0.5rem; margin-top: 0.5rem; border-radius: 4px; border: 1px solid #334155; background: #0f172a; color: white; }
  button { background: var(--accent); cursor: pointer; border: none; font-weight: bold; padding: 0.6rem 1.2rem; }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }
  th { color: var(--text-dim); text-transform: uppercase; font-size: 0.85rem; }
  .badge { padding: 0.25rem 0.5rem; border-radius: 99px; font-size: 0.85rem; font-weight: 500; background: #334155; }
  .badge.drone { background: #10b981; color: white; }
  .badge.nodrone { background: #ef4444; color: white; }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Acoustic Drone Recognition</h1>
    <p style="color:var(--text-dim); font-size:0.9rem;">Upload a .wav or .mp3 file (6-30 seconds) to test the inference pipeline.</p>
    <div style="display: flex; gap: 1rem; align-items: flex-end; margin-top: 1rem;">
      <div>
        <label style="font-size:0.85rem; color:var(--text-dim)">Audio File</label><br>
        <input type="file" id="file" accept="audio/*">
      </div>
      <div>
        <label style="font-size:0.85rem; color:var(--text-dim)">Pipeline / Model</label><br>
        <select id="mode">
          <option value="end_to_end_ast">End-to-End 4-Way Cascade</option>
          <option value="ast_v7">AST v7 Standalone (10s Context)</option>
          <option value="passt">PaSST v1 Standalone (10s Context)</option>
        </select>
      </div>
      <button id="btn" onclick="runInference()">Run Inference</button>
    </div>
    <div id="status" style="margin-top:1rem; font-size:0.9rem; color:var(--accent);"></div>
  </div>

  <div class="card" id="overall-card" style="display:none;">
    <h2>Overall Result</h2>
    <div id="overall"></div>
  </div>

  <div class="card" id="results-card" style="display:none;">
    <h2>Per-Segment Results</h2>
    <table>
      <thead><tr><th>Time</th><th>Play</th><th>Detection</th><th>Prediction</th><th>Probabilities</th><th>Logic Block</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<audio id="player" style="display:none;"></audio>

<script>
async function runInference() {
  const file = document.getElementById('file').files[0];
  if (!file) return alert('Select a file first');
  
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = "Processing audio and running neural networks. Please wait...";
  document.getElementById('results-card').style.display = 'none';
  document.getElementById('overall-card').style.display = 'none';

  const fd = new FormData();
  fd.append('file', file);
  fd.append('model', document.getElementById('mode').value);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    let html = '';
    data.segments.forEach(seg => {
      let badgeClass = seg.prediction.includes("No Drone") ? "nodrone" : "drone";
      
      // Stage-1 drone / no-drone detection and confidence.
      let detHtml = '<span style="color:var(--text-dim)">-</span>';
      if (seg.stage1) {
          const isDrone = seg.stage1.label === 'drone';
          detHtml = `<span class="badge ${isDrone?'drone':'nodrone'}">${isDrone?'Drone':'No Drone'}</span>
              <span style="font-size:0.85rem; color:var(--text-dim);">${(seg.stage1.confidence*100).toFixed(1)}%</span>`;
      }

      // Build the probabilities cell.
      let probsHtml = '<span style="color:var(--text-dim)">-</span>';
      if (seg.probs) {
          probsHtml = `<div style="font-size:0.85rem; color:var(--text-dim); line-height:1.5;">
              <span style="color:${seg.prediction==='2_motors'?'#fff':'inherit'}; font-weight:${seg.prediction==='2_motors'?'bold':'normal'}">2M: ${(seg.probs['2_motors']*100).toFixed(1)}%</span> &nbsp;|&nbsp;
              <span style="color:${seg.prediction==='4_motors'?'#fff':'inherit'}; font-weight:${seg.prediction==='4_motors'?'bold':'normal'}">4M: ${(seg.probs['4_motors']*100).toFixed(1)}%</span><br>
              <span style="color:${seg.prediction==='6_motors'?'#fff':'inherit'}; font-weight:${seg.prediction==='6_motors'?'bold':'normal'}">6M: ${(seg.probs['6_motors']*100).toFixed(1)}%</span> &nbsp;|&nbsp;
              <span style="color:${seg.prediction==='8_motors'?'#fff':'inherit'}; font-weight:${seg.prediction==='8_motors'?'bold':'normal'}">8M: ${(seg.probs['8_motors']*100).toFixed(1)}%</span>
          </div>`;
      }

      html += `<tr>
        <td>${seg.time}</td>
        <td><button onclick="play(${seg.idx})" style="padding:0.3rem 0.6rem">Play</button></td>
        <td>${detHtml}</td>
        <td><span class="badge ${badgeClass}">${seg.prediction}</span></td>
        <td>${probsHtml}</td>
        <td style="font-size:0.85rem; color:var(--text-dim)">${seg.type}</td>
      </tr>`;
    });
    document.getElementById('tbody').innerHTML = html;
    document.getElementById('results-card').style.display = 'block';

    // Overall (combined) verdict for the whole clip.
    const agg = data.aggregate || {};
    let oHtml = '';
    if (agg.detection) {
      const d = agg.detection, isD = d.verdict === 'drone';
      oHtml += `<div style="font-size:1.1rem; margin-bottom:0.4rem;">
          <span class="badge ${isD?'drone':'nodrone'}" style="font-size:1rem;">${isD?'DRONE DETECTED':'NO DRONE'}</span>
          <span style="margin-left:0.5rem;">${(d.confidence*100).toFixed(1)}% confidence</span></div>
        <div style="font-size:0.85rem; color:var(--text-dim); margin-bottom:0.75rem;">
          peak P(drone) ${(d.peak_p_drone*100).toFixed(1)}% &nbsp;|&nbsp; ${d.n_segments_drone}/${agg.n_segments} segments flagged (longest run ${d.longest_run}) &nbsp;|&nbsp; mean ${(d.mean_p_drone*100).toFixed(1)}%</div>`;
    }
    if (agg.motor && (!agg.detection || agg.detection.verdict === 'drone')) {
      const m = agg.motor;
      oHtml += `<div style="font-size:1rem;">Motor count: <span class="badge drone">${m.verdict}</span>
          <span style="font-size:0.8rem; color:var(--text-dim); margin-left:0.5rem;">(${m.weighted?'drone-weighted mean':'mean'} over ${m.n_segments} segment${m.n_segments===1?'':'s'})</span></div>
        <div style="font-size:0.85rem; color:var(--text-dim); margin-top:0.35rem;">
          2M ${(m.probs['2_motors']*100).toFixed(1)}% &nbsp;|&nbsp; 4M ${(m.probs['4_motors']*100).toFixed(1)}% &nbsp;|&nbsp; 6M ${(m.probs['6_motors']*100).toFixed(1)}% &nbsp;|&nbsp; 8M ${(m.probs['8_motors']*100).toFixed(1)}%</div>`;
    }
    if (oHtml) {
      document.getElementById('overall').innerHTML = oHtml;
      document.getElementById('overall-card').style.display = 'block';
    }
    status.textContent = "Inference complete.";
  } catch (e) {
    status.textContent = "Error: " + e.message;
  }
  btn.disabled = false;
}

function play(idx) {
  const p = document.getElementById('player');
  p.src = '/api/audio/' + idx;
  p.play();
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

if __name__ == "__main__":
    load_models()
    # Suppress Flask/Werkzeug's dev-server banner for a clean startup.
    try:
        import flask.cli
        flask.cli.show_server_banner = lambda *a, **k: None
    except Exception:
        pass
    print("\nAcoustic Drone Recognition - running at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5000)
