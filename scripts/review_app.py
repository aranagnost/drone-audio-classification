#!/usr/bin/env python3
"""
review_app.py

Local Flask web app for reviewing drone audio segments AND discovering
new drone audio from YouTube / local files.

Usage:
    python review_app.py              # default port 5000
    python review_app.py --port 8080  # custom port

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from flask import Flask, jsonify, request, send_file, Response

# ──────────────────── paths (same as discover_drone_audio.py) ────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "review_queue"
QUEUE_META = QUEUE_DIR / "queue.json"
DATASET_AUDIO = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "audio"
DATASET_META = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "metadata.json"
CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "stage1_3stages_cnnv1.pt"

SEG_DURATION = 2.0  # seconds — matches training config

# ──────────────────── audio config (must match training) ──────────────────────
SEG_LEN_MS = 2000
STEP_MS = 1500
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
F_MIN = 20
CLIP_SAMPLES = int(SAMPLE_RATE * (SEG_LEN_MS / 1000))

YTDLP = shutil.which("yt-dlp")

app = Flask(__name__)

# ──────────────────── model (lazy-loaded) ─────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def _import_torch():
    """Import torch/torchaudio lazily so the review tab works without them."""
    import torch
    import torch.nn as nn
    import torchaudio
    return torch, nn, torchaudio


def get_model():
    """Return the ThreeStagesCNNv1 stage-1 model (SmallCNNv2), loading it on first call."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        torch, nn, torchaudio = _import_torch()

        class _SEBlock(nn.Module):
            def __init__(self, channels: int, reduction: int = 4):
                super().__init__()
                mid = max(channels // reduction, 4)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, mid),
                    nn.ReLU(inplace=True),
                    nn.Linear(mid, channels),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                b, c, _, _ = x.shape
                w = self.pool(x).view(b, c)
                w = self.fc(w).view(b, c, 1, 1)
                return x * w

        class _ResBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.se = _SEBlock(out_ch)
                self.relu = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d((2, 2))
                self.dropout = nn.Dropout2d(dropout)
                self.shortcut = (
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                    )
                    if in_ch != out_ch else nn.Identity()
                )

            def forward(self, x):
                identity = self.shortcut(x)
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = self.se(out)
                out = self.relu(out + identity)
                out = self.pool(out)
                out = self.dropout(out)
                return out

        class SmallCNNv2(nn.Module):
            def __init__(self, num_classes: int, dropout: float = 0.2):
                super().__init__()
                self.features = nn.Sequential(
                    _ResBlock(1, 16, dropout=dropout),
                    _ResBlock(16, 32, dropout=dropout),
                    _ResBlock(32, 64, dropout=dropout),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.head = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                self.classifier = nn.Linear(32, num_classes)

            def forward(self, x):
                z = self.features(x)
                z = z.flatten(1)
                z = self.head(z)
                return self.classifier(z)

        ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
        model = SmallCNNv2(num_classes=2)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model = model
        return _model


def wav_to_logmel(wav_path: Path):
    """Load a 2-second wav and return a (1, 1, n_mels, time) log-mel tensor."""
    torch, _, torchaudio = _import_torch()
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] < CLIP_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :CLIP_SAMPLES]
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, f_min=F_MIN, n_mels=N_MELS,
    )
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-9)
    return log_mel.unsqueeze(0)


# ──────────────────── YouTube / audio helpers ─────────────────────────────────
def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def extract_youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")
    elif "youtube.com" in parsed.netloc:
        if "watch" in parsed.path:
            return parse_qs(parsed.query).get("v", ["ytclip"])[0]
        elif "shorts" in parsed.path:
            return parsed.path.split("/")[-1]
    return "ytclip"


def search_youtube(query: str, max_results: int) -> list[str]:
    if not YTDLP:
        return []
    result = subprocess.run(
        [YTDLP, f"ytsearch{max_results}:{query}",
         "--get-url", "--get-id", "--flat-playlist", "--no-warnings"],
        capture_output=True, text=True,
    )
    ids = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    return [f"https://www.youtube.com/watch?v={vid_id}" for vid_id in ids if len(vid_id) == 11]


def download_youtube_audio(url: str, log) -> tuple[Path, str]:
    vid_id = extract_youtube_id(url)
    tmp_wav = Path(tempfile.gettempdir()) / f"yt_audio_{vid_id}.wav"
    if tmp_wav.exists():
        tmp_wav.unlink()
    log(f"Downloading: {url}")
    subprocess.run(
        [YTDLP, "-x", "--audio-format", "wav", "-o", str(tmp_wav), url],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return tmp_wav, vid_id


def convert_to_wav(src_path: Path) -> Path:
    if src_path.suffix.lower() != ".wav":
        tmp_wav = Path(tempfile.gettempdir()) / "discover_temp_audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(tmp_wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return tmp_wav
    return src_path


def segment_wav_to_dir(wav_path: Path, out_dir: Path, prefix: str,
                       start_ms: int = 0, end_ms: int | None = None) -> list[Path]:
    from pydub import AudioSegment as PydubSeg
    from pydub.effects import normalize
    audio = PydubSeg.from_wav(str(wav_path))
    if end_ms is None or end_ms > len(audio):
        end_ms = len(audio)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, start in enumerate(range(start_ms, end_ms - SEG_LEN_MS, STEP_MS)):
        seg = audio[start:start + SEG_LEN_MS]
        seg = normalize(seg)
        fname = f"{prefix}_{i:03d}.wav"
        fpath = out_dir / fname
        seg.export(str(fpath), format="wav")
        paths.append(fpath)
    return paths


def format_ms(ms: int) -> str:
    total_sec = ms // 1000
    m, s = divmod(total_sec, 60)
    return f"{m}m{s:02d}s"


def _parse_time(s: str) -> int:
    parts = s.split(":")
    if len(parts) == 2:
        return (int(parts[0]) * 60 + int(parts[1])) * 1000
    elif len(parts) == 3:
        return (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
    raise ValueError(f"Invalid time: {s}")


def parse_time_range(s: str) -> tuple[int, int]:
    left, sep, right = s.partition("-")
    if not sep:
        raise ValueError(f"Invalid range (no '-'): {s}")
    return _parse_time(left), _parse_time(right)


def build_known_sources() -> set[str]:
    known: set[str] = set()
    for e in load_queue():
        known.add(e.get("source_ref", ""))
    for e in load_dataset_metadata():
        known.add(e.get("youtube_url", ""))
        known.add(e.get("local_path", ""))
    known.discard("")
    return known


# ──────────────────── background job system ───────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _process_source_bg(job_id: str, source: str, ranges: list[tuple[int, int]],
                       threshold: float, motor_label: str | None):
    """Run process_source in a background thread, writing logs to the job."""
    torch, _, _ = _import_torch()

    def log(msg: str):
        with _jobs_lock:
            _jobs[job_id]["logs"].append(msg)

    try:
        log(f"Processing: {source}")

        # Download / locate
        if is_url(source):
            wav_path, source_id = download_youtube_audio(source, log)
            source_type = "youtube"
            source_ref = source
        else:
            src_path = Path(source)
            if not src_path.exists():
                log(f"ERROR: File not found: {source}")
                with _jobs_lock:
                    _jobs[job_id]["status"] = "error"
                return
            wav_path = convert_to_wav(src_path)
            source_id = src_path.stem
            source_type = "local"
            source_ref = str(src_path.resolve())

        from pydub import AudioSegment as PydubSeg
        audio_obj = PydubSeg.from_wav(str(wav_path))
        log(f"Duration: {format_ms(len(audio_obj))}")

        tmp_seg_dir = Path(tempfile.mkdtemp(prefix="drone_discover_"))

        if ranges:
            seg_paths: list[Path] = []
            for ri, (start_ms, end_ms) in enumerate(ranges):
                range_prefix = f"{source_id}_r{ri}"
                paths = segment_wav_to_dir(wav_path, tmp_seg_dir, range_prefix,
                                           start_ms=start_ms, end_ms=end_ms)
                seg_paths.extend(paths)
                log(f"Range {format_ms(start_ms)}-{format_ms(end_ms)}: {len(paths)} segments")
        else:
            seg_paths = segment_wav_to_dir(wav_path, tmp_seg_dir, source_id)

        log(f"{len(seg_paths)} chunks total. Running pre-classifier...")

        model = get_model()
        queue = load_queue()
        kept = 0
        for seg_path in seg_paths:
            logmel = wav_to_logmel(seg_path)
            with torch.no_grad():
                logits = model(logmel)
                probs = torch.softmax(logits, dim=1)
                drone_prob = probs[0, 1].item()

            if drone_prob >= threshold:
                queue_subdir = QUEUE_DIR / source_id
                queue_subdir.mkdir(parents=True, exist_ok=True)
                dest = queue_subdir / seg_path.name
                shutil.move(str(seg_path), str(dest))
                queue.append({
                    "filename": seg_path.name,
                    "source_id": source_id,
                    "queue_path": str(dest),
                    "drone_prob": round(drone_prob, 4),
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "motor_hint": motor_label,
                })
                kept += 1
            else:
                seg_path.unlink()

        shutil.rmtree(tmp_seg_dir, ignore_errors=True)
        save_queue(queue)
        log(f"Queued {kept}/{len(seg_paths)} segments (threshold={threshold})")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {"queued": kept, "total": len(seg_paths)}

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["logs"].append(f"ERROR: {e}")
            _jobs[job_id]["status"] = "error"


# ──────────────────── queue / metadata helpers ───────────────────────────────
def load_queue() -> list[dict]:
    if QUEUE_META.exists() and QUEUE_META.stat().st_size > 0:
        try:
            with open(QUEUE_META) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_queue(data: list[dict]):
    QUEUE_META.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_META, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset_metadata() -> list[dict]:
    if DATASET_META.exists() and DATASET_META.stat().st_size > 0:
        try:
            with open(DATASET_META) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_dataset_metadata(data: list[dict]):
    DATASET_META.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_META, "w") as f:
        json.dump(data, f, indent=2)


def cleanup_empty_dirs():
    """Remove empty source directories inside the review queue."""
    if QUEUE_DIR.exists():
        for d in QUEUE_DIR.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()


# ──────────────────── API endpoints ──────────────────────────────────────────
@app.route("/api/queue")
def api_queue():
    queue = load_queue()
    queue.reverse()  # LIFO: most recently added first
    return jsonify(queue)


@app.route("/api/audio/<path:filepath>")
def api_audio(filepath):
    full = QUEUE_DIR / filepath
    if not full.exists():
        return Response("File not found", status=404)
    return send_file(full, mimetype="audio/wav")


@app.route("/api/accept", methods=["POST"])
def api_accept():
    body = request.get_json(force=True)
    filename = body["filename"]
    source_id = body["source_id"]
    motor_label = body["motor_label"]
    quality = int(body["quality"])

    seg_path = QUEUE_DIR / source_id / filename
    if not seg_path.exists():
        return jsonify({"error": "File not found"}), 404

    dest_dir = DATASET_AUDIO / motor_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.move(str(seg_path), str(dest))

    # find the queue entry to get source info
    queue = load_queue()
    entry = next((e for e in queue if e["filename"] == filename and e["source_id"] == source_id), None)

    meta_entry = {
        "filename": filename,
        "binary_label": "drone",
        "motor_label": motor_label,
        "quality": quality,
        "source": entry["source_type"] if entry else "unknown",
        "duration": SEG_DURATION,
    }
    if entry:
        if entry["source_type"] == "youtube":
            meta_entry["youtube_url"] = entry["source_ref"]
        else:
            meta_entry["local_path"] = entry["source_ref"]

    ds_meta = load_dataset_metadata()
    ds_meta.append(meta_entry)
    save_dataset_metadata(ds_meta)

    queue = [e for e in queue if not (e["filename"] == filename and e["source_id"] == source_id)]
    save_queue(queue)
    cleanup_empty_dirs()

    return jsonify({"ok": True, "remaining": len(queue)})


@app.route("/api/accept_nodrone", methods=["POST"])
def api_accept_nodrone():
    body = request.get_json(force=True)
    filename = body["filename"]
    source_id = body["source_id"]
    subtype = body["subtype"]

    seg_path = QUEUE_DIR / source_id / filename
    if not seg_path.exists():
        return jsonify({"error": "File not found"}), 404

    dest_dir = DATASET_AUDIO / "not_a_drone" / subtype
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.move(str(seg_path), str(dest))

    queue = load_queue()
    entry = next((e for e in queue if e["filename"] == filename and e["source_id"] == source_id), None)

    meta_entry = {
        "filename": filename,
        "binary_label": "no_drone",
        "motor_label": None,
        "subtype": subtype,
        "source": entry["source_type"] if entry else "unknown",
        "duration": SEG_DURATION,
    }
    if entry:
        if entry["source_type"] == "youtube":
            meta_entry["youtube_url"] = entry["source_ref"]
        else:
            meta_entry["local_path"] = entry["source_ref"]

    ds_meta = load_dataset_metadata()
    ds_meta.append(meta_entry)
    save_dataset_metadata(ds_meta)

    queue = [e for e in queue if not (e["filename"] == filename and e["source_id"] == source_id)]
    save_queue(queue)
    cleanup_empty_dirs()

    return jsonify({"ok": True, "remaining": len(queue)})


@app.route("/api/discard", methods=["POST"])
def api_discard():
    body = request.get_json(force=True)
    filename = body["filename"]
    source_id = body["source_id"]

    seg_path = QUEUE_DIR / source_id / filename
    if seg_path.exists():
        seg_path.unlink()

    queue = load_queue()
    queue = [e for e in queue if not (e["filename"] == filename and e["source_id"] == source_id)]
    save_queue(queue)
    cleanup_empty_dirs()

    return jsonify({"ok": True, "remaining": len(queue)})


@app.route("/api/bulk_discard", methods=["POST"])
def api_bulk_discard():
    body = request.get_json(force=True)
    source_id = body["source_id"]

    queue = load_queue()
    to_delete = [e for e in queue if e["source_id"] == source_id]
    for entry in to_delete:
        seg_path = QUEUE_DIR / entry["source_id"] / entry["filename"]
        if seg_path.exists():
            seg_path.unlink()

    queue = [e for e in queue if e["source_id"] != source_id]
    save_queue(queue)
    cleanup_empty_dirs()

    return jsonify({"ok": True, "deleted": len(to_delete), "remaining": len(queue)})


@app.route("/api/bulk_nodrone", methods=["POST"])
def api_bulk_nodrone():
    body = request.get_json(force=True)
    source_id = body["source_id"]
    subtype = body["subtype"]

    queue = load_queue()
    to_move = [e for e in queue if e["source_id"] == source_id]

    dest_dir = DATASET_AUDIO / "not_a_drone" / subtype
    dest_dir.mkdir(parents=True, exist_ok=True)

    ds_meta = load_dataset_metadata()
    for entry in to_move:
        seg_path = QUEUE_DIR / entry["source_id"] / entry["filename"]
        if not seg_path.exists():
            continue
        shutil.move(str(seg_path), str(dest_dir / entry["filename"]))
        meta_entry = {
            "filename": entry["filename"],
            "binary_label": "no_drone",
            "motor_label": None,
            "subtype": subtype,
            "source": entry["source_type"],
            "duration": SEG_DURATION,
        }
        if entry["source_type"] == "youtube":
            meta_entry["youtube_url"] = entry["source_ref"]
        else:
            meta_entry["local_path"] = entry["source_ref"]
        ds_meta.append(meta_entry)
    save_dataset_metadata(ds_meta)

    queue = [e for e in queue if e["source_id"] != source_id]
    save_queue(queue)
    cleanup_empty_dirs()

    return jsonify({"ok": True, "moved": len(to_move), "remaining": len(queue)})


@app.route("/api/empty_queue", methods=["POST"])
def api_empty_queue():
    queue = load_queue()
    if not queue:
        return jsonify({"ok": True, "deleted": 0})

    for entry in queue:
        seg_path = Path(entry["queue_path"])
        if seg_path.exists():
            seg_path.unlink()

    save_queue([])
    cleanup_empty_dirs()

    return jsonify({"ok": True, "deleted": len(queue)})


@app.route("/api/stats")
def api_stats():
    queue = load_queue()
    ds_meta = load_dataset_metadata()

    # dataset counts per motor label
    class_counts = {}
    for e in ds_meta:
        label = e.get("motor_label") or e.get("binary_label", "unknown")
        class_counts[label] = class_counts.get(label, 0) + 1

    return jsonify({
        "queue_size": len(queue),
        "dataset_total": len(ds_meta),
        "class_counts": class_counts,
    })


# ──────────────────── discover API endpoints ──────────────────────────────────
@app.route("/api/discover_info")
def api_discover_info():
    return jsonify({
        "ytdlp_available": YTDLP is not None,
        "model_available": CHECKPOINT.exists(),
    })


@app.route("/api/search_youtube", methods=["POST"])
def api_search_youtube():
    body = request.get_json(force=True)
    query = body.get("query", "").strip()
    count = int(body.get("count", 5))
    if not query:
        return jsonify({"error": "Empty query"}), 400
    if not YTDLP:
        return jsonify({"error": "yt-dlp not installed"}), 500

    known = build_known_sources()
    raw_urls = search_youtube(query, count * 3)
    all_new = [u for u in raw_urls if u not in known]
    skipped = len(raw_urls) - len(all_new)
    new_urls = all_new[:count]
    return jsonify({"urls": new_urls, "skipped_known": skipped})


@app.route("/api/process_url", methods=["POST"])
def api_process_url():
    body = request.get_json(force=True)
    source = body.get("url", "").strip()
    ranges_raw = body.get("ranges", "").strip()
    motor = body.get("motor")
    threshold = float(body.get("threshold", 0.3))

    if not source:
        return jsonify({"error": "No URL or path provided"}), 400
    if not CHECKPOINT.exists():
        return jsonify({"error": "Model checkpoint not found"}), 500

    # Check for duplicates (resolve local paths for comparison)
    known = build_known_sources()
    ref = source if is_url(source) else str(Path(source).resolve())
    if ref in known:
        return jsonify({"error": "Already processed. Remove from queue/dataset first to reprocess."}), 409

    motor_label = f"{motor}_motors" if motor else None

    ranges: list[tuple[int, int]] = []
    if ranges_raw:
        for token in ranges_raw.split():
            try:
                ranges.append(parse_time_range(token))
            except ValueError:
                return jsonify({"error": f"Invalid time range: {token}"}), 400

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "logs": [], "result": None}

    t = threading.Thread(target=_process_source_bg,
                         args=(job_id, source, ranges, threshold, motor_label),
                         daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/job_status/<job_id>")
def api_job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


# ──────────────────── frontend (single-page inline HTML) ─────────────────────
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drone Audio Review</title>
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
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}

/* ── tab bar ── */
.tab-bar {
  display: flex;
  background: var(--surface);
  border-bottom: 2px solid var(--surface2);
}
.tab-bar button {
  padding: 10px 28px;
  border: none;
  background: transparent;
  color: var(--text-dim);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  transition: all 0.15s;
}
.tab-bar button:hover { color: var(--text); }
.tab-bar button.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

/* ── stats bar ── */
.stats-bar {
  display: flex;
  gap: 16px;
  padding: 12px 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--surface2);
  flex-wrap: wrap;
  align-items: center;
}
.stats-bar .stat {
  font-size: 13px;
  color: var(--text-dim);
}
.stats-bar .stat b {
  color: var(--text);
  font-size: 15px;
}
.stats-bar h1 {
  font-size: 16px;
  margin-right: auto;
  color: var(--accent);
}

/* ── layout ── */
.container {
  display: flex;
  height: calc(100vh - 90px);
}
.queue-panel {
  width: 420px;
  min-width: 320px;
  border-right: 1px solid var(--surface2);
  display: flex;
  flex-direction: column;
  background: var(--surface);
}
.queue-header {
  padding: 10px 14px;
  font-size: 13px;
  color: var(--text-dim);
  border-bottom: 1px solid var(--surface2);
  display: flex;
  justify-content: space-between;
}
.queue-list {
  flex: 1;
  overflow-y: auto;
}
.queue-item {
  padding: 10px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  cursor: pointer;
  transition: background 0.15s;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.queue-item:hover { background: var(--surface2); }
.queue-item.active { background: var(--accent2); }
.queue-item .filename {
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.queue-item .meta {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: var(--text-dim);
}
.queue-item .prob {
  color: var(--green);
  font-weight: 600;
}

/* ── review panel ── */
.review-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 30px 20px;
  overflow-y: auto;
}
.review-panel.empty {
  justify-content: center;
}
.empty-msg {
  color: var(--text-dim);
  font-size: 18px;
}
.segment-info {
  text-align: center;
  margin-bottom: 20px;
}
.segment-info h2 {
  font-size: 18px;
  margin-bottom: 6px;
}
.segment-info .details {
  font-size: 13px;
  color: var(--text-dim);
}
.segment-info a {
  color: var(--accent);
  text-decoration: none;
}
.segment-info a:hover { text-decoration: underline; }

audio {
  width: 100%;
  max-width: 500px;
  margin-bottom: 28px;
}

/* ── label controls ── */
.controls {
  display: flex;
  flex-direction: column;
  gap: 16px;
  width: 100%;
  max-width: 500px;
}
.control-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.control-group label {
  font-size: 12px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.btn-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.btn {
  padding: 8px 18px;
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px;
  background: var(--surface2);
  color: var(--text);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.15s;
  user-select: none;
}
.btn:hover { border-color: var(--accent); background: rgba(233,69,96,0.15); }
.btn.selected { background: var(--accent); border-color: var(--accent); }
.btn.accept {
  background: var(--green);
  border-color: var(--green);
  font-weight: 600;
  padding: 10px 28px;
}
.btn.accept:hover { opacity: 0.85; }
.btn.accept:disabled { opacity: 0.35; cursor: not-allowed; }
.btn.discard {
  background: var(--red);
  border-color: var(--red);
  font-weight: 600;
  padding: 10px 28px;
}
.btn.discard:hover { opacity: 0.85; }
.btn.nodrone {
  background: var(--orange);
  border-color: var(--orange);
  font-weight: 600;
}
.btn.nodrone:hover { opacity: 0.85; }

.action-row {
  display: flex;
  gap: 12px;
  margin-top: 10px;
  justify-content: center;
}

/* ── nodrone subtype panel ── */
.nodrone-panel {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--orange);
  border-radius: 8px;
  background: rgba(255,152,0,0.08);
}
.nodrone-panel label {
  font-size: 12px;
  color: var(--orange);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
  display: block;
}

/* ── source group headers in queue ── */
.source-header {
  padding: 8px 14px;
  background: rgba(255,255,255,0.03);
  border-bottom: 1px solid var(--surface2);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-dim);
  position: sticky;
  top: 0;
  z-index: 1;
}
.source-header .source-name {
  font-weight: 600;
  color: var(--text);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.source-header .bulk-btn {
  padding: 3px 8px;
  font-size: 11px;
  border-radius: 4px;
  border: 1px solid rgba(255,255,255,0.15);
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  white-space: nowrap;
}
.source-header .bulk-btn:hover { color: var(--text); border-color: var(--text); }
.source-header .bulk-btn.bulk-discard:hover { color: var(--red); border-color: var(--red); }
.source-header .bulk-btn.bulk-nodrone:hover { color: var(--orange); border-color: var(--orange); }

/* ── bulk nodrone subtype picker (dropdown under source header) ── */
.bulk-subtype-picker {
  padding: 6px 14px 10px;
  background: rgba(255,152,0,0.06);
  border-bottom: 1px solid var(--orange);
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}
.bulk-subtype-picker .bulk-sub-label {
  font-size: 11px;
  color: var(--orange);
  text-transform: uppercase;
  margin-right: 4px;
}
.bulk-subtype-picker .bulk-sub-btn {
  padding: 3px 10px;
  font-size: 11px;
  border-radius: 4px;
  border: 1px solid var(--orange);
  background: transparent;
  color: var(--orange);
  cursor: pointer;
}
.bulk-subtype-picker .bulk-sub-btn:hover { background: var(--orange); color: #fff; }

.hidden { display: none !important; }

/* ── keyboard hints ── */
.kbd-hints {
  margin-top: 24px;
  font-size: 11px;
  color: var(--text-dim);
  text-align: center;
  line-height: 1.8;
}
kbd {
  display: inline-block;
  padding: 1px 6px;
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 3px;
  font-family: monospace;
  font-size: 11px;
  background: rgba(255,255,255,0.06);
}

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
.toast.success { background: var(--green); }
.toast.error { background: var(--red); }

/* ── discover tab ── */
.discover-page {
  padding: 24px;
  max-width: 800px;
  margin: 0 auto;
  overflow-y: auto;
  height: calc(100vh - 42px);
}
.card {
  background: var(--surface);
  border: 1px solid var(--surface2);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 16px;
}
.card h3 {
  font-size: 14px;
  color: var(--accent);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.card .form-row {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.card input[type="text"],
.card input[type="number"] {
  padding: 7px 12px;
  border: 1px solid var(--surface2);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  outline: none;
}
.card input[type="text"]:focus,
.card input[type="number"]:focus {
  border-color: var(--accent);
}
.card input[type="text"] { flex: 1; min-width: 200px; }
.card input[type="number"] { width: 80px; }
.card .small-label {
  font-size: 12px;
  color: var(--text-dim);
}
.warning-box {
  padding: 8px 14px;
  border-radius: 6px;
  background: rgba(255,152,0,0.1);
  border: 1px solid var(--orange);
  color: var(--orange);
  font-size: 13px;
  margin-bottom: 12px;
}
.search-result {
  padding: 8px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.search-result:last-child { border-bottom: none; }
.search-result .sr-num {
  color: var(--text-dim);
  font-size: 13px;
  min-width: 24px;
}
.search-result a {
  color: var(--accent);
  text-decoration: none;
  font-size: 13px;
  word-break: break-all;
  flex: 1;
}
.search-result a:hover { text-decoration: underline; }
.search-result input[type="text"] {
  width: 160px;
  flex: unset;
  min-width: unset;
  font-size: 12px;
  padding: 4px 8px;
}
.search-result .btn { padding: 4px 12px; font-size: 12px; }
.log-area {
  background: var(--bg);
  border: 1px solid var(--surface2);
  border-radius: 6px;
  padding: 10px 14px;
  font-family: monospace;
  font-size: 13px;
  max-height: 300px;
  overflow-y: auto;
  white-space: pre-wrap;
  color: var(--text-dim);
  line-height: 1.6;
}
</style>
</head>
<body>

<!-- ── tab bar ── -->
<div class="tab-bar">
  <button class="active" id="tab-review-btn">Review</button>
  <button id="tab-discover-btn">Discover</button>
</div>

<!-- ════════════════ REVIEW TAB ════════════════ -->
<div id="tab-review">

<div class="stats-bar">
  <h1>Drone Audio Review</h1>
  <div class="stat">Queue: <b id="stat-queue">-</b></div>
  <div class="stat">Dataset: <b id="stat-dataset">-</b></div>
  <div class="stat" id="stat-classes"></div>
  <button class="btn discard" id="btn-empty-queue" style="padding:6px 14px;font-size:12px;">Empty Queue</button>
</div>

<div class="container">
  <!-- queue list -->
  <div class="queue-panel">
    <div class="queue-header">
      <span id="queue-count">0 segments</span>
      <span>newest first</span>
    </div>
    <div class="queue-list" id="queue-list"></div>
  </div>

  <!-- review panel -->
  <div class="review-panel empty" id="review-panel">
    <div class="empty-msg" id="empty-msg">Select a segment from the queue to begin</div>

    <div class="segment-info hidden" id="segment-info">
      <h2 id="seg-filename"></h2>
      <div class="details">
        <span>P(drone): <b id="seg-prob"></b></span> &nbsp;|&nbsp;
        <span>Hint: <b id="seg-hint"></b></span> &nbsp;|&nbsp;
        <span id="seg-source-wrap">Source: <a id="seg-source" target="_blank" rel="noopener"></a></span>
      </div>
    </div>

    <audio controls id="audio-player" class="hidden"></audio>

    <div class="controls hidden" id="controls">
      <div class="control-group">
        <label>Motor class</label>
        <div class="btn-row" id="motor-btns">
          <button class="btn motor-btn" data-motor="2_motors">2 motors</button>
          <button class="btn motor-btn" data-motor="4_motors">4 motors</button>
          <button class="btn motor-btn" data-motor="6_motors">6 motors</button>
          <button class="btn motor-btn" data-motor="8_motors">8 motors</button>
        </div>
      </div>

      <div class="control-group">
        <label>Quality (1 = poor, 5 = excellent)</label>
        <div class="btn-row" id="quality-btns">
          <button class="btn quality-btn" data-quality="1">1</button>
          <button class="btn quality-btn" data-quality="2">2</button>
          <button class="btn quality-btn" data-quality="3">3</button>
          <button class="btn quality-btn" data-quality="4">4</button>
          <button class="btn quality-btn" data-quality="5">5</button>
        </div>
      </div>

      <div class="action-row">
        <button class="btn accept" id="btn-accept" disabled>Accept (Enter)</button>
        <button class="btn discard" id="btn-discard">Discard (D)</button>
        <button class="btn nodrone" id="btn-nodrone">Not a drone (N)</button>
      </div>

      <div class="nodrone-panel hidden" id="nodrone-panel">
        <label>Select subtype</label>
        <div class="btn-row" id="subtype-btns">
          <button class="btn subtype-btn" data-subtype="airplanes">Airplanes</button>
          <button class="btn subtype-btn" data-subtype="birds">Birds</button>
          <button class="btn subtype-btn" data-subtype="cars">Cars</button>
          <button class="btn subtype-btn" data-subtype="crowd">Crowd</button>
          <button class="btn subtype-btn" data-subtype="electronics">Electronics</button>
          <button class="btn subtype-btn" data-subtype="motors">Motors</button>
          <button class="btn subtype-btn" data-subtype="random">Random</button>
          <button class="btn subtype-btn" data-subtype="speech">Speech</button>
          <button class="btn subtype-btn" data-subtype="wind">Wind</button>
        </div>
      </div>

      <div class="kbd-hints">
        <kbd>&#8593;</kbd><kbd>&#8595;</kbd> navigate &nbsp;
        <kbd>1</kbd>-<kbd>5</kbd> quality &nbsp;
        <kbd>Enter</kbd> accept &nbsp;
        <kbd>D</kbd> discard &nbsp;
        <kbd>N</kbd> not-a-drone &nbsp;
        <kbd>Space</kbd> play/pause
      </div>
    </div>
  </div>
</div>

</div><!-- /tab-review -->

<!-- ════════════════ DISCOVER TAB ════════════════ -->
<div id="tab-discover" class="hidden">
<div class="discover-page">

  <div id="discover-warnings"></div>

  <!-- help card -->
  <div class="card">
    <h3>How it works</h3>
    <div style="font-size:13px;color:var(--text-dim);line-height:1.7">
      <b>1.</b> Search YouTube or paste a URL / local path below.<br>
      <b>2.</b> Inspect results (links open in new tab), then click <b>Process</b> on the ones you want.<br>
      <b>3.</b> The app downloads the audio, splits it into 2-second segments, and runs the
      Stage 1 classifier. Segments above the threshold are added to the review queue.<br>
      <b>4.</b> Switch to the <b>Review</b> tab to listen and label the new segments.<br><br>
      <span style="color:var(--text)">Time ranges</span> &mdash;
      Use <code>MM:SS-MM:SS</code> or <code>H:MM:SS-H:MM:SS</code> format. Separate multiple ranges with spaces.<br>
      <span style="color:var(--text)">Motor hint</span> &mdash;
      Pre-fills the motor class during review (2/4/6/8 motors, or Any).<br>
      <span style="color:var(--text)">Threshold</span> &mdash;
      Minimum P(drone) to keep a segment (default 0.3). Lower = more segments, higher = fewer but higher confidence.
    </div>
  </div>

  <!-- settings card -->
  <div class="card">
    <h3>Settings</h3>
    <div class="form-row">
      <span class="small-label">Motor hint:</span>
      <button class="btn disc-motor-btn selected" data-motor="">Any</button>
      <button class="btn disc-motor-btn" data-motor="2">2</button>
      <button class="btn disc-motor-btn" data-motor="4">4</button>
      <button class="btn disc-motor-btn" data-motor="6">6</button>
      <button class="btn disc-motor-btn" data-motor="8">8</button>
    </div>
    <div class="form-row">
      <span class="small-label">Threshold:</span>
      <input type="number" id="disc-threshold" value="0.3" min="0" max="1" step="0.05">
    </div>
  </div>

  <!-- search youtube card -->
  <div class="card">
    <h3>Search YouTube</h3>
    <div class="form-row">
      <input type="text" id="disc-search-query" placeholder="e.g. fpv drone flight audio">
      <input type="number" id="disc-search-count" value="5" min="1" max="20" style="width:60px">
      <button class="btn" id="btn-search" style="white-space:nowrap">Search</button>
    </div>
  </div>

  <!-- process URL card -->
  <div class="card">
    <h3>Process URL / Local Path</h3>
    <div class="form-row">
      <input type="text" id="disc-url" placeholder="https://youtube.com/watch?v=... or /path/to/file.wav">
    </div>
    <div class="form-row">
      <span class="small-label">Time ranges (optional):</span>
      <input type="text" id="disc-url-ranges" placeholder="2:15-3:02 14:48-17:50" style="flex:1">
      <button class="btn" id="btn-process-url" style="white-space:nowrap">Process</button>
    </div>
  </div>

  <!-- search results card (hidden initially) -->
  <div class="card hidden" id="search-results-card">
    <h3>Search Results</h3>
    <div id="search-results-list"></div>
  </div>

  <!-- activity log card (hidden initially) -->
  <div class="card hidden" id="activity-card">
    <h3>Activity Log</h3>
    <div class="log-area" id="activity-log"></div>
  </div>

</div>
</div><!-- /tab-discover -->

<div class="toast" id="toast"></div>

<script>
(function() {
  // ──────────────────── tab switching ──────────────────────
  let activeTab = 'review';
  const tabReviewBtn = document.getElementById('tab-review-btn');
  const tabDiscoverBtn = document.getElementById('tab-discover-btn');
  const tabReview = document.getElementById('tab-review');
  const tabDiscover = document.getElementById('tab-discover');

  function switchTab(tab) {
    activeTab = tab;
    tabReviewBtn.classList.toggle('active', tab === 'review');
    tabDiscoverBtn.classList.toggle('active', tab === 'discover');
    tabReview.classList.toggle('hidden', tab !== 'review');
    tabDiscover.classList.toggle('hidden', tab !== 'discover');
    if (tab === 'review') loadQueue();
    if (tab === 'discover') loadDiscoverInfo();
  }

  tabReviewBtn.addEventListener('click', () => switchTab('review'));
  tabDiscoverBtn.addEventListener('click', () => switchTab('discover'));

  // ──────────────────── shared helpers ──────────────────────
  async function fetchJSON(url, opts) {
    const resp = await fetch(url, opts);
    return resp.json();
  }

  function toast(msg, type) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = 'toast show ' + type;
    setTimeout(() => el.className = 'toast', 2000);
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ──────────────────── REVIEW TAB ──────────────────────
  let queue = [];
  let selectedIdx = -1;
  let selectedMotor = null;
  let selectedQuality = null;

  const queueListEl = document.getElementById('queue-list');
  const reviewPanel = document.getElementById('review-panel');
  const emptyMsg = document.getElementById('empty-msg');
  const segInfo = document.getElementById('segment-info');
  const audioPlayer = document.getElementById('audio-player');
  const controlsEl = document.getElementById('controls');
  const btnAccept = document.getElementById('btn-accept');
  const btnDiscard = document.getElementById('btn-discard');
  const btnNodrone = document.getElementById('btn-nodrone');
  const nodronePanel = document.getElementById('nodrone-panel');

  async function loadQueue() {
    queue = await fetchJSON('/api/queue');
    renderQueue();
    loadStats();
  }

  async function loadStats() {
    const s = await fetchJSON('/api/stats');
    document.getElementById('stat-queue').textContent = s.queue_size;
    document.getElementById('stat-dataset').textContent = s.dataset_total;
    const classEl = document.getElementById('stat-classes');
    const parts = Object.entries(s.class_counts).map(([k, v]) => k + ': ' + v);
    classEl.innerHTML = parts.length ? parts.join(' &nbsp;|&nbsp; ') : '';
  }

  const SUBTYPES = ['airplanes','birds','cars','crowd','electronics','motors','random','speech','wind'];

  function renderQueue() {
    document.getElementById('queue-count').textContent = queue.length + ' segments';
    queueListEl.innerHTML = '';

    const groups = [];
    const seen = new Set();
    queue.forEach((entry, idx) => {
      if (!seen.has(entry.source_id)) {
        seen.add(entry.source_id);
        groups.push({ source_id: entry.source_id, items: [] });
      }
      groups.find(g => g.source_id === entry.source_id).items.push({ entry, idx });
    });

    groups.forEach(group => {
      const header = document.createElement('div');
      header.className = 'source-header';
      header.innerHTML =
        '<span class="source-name">' + esc(group.source_id) + ' (' + group.items.length + ')</span>' +
        '<button class="bulk-btn bulk-nodrone">All not-drone</button>' +
        '<button class="bulk-btn bulk-discard">Discard all</button>';

      const btnBulkDiscard = header.querySelector('.bulk-discard');
      const btnBulkNodrone = header.querySelector('.bulk-nodrone');

      btnBulkDiscard.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!confirm('Discard ALL ' + group.items.length + ' segments from ' + group.source_id + '?')) return;
        doBulkDiscard(group.source_id);
      });

      btnBulkNodrone.addEventListener('click', (e) => {
        e.stopPropagation();
        const existing = header.nextElementSibling;
        if (existing && existing.classList.contains('bulk-subtype-picker')) {
          existing.remove();
          return;
        }
        const picker = document.createElement('div');
        picker.className = 'bulk-subtype-picker';
        picker.innerHTML = '<span class="bulk-sub-label">Subtype:</span>' +
          SUBTYPES.map(s => '<button class="bulk-sub-btn" data-subtype="' + s + '">' + s + '</button>').join('');
        picker.querySelectorAll('.bulk-sub-btn').forEach(btn => {
          btn.addEventListener('click', () => {
            doBulkNodrone(group.source_id, btn.dataset.subtype, group.items.length);
            picker.remove();
          });
        });
        header.after(picker);
      });

      queueListEl.appendChild(header);

      group.items.forEach(({ entry, idx }) => {
        const div = document.createElement('div');
        div.className = 'queue-item' + (idx === selectedIdx ? ' active' : '');
        div.innerHTML =
          '<div class="filename">' + esc(entry.filename) + '</div>' +
          '<div class="meta">' +
          '<span class="prob">P=' + entry.drone_prob.toFixed(3) + '</span>' +
          '<span>' + esc(entry.motor_hint || '-') + '</span>' +
          '</div>';
        div.addEventListener('click', () => selectSegment(idx));
        queueListEl.appendChild(div);
      });
    });

    if (queue.length === 0) {
      showEmpty('Queue is empty — all segments reviewed!');
    }
  }

  async function doBulkDiscard(sourceId) {
    await fetchJSON('/api/bulk_discard', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ source_id: sourceId }),
    });
    toast('Discarded all from ' + sourceId, 'error');
    afterAction();
  }

  async function doBulkNodrone(sourceId, subtype, count) {
    await fetchJSON('/api/bulk_nodrone', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ source_id: sourceId, subtype: subtype }),
    });
    toast(count + ' segments -> not_a_drone/' + subtype, 'success');
    afterAction();
  }

  function selectSegment(idx) {
    if (idx < 0 || idx >= queue.length) return;
    selectedIdx = idx;
    selectedMotor = null;
    selectedQuality = null;
    nodronePanel.classList.add('hidden');
    const entry = queue[idx];

    queueListEl.querySelectorAll('.queue-item').forEach((el, i) => {
      el.classList.toggle('active', i === idx);
    });
    const activeEl = queueListEl.querySelector('.active');
    if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });

    reviewPanel.classList.remove('empty');
    emptyMsg.classList.add('hidden');
    segInfo.classList.remove('hidden');
    audioPlayer.classList.remove('hidden');
    controlsEl.classList.remove('hidden');

    document.getElementById('seg-filename').textContent = entry.filename;
    document.getElementById('seg-prob').textContent = entry.drone_prob.toFixed(4);
    document.getElementById('seg-hint').textContent = entry.motor_hint || '-';

    const srcLink = document.getElementById('seg-source');
    const srcWrap = document.getElementById('seg-source-wrap');
    if (entry.source_type === 'youtube') {
      srcLink.href = entry.source_ref;
      srcLink.textContent = entry.source_id;
      srcWrap.classList.remove('hidden');
    } else {
      srcLink.href = '#';
      srcLink.textContent = entry.source_ref || entry.source_id;
      srcWrap.classList.remove('hidden');
    }

    if (entry.motor_hint) {
      selectedMotor = entry.motor_hint;
    }

    audioPlayer.src = '/api/audio/' + entry.source_id + '/' + entry.filename;
    audioPlayer.load();
    audioPlayer.play().catch(() => {});

    updateBtnStates();
  }

  function showEmpty(msg) {
    reviewPanel.classList.add('empty');
    emptyMsg.classList.remove('hidden');
    emptyMsg.textContent = msg;
    segInfo.classList.add('hidden');
    audioPlayer.classList.add('hidden');
    controlsEl.classList.add('hidden');
    selectedIdx = -1;
  }

  function updateBtnStates() {
    document.querySelectorAll('.motor-btn').forEach(b => {
      b.classList.toggle('selected', b.dataset.motor === selectedMotor);
    });
    document.querySelectorAll('.quality-btn').forEach(b => {
      b.classList.toggle('selected', b.dataset.quality === String(selectedQuality));
    });
    btnAccept.disabled = !(selectedMotor && selectedQuality);
  }

  document.querySelectorAll('.motor-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedMotor = btn.dataset.motor;
      updateBtnStates();
    });
  });

  document.querySelectorAll('.quality-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedQuality = parseInt(btn.dataset.quality);
      updateBtnStates();
    });
  });

  btnAccept.addEventListener('click', doAccept);
  async function doAccept() {
    if (selectedIdx < 0 || !selectedMotor || !selectedQuality) return;
    const entry = queue[selectedIdx];
    await fetchJSON('/api/accept', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        filename: entry.filename,
        source_id: entry.source_id,
        motor_label: selectedMotor,
        quality: selectedQuality,
      }),
    });
    toast('Accepted as ' + selectedMotor + ' q=' + selectedQuality, 'success');
    afterAction();
  }

  btnDiscard.addEventListener('click', doDiscard);
  async function doDiscard() {
    if (selectedIdx < 0) return;
    const entry = queue[selectedIdx];
    await fetchJSON('/api/discard', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        filename: entry.filename,
        source_id: entry.source_id,
      }),
    });
    toast('Discarded', 'error');
    afterAction();
  }

  btnNodrone.addEventListener('click', () => {
    nodronePanel.classList.toggle('hidden');
  });

  document.querySelectorAll('.subtype-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (selectedIdx < 0) return;
      const entry = queue[selectedIdx];
      await fetchJSON('/api/accept_nodrone', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          filename: entry.filename,
          source_id: entry.source_id,
          subtype: btn.dataset.subtype,
        }),
      });
      toast('Classified as not_a_drone/' + btn.dataset.subtype, 'success');
      afterAction();
    });
  });

  async function afterAction() {
    audioPlayer.pause();
    const prevIdx = selectedIdx;
    await loadQueue();
    if (queue.length > 0) {
      const nextIdx = Math.min(prevIdx, queue.length - 1);
      selectSegment(nextIdx);
    } else {
      showEmpty('Queue is empty — all segments reviewed!');
    }
  }

  // ── keyboard shortcuts (only when review tab active) ──
  document.addEventListener('keydown', (e) => {
    if (activeTab !== 'review') return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const key = e.key;

    if (key >= '1' && key <= '5' && selectedIdx >= 0) {
      e.preventDefault();
      selectedQuality = parseInt(key);
      updateBtnStates();
      return;
    }
    if (key === 'ArrowUp' && selectedIdx > 0) {
      e.preventDefault();
      selectSegment(selectedIdx - 1);
      return;
    }
    if (key === 'ArrowDown' && selectedIdx < queue.length - 1) {
      e.preventDefault();
      selectSegment(selectedIdx + 1);
      return;
    }
    if (key === 'Enter' && selectedIdx >= 0 && selectedMotor && selectedQuality) {
      e.preventDefault();
      doAccept();
      return;
    }
    if ((key === 'd' || key === 'D') && selectedIdx >= 0) {
      e.preventDefault();
      doDiscard();
      return;
    }
    if ((key === 'n' || key === 'N') && selectedIdx >= 0) {
      e.preventDefault();
      nodronePanel.classList.toggle('hidden');
      return;
    }
    if (key === ' ' && selectedIdx >= 0) {
      e.preventDefault();
      if (audioPlayer.paused) audioPlayer.play();
      else audioPlayer.pause();
      return;
    }
  });

  document.getElementById('btn-empty-queue').addEventListener('click', async () => {
    if (!confirm('Delete ALL segments in the queue? This cannot be undone.')) return;
    await fetchJSON('/api/empty_queue', { method: 'POST' });
    toast('Queue emptied', 'error');
    afterAction();
  });

  // ──────────────────── DISCOVER TAB ──────────────────────
  let discoverMotor = '';
  let currentJobId = null;
  let pollTimer = null;

  function getDiscoverSettings() {
    return {
      motor: discoverMotor || null,
      threshold: parseFloat(document.getElementById('disc-threshold').value) || 0.3,
    };
  }

  // motor hint buttons
  document.querySelectorAll('.disc-motor-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      discoverMotor = btn.dataset.motor;
      document.querySelectorAll('.disc-motor-btn').forEach(b =>
        b.classList.toggle('selected', b === btn));
    });
  });

  // load discover info (warnings)
  async function loadDiscoverInfo() {
    const info = await fetchJSON('/api/discover_info');
    const box = document.getElementById('discover-warnings');
    let html = '';
    if (!info.ytdlp_available) {
      html += '<div class="warning-box">yt-dlp not installed. YouTube search and download will not work.</div>';
    }
    if (!info.model_available) {
      html += '<div class="warning-box">Model checkpoint not found. Processing will not work.</div>';
    }
    box.innerHTML = html;
  }

  // ── search ──
  const btnSearch = document.getElementById('btn-search');
  btnSearch.addEventListener('click', doSearch);

  async function doSearch() {
    const query = document.getElementById('disc-search-query').value.trim();
    const count = parseInt(document.getElementById('disc-search-count').value) || 5;
    if (!query) { toast('Enter a search query', 'error'); return; }
    if (currentJobId) { toast('A job is already running', 'error'); return; }

    btnSearch.disabled = true;
    btnSearch.textContent = 'Searching...';
    try {
      const data = await fetchJSON('/api/search_youtube', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query, count }),
      });
      if (data.error) { toast(data.error, 'error'); return; }
      if (data.skipped_known > 0) {
        toast(data.skipped_known + ' already-processed videos filtered out', 'success');
      }
      showSearchResults(data.urls);
    } catch (e) {
      toast('Search failed: ' + e.message, 'error');
    } finally {
      btnSearch.disabled = false;
      btnSearch.textContent = 'Search';
    }
  }

  function showSearchResults(urls) {
    const card = document.getElementById('search-results-card');
    const list = document.getElementById('search-results-list');
    if (!urls.length) {
      card.classList.add('hidden');
      toast('No new videos found', 'error');
      return;
    }
    card.classList.remove('hidden');
    list.innerHTML = '';
    urls.forEach((url, i) => {
      const row = document.createElement('div');
      row.className = 'search-result';
      row.innerHTML =
        '<span class="sr-num">[' + (i + 1) + ']</span>' +
        '<a href="' + esc(url) + '" target="_blank" rel="noopener">' + esc(url) + '</a>' +
        '<input type="text" placeholder="MM:SS-MM:SS" class="sr-ranges">' +
        '<button class="btn sr-process">Process</button>' +
        '<button class="btn sr-skip" style="font-size:12px;padding:4px 10px;opacity:0.6">Skip</button>';
      row.querySelector('.sr-process').addEventListener('click', () => {
        const ranges = row.querySelector('.sr-ranges').value.trim();
        startProcessing(url, ranges);
      });
      row.querySelector('.sr-skip').addEventListener('click', function() {
        const skipped = row.classList.toggle('sr-skipped');
        row.style.opacity = skipped ? '0.3' : '1';
        row.querySelector('.sr-process').disabled = skipped;
        row.querySelector('.sr-ranges').disabled = skipped;
        this.textContent = skipped ? 'Unskip' : 'Skip';
      });
      list.appendChild(row);
    });
  }

  // ── process URL ──
  const btnProcessUrl = document.getElementById('btn-process-url');
  btnProcessUrl.addEventListener('click', () => {
    const url = document.getElementById('disc-url').value.trim();
    const ranges = document.getElementById('disc-url-ranges').value.trim();
    if (!url) { toast('Enter a URL or path', 'error'); return; }
    startProcessing(url, ranges);
  });

  function startProcessing(url, ranges) {
    if (currentJobId) { toast('A job is already running', 'error'); return; }
    const settings = getDiscoverSettings();
    setDiscoverBusy(true);

    fetchJSON('/api/process_url', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        url: url,
        ranges: ranges,
        motor: settings.motor,
        threshold: settings.threshold,
      }),
    }).then(data => {
      if (data.error) {
        toast(data.error, 'error');
        setDiscoverBusy(false);
        return;
      }
      currentJobId = data.job_id;
      showActivityLog('Started processing: ' + url + '\n');
      pollTimer = setInterval(pollJob, 500);
    }).catch(e => {
      toast('Failed: ' + e.message, 'error');
      setDiscoverBusy(false);
    });
  }

  function showActivityLog(initialText) {
    const card = document.getElementById('activity-card');
    card.classList.remove('hidden');
    const log = document.getElementById('activity-log');
    log.textContent = initialText || '';
  }

  let lastLogCount = 0;

  async function pollJob() {
    if (!currentJobId) return;
    try {
      const data = await fetchJSON('/api/job_status/' + currentJobId);
      const logEl = document.getElementById('activity-log');
      // append only new log lines
      if (data.logs && data.logs.length > lastLogCount) {
        for (let i = lastLogCount; i < data.logs.length; i++) {
          logEl.textContent += data.logs[i] + '\n';
        }
        lastLogCount = data.logs.length;
        logEl.scrollTop = logEl.scrollHeight;
      }
      if (data.status !== 'running') {
        clearInterval(pollTimer);
        pollTimer = null;
        currentJobId = null;
        lastLogCount = 0;
        setDiscoverBusy(false);
        if (data.status === 'done' && data.result) {
          toast('Done: ' + data.result.queued + '/' + data.result.total + ' segments queued', 'success');
        } else if (data.status === 'error') {
          toast('Processing failed — see log', 'error');
        }
      }
    } catch (e) {
      // network error, keep polling
    }
  }

  function setDiscoverBusy(busy) {
    btnSearch.disabled = busy;
    btnProcessUrl.disabled = busy;
    document.querySelectorAll('.sr-process').forEach(b => b.disabled = busy);
  }

  // ── init ──
  loadQueue();
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
    parser = argparse.ArgumentParser(description="Drone audio review web app")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Starting review app at http://{args.host}:{args.port}")
    print(f"Queue: {QUEUE_META}")
    print(f"Dataset: {DATASET_AUDIO}")
    app.run(host=args.host, port=args.port, debug=True)
