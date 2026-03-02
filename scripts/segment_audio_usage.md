# segment_audio.py — Usage Guide

## What it does

Splits audio (local files or YouTube videos) into 2-second overlapping segments (0.5s overlap), normalizes them, and builds a `metadata.json` dataset. Designed for drone acoustic signature collection.

## Requirements

- Python 3.10+
- `pydub`, `ffmpeg`, `ffplay`
- `yt-dlp` (only needed for YouTube sources)

## Output Structure

```
datasets/Drone_Audio_Dataset/
├── metadata.json
└── audio/
    ├── 2_motors/
    ├── 4_motors/
    ├── 6_motors/
    ├── 8_motors/
    └── not_a_drone/
        ├── airplanes/
        ├── birds/
        ├── cars/
        ├── crowd/
        ├── electronics/
        ├── motors/
        ├── random/
        └── wind/
```

## Usage Modes

### 1. Interactive (no arguments)

```bash
python segment_audio.py
```

Prompts for source, motor count, and quality per segment.

### 2. Full file / full YouTube video

```bash
python segment_audio.py <source>
```

`<source>` is a local file path or YouTube URL. Processes the entire audio.

### 3. Time-sliced

```bash
python segment_audio.py <source> <start> <end>
```

Times use `minutes.seconds` format:

```bash
python segment_audio.py https://youtu.be/xyz 1.30 3.00
# Processes from 1m30s to 3m00s
```

```bash
python segment_audio.py recording.mp3 0.10 2.45
# Processes from 10s to 2m45s
```

## Workflow

1. **Source** → downloaded (YouTube) or converted to WAV (local).
2. **Motor count prompt** → enter `2`, `4`, `6`, `8` for drone audio, or leave empty for `no_drone`.
3. **Segmentation** → audio is split into 2s chunks with 0.5s overlap.
4. **Labeling**:
   - **If `no_drone` globally**: optionally set one subtype for all segments (skips per-segment review).
   - **If drone**: each segment plays, you rate quality `1–5`. Enter `0` to reclassify that segment as `no_drone` and pick a subtype.

## Subtype Codes (for no_drone segments)

| Code | Subtype |
|------|---------|
| `a`  | airplanes |
| `b`  | birds |
| `ca` | cars |
| `cr` | crowd |
| `e`  | electronics |
| `m`  | motors |
| `r`  | random |
| `w`  | wind |

## Metadata Entry Examples

**Drone segment:**
```json
{
  "filename": "xyz_4_motors_012.wav",
  "binary_label": "drone",
  "motor_label": "4_motors",
  "source": "youtube",
  "youtube_url": "https://youtu.be/xyz",
  "duration": 2.0,
  "quality": 4
}
```

**No-drone segment:**
```json
{
  "filename": "xyz_no_label_003.wav",
  "binary_label": "no_drone",
  "motor_label": null,
  "source": "youtube",
  "youtube_url": "https://youtu.be/xyz",
  "duration": 2.0,
  "subtype": "wind"
}
```

## Reprocessing

If you run the script on a YouTube URL that was already processed, it asks whether to delete old segments and metadata before reprocessing. This prevents duplicates.

## Notes

- Overlap detection prevents you from processing the same time range twice in one session.
- Segment numbering auto-increments from existing files in the output folder.
- `ffplay` is used for audio preview; if missing, labeling still works but you won't hear the segments.
