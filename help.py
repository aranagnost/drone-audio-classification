#!/usr/bin/env python3
"""
help.py  —  Drone Audio Classification project reference

Run from the project root:
    python help.py
    python help.py --section data
    python help.py --section train
    python help.py --section eval
    python help.py --section apps
"""

import argparse
import textwrap

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
DIM    = "\033[2m"

SECTIONS = {
    "data": {
        "title": "Data Collection & Dataset Management",
        "entries": [
            {
                "script": "scripts/discover_drone_audio.py",
                "summary": "Main pipeline for finding and ingesting new drone audio.",
                "commands": [
                    ("search", 'python scripts/discover_drone_audio.py search "fpv drone" --count 5 --motor 4',
                     "Search YouTube, preview links, download & pre-classify with Stage 1 model."),
                    ("fetch",  "python scripts/discover_drone_audio.py fetch <URL|file> --motor 4 [2:15-3:02]",
                     "Download/load a single source with optional time range, segment & queue."),
                    ("review", "python scripts/discover_drone_audio.py review",
                     "CLI: listen to queued segments, assign quality 0–5, ingest into dataset."),
                    ("status", "python scripts/discover_drone_audio.py status",
                     "Print current queue size and class breakdown."),
                ],
            },
            {
                "script": "scripts/segment_audio.py",
                "summary": "Low-level audio segmenter (2 s clips, 1.5 s step). Interactive labelling.",
                "commands": [
                    ("basic",       "python scripts/segment_audio.py",
                     "Prompt for a file/URL and process the whole audio."),
                    ("time-range",  "python scripts/segment_audio.py <file> 1.20 3.45",
                     "Process only the range 1m20s → 3m45s from a local file."),
                ],
            },
            {
                "script": "scripts/add_to_queue.py",
                "summary": "Batch-import pre-cut 2 s WAV clips from add_queue/ into the review queue.",
                "commands": [
                    ("run", "python scripts/add_to_queue.py",
                     "Scans add_queue/{2,4,6,8}_motors/ and adds all clips to the review queue."),
                ],
            },
            {
                "script": "scripts/check_dataset_files.py",
                "summary": "Audit: compare metadata.json against actual files on disk.",
                "commands": [
                    ("run", "python scripts/check_dataset_files.py",
                     "Reports files in metadata but missing on disk, and vice-versa."),
                ],
            },
            {
                "script": "scripts/dataset_stats.py",
                "summary": "Print per-class segment counts and quality distribution.",
                "commands": [
                    ("run", "python scripts/dataset_stats.py",
                     "Prints drone summary (motor × quality) and no-drone subtype counts."),
                ],
            },
        ],
    },

    "train": {
        "title": "Model Training",
        "entries": [
            {
                "script": "ml/train/train_stage1.py",
                "summary": "Train Stage 1 binary classifier: drone vs no_drone.",
                "commands": [
                    ("basic",     "python -m ml.train.train_stage1",
                     "Train with defaults (15 epochs, lr=1e-3, batch=32)."),
                    ("balanced",  "python -m ml.train.train_stage1 --use_weighted_sampler",
                     "Use WeightedRandomSampler to handle class imbalance."),
                    ("custom",    "python -m ml.train.train_stage1 --epochs 30 --lr 3e-4 --out artifacts/checkpoints/my_stage1.pt",
                     "Custom epochs/lr/output path."),
                ],
            },
            {
                "script": "ml/train/train_stage2.py",
                "summary": "Train Stage 2 motor-count classifier: 2 / 4 / 6 / 8 motors (drone-only).",
                "commands": [
                    ("basic",        "python -m ml.train.train_stage2",
                     "Train with defaults (20 epochs, min_quality=3)."),
                    ("full",         "python -m ml.train.train_stage2 --use_weighted_sampler --use_quality_loss --min_quality 2",
                     "Balanced sampler + quality-weighted CE loss."),
                ],
            },
        ],
    },

    "eval": {
        "title": "Evaluation",
        "entries": [
            {
                "script": "ml/train/eval.py",
                "summary": "Evaluate Stage 1 + Stage 2 models together on a test split.",
                "commands": [
                    ("basic",  "python -m ml.train.eval",
                     "Use default checkpoint paths and ml/splits/test.csv."),
                    ("custom", "python -m ml.train.eval --stage1_ckpt artifacts/checkpoints/stage1_smallcnn.pt --stage2_ckpt artifacts/checkpoints/stage2_smallcnn.pt --test_csv ml/splits/test.csv",
                     "Explicit checkpoint and CSV paths."),
                ],
            },
            {
                "script": "ml/train/eval_stage1_only.py",
                "summary": "Evaluate only the Stage 1 binary model.",
                "commands": [
                    ("basic",  "python -m ml.train.eval_stage1_only",
                     "Default checkpoint + test CSV."),
                    ("custom", "python -m ml.train.eval_stage1_only --ckpt artifacts/checkpoints/stage1_smallcnn.pt --test_csv ml/splits/test.csv",
                     "Custom paths."),
                ],
            },
        ],
    },

    "apps": {
        "title": "Web Apps",
        "entries": [
            {
                "script": "scripts/review_app.py",
                "summary": "Flask app: review & label queued segments, discover new audio from YouTube.",
                "commands": [
                    ("default", "python scripts/review_app.py",
                     "Start on http://localhost:5000"),
                    ("custom",  "python scripts/review_app.py --port 8080",
                     "Start on a custom port."),
                ],
            },
            {
                "script": "scripts/inference_app.py",
                "summary": "Flask app: upload audio, pick a model, view per-segment classification probabilities.",
                "commands": [
                    ("default", "python scripts/inference_app.py",
                     "Start on http://localhost:5001"),
                    ("custom",  "python scripts/inference_app.py --port 8080",
                     "Start on a custom port."),
                ],
            },
        ],
    },
}


def print_entry(entry):
    print(f"  {BOLD}{CYAN}{entry['script']}{RESET}")
    wrapped = textwrap.fill(entry["summary"], width=72,
                            initial_indent="    ", subsequent_indent="    ")
    print(f"{DIM}{wrapped}{RESET}")
    for _name, cmd, desc in entry["commands"]:
        print(f"\n    {GREEN}${RESET} {cmd}")
        wrapped_desc = textwrap.fill(desc, width=70,
                                     initial_indent="      ",
                                     subsequent_indent="      ")
        print(f"{DIM}{wrapped_desc}{RESET}")
    print()


def print_section(key, section):
    print(f"\n{BOLD}{YELLOW}{'─' * 60}{RESET}")
    print(f"{BOLD}{YELLOW}  {section['title']}{RESET}")
    print(f"{BOLD}{YELLOW}{'─' * 60}{RESET}\n")
    for entry in section["entries"]:
        print_entry(entry)


def main():
    ap = argparse.ArgumentParser(
        description="Drone Audio Classification — project help",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--section", "-s",
        choices=list(SECTIONS.keys()),
        help="Show only one section: data | train | eval | apps",
    )
    args = ap.parse_args()

    print(f"\n{BOLD}Drone Audio Classification — available scripts & apps{RESET}")
    print(f"{DIM}Run all commands from the project root directory.{RESET}")

    if args.section:
        print_section(args.section, SECTIONS[args.section])
    else:
        for key, section in SECTIONS.items():
            print_section(key, section)

    print(f"{DIM}Tip: python help.py --section <data|train|eval|apps>{RESET}\n")


if __name__ == "__main__":
    main()
