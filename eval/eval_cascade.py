"""
Evaluate the cascaded pipeline on the test set.

Stage A — 2-motor gate:
    Clips classified as 2_motors exit here.
    Mode 1 (--use-oracle-2motor): simulate a perfect gate using ground truth.
    Mode 2 (--ast-preds FILE): use real AST predictions from a CSV file.
                               The CSV must have columns: relpath, pred_label
                               where pred_label is one of 2_motors/4_motors/6_motors/8_motors.

Stage B — 4/6/8 motor classifier:
    Clips that pass Stage A go here (XGBoost or LightGBM).

Reports:
    - 4-class macro F1, per-class F1, confusion matrix for the cascade
    - Side-by-side comparison with AST v6 standalone results
    - How many clips each stage handles

Usage:
    # Oracle mode (upper bound):
    python eval/eval_cascade.py \\
        --features  features/stage2_features.parquet \\
        --model     artifacts/xgb_stage2/best_model.joblib \\
        --feat_names artifacts/xgb_stage2/feature_names.txt \\
        --use-oracle-2motor

    # Real AST predictions mode:
    python eval/eval_cascade.py \\
        --features  features/stage2_features.parquet \\
        --model     artifacts/xgb_stage2/best_model.joblib \\
        --feat_names artifacts/xgb_stage2/feature_names.txt \\
        --ast-preds artifacts/ast_v6_test_preds.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# 4-class ordering (matches AST v6 convention)
ALL_LABELS    = ["2_motors", "4_motors", "6_motors", "8_motors"]
STAGE_B_LABELS = ["4_motors", "6_motors", "8_motors"]

# AST v6 standalone test results (from our experiments) — used for comparison
AST_V6_RESULTS = {
    "macroF1":    0.5446,
    "2_motors":   0.8661,
    "4_motors":   0.5538,
    "6_motors":   0.3674,
    "8_motors":   0.3911,
}

META_COLS = ["filepath", "relpath", "motor_label", "binary_label",
             "quality", "youtube_url", "split"]

LABEL_ENC_B = LabelEncoder().fit(STAGE_B_LABELS)   # 4→0, 6→1, 8→2
LABEL_TO_IDX = {l: i for i, l in enumerate(ALL_LABELS)}
IDX_TO_LABEL = {i: l for i, l in enumerate(ALL_LABELS)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def get_feature_cols(df: pd.DataFrame, feat_names_path: str | None) -> list[str]:
    if feat_names_path and Path(feat_names_path).exists():
        cols = Path(feat_names_path).read_text().strip().splitlines()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from DataFrame: {missing[:5]} ...")
        return cols
    drop = set(META_COLS) | {"_ok", "_error"}
    return [c for c in df.columns if c not in drop]


def print_cascade_results(y_true: np.ndarray, y_pred: np.ndarray):
    mf1 = f1_score(y_true, y_pred, average="macro", labels=list(range(4)))
    sep = "═" * 64

    print(f"\n{sep}")
    print("  CASCADE RESULTS (test set)")
    print(sep)
    print(f"  macroF1  : {mf1:.4f}")
    print()

    report = classification_report(
        y_true, y_pred,
        labels=list(range(4)),
        target_names=ALL_LABELS,
        digits=4,
        zero_division=0,
    )
    for line in report.splitlines():
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "  " + " ".join(f"{n:>12}" for n in ALL_LABELS)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {ALL_LABELS[i]:12s}" + " ".join(f"{v:>12}" for v in row))

    return mf1


def print_comparison(cascade_mf1: float):
    sep = "─" * 64
    print(f"\n{sep}")
    print("  COMPARISON: Cascade vs AST v6 standalone")
    print(sep)
    print(f"  {'Metric':<20} {'AST v6':>10} {'Cascade':>10}  {'Δ':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}  {'─'*8}")

    ast_mf1 = AST_V6_RESULTS["macroF1"]
    delta = cascade_mf1 - ast_mf1
    sign  = "+" if delta >= 0 else ""
    print(f"  {'macroF1':<20} {ast_mf1:>10.4f} {cascade_mf1:>10.4f}  {sign}{delta:>7.4f}")
    print(f"\n  (Per-class AST v6 reference)")
    for label in ALL_LABELS:
        print(f"    {label:<15s} F1={AST_V6_RESULTS.get(label, float('nan')):.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Stage A: 2-motor gate
# ---------------------------------------------------------------------------

def apply_oracle_gate(df_test: pd.DataFrame):
    """
    Oracle mode: clips whose true label is 2_motors are perfectly classified.
    All others go to Stage B.
    Returns (stage_a_mask, stage_a_predictions_4class).
    """
    is_2motor = df_test["motor_label"] == "2_motors"
    print(f"[Stage A — Oracle] 2_motors clips correctly gated: "
          f"{is_2motor.sum()} / {len(df_test)}")
    return is_2motor


def apply_ast_gate(df_test: pd.DataFrame, ast_preds_path: str):
    """
    Real AST mode: load AST predictions from CSV, mark clips predicted as 2_motors.
    Returns is_2motor mask (based on AST prediction, not ground truth).
    """
    preds = pd.read_csv(ast_preds_path)
    # Merge on relpath
    merged = df_test.merge(preds[["relpath", "pred_label"]], on="relpath", how="left")
    if merged["pred_label"].isna().any():
        n_missing = merged["pred_label"].isna().sum()
        print(f"[WARN] {n_missing} test clips not found in AST predictions file — "
              f"treating as non-2motor (sent to Stage B).")
        merged["pred_label"] = merged["pred_label"].fillna("4_motors")

    is_2motor_pred = merged["pred_label"] == "2_motors"
    true_2motor    = df_test["motor_label"] == "2_motors"

    n_correct = (is_2motor_pred & true_2motor).sum()
    n_false_pos = (is_2motor_pred & ~true_2motor).sum()
    n_false_neg = (~is_2motor_pred & true_2motor).sum()
    print(f"[Stage A — AST preds]")
    print(f"  Clips predicted as 2_motors: {is_2motor_pred.sum()}")
    print(f"  True positives (correct 2motor): {n_correct}")
    print(f"  False positives (non-2motor sent to exit): {n_false_pos}")
    print(f"  False negatives (2motor sent to Stage B):  {n_false_neg}")

    return is_2motor_pred, merged["pred_label"].values


# ---------------------------------------------------------------------------
# Stage B: 4/6/8 classifier
# ---------------------------------------------------------------------------

def apply_stage_b(df_stage_b: pd.DataFrame, model, feature_cols: list[str]) -> np.ndarray:
    X = df_stage_b[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    pred_3class = model.predict(X)                              # 0/1/2
    pred_labels = LABEL_ENC_B.inverse_transform(pred_3class)   # 4/6/8_motors
    return pred_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate cascade pipeline (Stage A + Stage B)")
    ap.add_argument("--features",   default="features/stage2_features.parquet")
    ap.add_argument("--model",      default="artifacts/xgb_stage2/best_model.joblib",
                    help="Stage B model (joblib file from train_xgb_stage2.py)")
    ap.add_argument("--feat_names", default="artifacts/xgb_stage2/feature_names.txt",
                    help="Feature names file saved by train_xgb_stage2.py")
    ap.add_argument("--use-oracle-2motor", action="store_true",
                    help="Simulate perfect Stage A using ground-truth 2_motors labels")
    ap.add_argument("--ast-preds", default=None,
                    help="CSV with real AST predictions (columns: relpath, pred_label)")
    ap.add_argument("--split", default="test",
                    help="Which split to evaluate on (default: test)")
    args = ap.parse_args()

    use_oracle = args.use_oracle_2motor
    use_ast    = args.ast_preds is not None

    if not use_oracle and not use_ast:
        ap.error("Specify either --use-oracle-2motor or --ast-preds FILE")

    # ── Load features ────────────────────────────────────────────────────────
    print(f"[INFO] Loading features: {args.features}")
    df = load_features(args.features)
    df_test = df[df["split"] == args.split].copy().reset_index(drop=True)
    print(f"[INFO] Test clips: {len(df_test)}")
    print(f"[INFO] True distribution:\n{df_test['motor_label'].value_counts().to_string()}")

    feature_cols = get_feature_cols(df, args.feat_names)

    # ── Load Stage B model ───────────────────────────────────────────────────
    print(f"\n[INFO] Loading Stage B model: {args.model}")
    model = joblib.load(args.model)

    # ── Stage A ──────────────────────────────────────────────────────────────
    stage_a_preds = np.full(len(df_test), fill_value="", dtype=object)

    if use_oracle:
        is_2motor = apply_oracle_gate(df_test)
        stage_a_preds[is_2motor] = "2_motors"
    else:
        is_2motor, ast_all_preds = apply_ast_gate(df_test, args.ast_preds)
        # For clips gated as 2_motors, the cascade prediction is 2_motors
        stage_a_preds[is_2motor] = "2_motors"

    # ── Stage B ──────────────────────────────────────────────────────────────
    stage_b_mask = ~is_2motor
    df_stage_b   = df_test[stage_b_mask].copy()

    print(f"\n[Stage B] Clips to classify (4/6/8 motors): {stage_b_mask.sum()}")
    if len(df_stage_b) > 0:
        stage_b_preds = apply_stage_b(df_stage_b, model, feature_cols)
        stage_a_preds[stage_b_mask] = stage_b_preds
    else:
        print("[WARN] No clips reached Stage B.")

    # ── Build 4-class integer arrays ─────────────────────────────────────────
    y_true = np.array([LABEL_TO_IDX[l] for l in df_test["motor_label"].values])
    y_pred = np.array([LABEL_TO_IDX.get(l, 0) for l in stage_a_preds])

    # ── Report ───────────────────────────────────────────────────────────────
    mode_tag = "oracle 2-motor gate" if use_oracle else f"AST preds ({args.ast_preds})"
    print(f"\n  Mode: {mode_tag}")

    cascade_mf1 = print_cascade_results(y_true, y_pred)
    print_comparison(cascade_mf1)

    # ── Per-stage breakdown ──────────────────────────────────────────────────
    print(f"\n  Stage breakdown:")
    print(f"    Stage A (2-motor exit) : {is_2motor.sum():>5} clips")
    print(f"    Stage B (4/6/8 classif): {stage_b_mask.sum():>5} clips")


if __name__ == "__main__":
    main()
