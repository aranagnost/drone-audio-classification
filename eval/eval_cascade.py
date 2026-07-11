#!/usr/bin/env python3
"""
4-way soft cascade - adds AST v6 as a fourth signal alongside v7.

    Stage A - 2-motor gate using one of the AST models. Default = v7 (stronger
      on 2_motors than v6). Use --gate v6 to flip.

    Stage B - 4/6/8-motor decision is a weighted blend of four classifiers:

        pred_probs(4,6,8) = w_v6 * AST_v6_rest
                          + w_v7 * AST_v7_rest
                          + w_2s * XGB_2s
                          + w_10s * LGBM_10s
        with w_v6 + w_v7 + w_2s + w_10s == 1.

    Each component lives in a different signal regime:
        - AST v6    : 2-second audio, HF pretrained AST fine-tuned with small context.
        - AST v7    : 10-second stitched audio, fine-tuned on the long receptive field.
        - XGB 2s    : 2-second handcrafted features + XGBoost.
        - LGBM 10s  : 10-second stitched handcrafted features + LightGBM.

    We sweep the threshold tau on the gate model's p(2_motors) and the full
    4-vertex weight simplex at a user-chosen step size.

Usage:
    python eval/eval_cascade_4way.py                      # defaults: gate=v7, step=0.1
    python eval/eval_cascade_4way.py --gate v6            # use v6 for Stage A gate
    python eval/eval_cascade_4way.py --weight-step 0.2    # coarser sweep (quicker)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

ALL_LABELS     = ["2_motors", "4_motors", "6_motors", "8_motors"]
STAGE_B_LABELS = ["4_motors", "6_motors", "8_motors"]
AST_PROB_COLS  = [f"p_{l}" for l in ALL_LABELS]


def load_ast_preds(path: str, relpaths: list[str]) -> np.ndarray:
    """Load AST CSV and return (N, 4) probs in ALL_LABELS order, aligned to relpaths."""
    df = pd.read_csv(path)
    df = df.set_index("relpath").loc[relpaths].reset_index()
    return df[AST_PROB_COLS].values.astype(np.float64)


def load_tree_probs(features_path, model_path, feat_names_path, relpaths):
    """Return (N, 3) probs in STAGE_B_LABELS order, aligned to relpaths."""
    df = pd.read_parquet(features_path)
    df = df[df["split"] == "test"].set_index("relpath").loc[relpaths].reset_index()
    feat_names = Path(feat_names_path).read_text().strip().splitlines()
    X = df[feat_names].values.astype(np.float32)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    model = joblib.load(model_path)
    probs = model.predict_proba(X)
    assert probs.shape == (len(relpaths), 3)
    return probs, type(model).__name__


def renormalize_rest(ast_probs: np.ndarray) -> np.ndarray:
    """Drop the 2_motors column and renormalize the remaining 3 probs to sum to 1."""
    rest = ast_probs[:, 1:]
    s    = rest.sum(axis=1, keepdims=True)
    return rest / np.clip(s, 1e-12, None)


def simplex_weights_4(step: float = 0.1) -> list[tuple[float, float, float, float]]:
    """All (w_v6, w_v7, w_2s, w_10s) >= 0 summing to 1 at the given step."""
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l = n - i - j - k
                out.append((round(i/n, 3), round(j/n, 3),
                            round(k/n, 3), round(l/n, 3)))
    return out


def print_report(title, y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, labels=ALL_LABELS,
                   average="macro", zero_division=0)
    sep = "=" * 64
    print(f"\n{sep}\n  {title}\n{sep}")
    print(f"  macroF1: {mf1:.4f}\n")
    print(classification_report(y_true, y_pred, labels=ALL_LABELS,
                                digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    print("  Confusion matrix (rows=true, cols=predicted):")
    print("  " + " ".join(f"{n:>10}" for n in ALL_LABELS))
    for i, row in enumerate(cm):
        print(f"  {ALL_LABELS[i]:10s}" + " ".join(f"{v:>10}" for v in row))
    return mf1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ast-v6",          default="artifacts/ast_v6_test_preds.csv")
    ap.add_argument("--ast-v7",          default="artifacts/ast_v7_test_preds.csv")
    ap.add_argument("--features-2s",     default="features/stage2_features.parquet")
    ap.add_argument("--xgb-2s",          default="artifacts/xgb_stage2/best_model.joblib")
    ap.add_argument("--feat-names-2s",   default="artifacts/xgb_stage2/feature_names.txt")
    ap.add_argument("--features-10s",    default="features/stage2_features_10s.parquet")
    ap.add_argument("--lgbm-10s",        default="artifacts/xgb_stage2_10s/best_model.joblib")
    ap.add_argument("--feat-names-10s",  default="artifacts/xgb_stage2_10s/feature_names.txt")
    ap.add_argument("--gate",            default="v7", choices=["v6", "v7"],
                    help="Which AST model's p(2_motors) to use for the Stage A gate.")
    ap.add_argument("--weight-step",     type=float, default=0.1)
    ap.add_argument("--top-k",           type=int,   default=20)
    args = ap.parse_args()

    # -- Anchor ordering on the gate AST's CSV (both ASTs should have same 1594 rows) --
    ast_gate_csv = args.ast_v7 if args.gate == "v7" else args.ast_v6
    df_gate = pd.read_csv(ast_gate_csv)
    relpaths = df_gate["relpath"].tolist()
    print(f"[INFO] {len(relpaths)} test clips (anchored on AST {args.gate})")

    # -- Load everything aligned to `relpaths` --------------------------------
    v6_probs = load_ast_preds(args.ast_v6, relpaths)
    v7_probs = load_ast_preds(args.ast_v7, relpaths)
    v6_rest  = renormalize_rest(v6_probs)
    v7_rest  = renormalize_rest(v7_probs)

    # Labels (from the 2s features parquet - any of the aligned sources works)
    df_feat_2s = pd.read_parquet(args.features_2s)
    df_feat_2s = df_feat_2s[df_feat_2s["split"] == "test"].set_index("relpath")
    y_true = df_feat_2s.loc[relpaths, "motor_label"].values

    print("[INFO] Computing 2s XGB probabilities...")
    probs_2s, name_2s = load_tree_probs(
        args.features_2s, args.xgb_2s, args.feat_names_2s, relpaths)
    print(f"         model={name_2s}")

    print("[INFO] Computing 10s LGBM probabilities...")
    probs_10s, name_10s = load_tree_probs(
        args.features_10s, args.lgbm_10s, args.feat_names_10s, relpaths)
    print(f"         model={name_10s}")

    # -- Gate probs (from chosen model) ---------------------------------------
    gate_probs   = v7_probs if args.gate == "v7" else v6_probs
    gate_argmax  = gate_probs.argmax(axis=1)
    gate_p2      = gate_probs[:, 0]

    # -- Sweep ----------------------------------------------------------------
    thresholds = ["argmax", 0.5, 0.6]
    quads = simplex_weights_4(args.weight_step)
    print(f"[INFO] Sweep size: {len(thresholds)} thresholds x {len(quads)} weight quadruplets "
          f"= {len(thresholds) * len(quads)} configs")

    results = []
    for tau in thresholds:
        gate = (gate_argmax == 0) if tau == "argmax" else (gate_p2 > float(tau))
        n_gated = int(gate.sum())
        for (w_v6, w_v7, w_2s, w_10s) in quads:
            blend = (w_v6 * v6_rest + w_v7 * v7_rest
                     + w_2s * probs_2s + w_10s * probs_10s)
            sb_idx = blend.argmax(axis=1)
            sb_lbl = np.array([STAGE_B_LABELS[i] for i in sb_idx])
            y_pred = np.where(gate, "2_motors", sb_lbl)
            mf1 = f1_score(y_true, y_pred, labels=ALL_LABELS,
                           average="macro", zero_division=0)
            results.append({
                "tau":     str(tau),
                "w_v6":    w_v6,
                "w_v7":    w_v7,
                "w_2s":    w_2s,
                "w_10s":   w_10s,
                "macroF1": mf1,
                "n_gated": n_gated,
            })

    df_res = pd.DataFrame(results).sort_values("macroF1", ascending=False).reset_index(drop=True)

    print(f"\nTop {args.top_k} configs by test macroF1 (gate={args.gate}):\n")
    print(f"  {'rank':>4}  {'tau':<8s}  {'w_v6':>5s}  {'w_v7':>5s}  {'w_2s':>5s}  "
          f"{'w_10s':>5s}  {'gated':>6s}  {'macroF1':>8s}")
    print("  " + "-" * 62)
    for i, row in df_res.head(args.top_k).iterrows():
        print(f"  {i+1:>4}  {row['tau']:<8s}  {row['w_v6']:>5.2f}  {row['w_v7']:>5.2f}  "
              f"{row['w_2s']:>5.2f}  {row['w_10s']:>5.2f}  "
              f"{row['n_gated']:>6d}  {row['macroF1']:>8.4f}")

    # -- Detailed report for winner ------------------------------------------
    best = df_res.iloc[0]
    tau = best["tau"]
    w_v6, w_v7, w_2s, w_10s = best["w_v6"], best["w_v7"], best["w_2s"], best["w_10s"]
    gate_best = (gate_argmax == 0) if tau == "argmax" else (gate_p2 > float(tau))
    blend_best = w_v6 * v6_rest + w_v7 * v7_rest + w_2s * probs_2s + w_10s * probs_10s
    y_pred = np.where(gate_best, "2_motors",
                      np.array([STAGE_B_LABELS[i] for i in blend_best.argmax(axis=1)]))

    print_report(
        f"BEST CONFIG  gate={args.gate}  tau={tau}  "
        f"w_v6={w_v6}  w_v7={w_v7}  w_2s={w_2s}  w_10s={w_10s}",
        y_true, y_pred,
    )

    # -- Summary vs all prior baselines --------------------------------------
    print("\n" + "-" * 64)
    print("  BASELINES")
    print("  AST v6 standalone                        : 0.5446")
    print("  AST v7 standalone                        : 0.6002")
    print("  Old cascade (argmax + 2s XGB)            : 0.6005")
    print("  2-way ensemble (v6)  tau=0.5, w=0.25       : 0.6154")
    print("  3-way ensemble (v6)  tau=0.5, 20/60/20     : 0.6232")
    print("  3-way ensemble (v7)  tau=0.5, 20/60/20     : 0.6510")
    print(f"  4-way ensemble best                      : {best['macroF1']:.4f}   "
          f"(delta vs 3-way w/v7: {best['macroF1'] - 0.6510:+.4f})")
    print("-" * 64)


if __name__ == "__main__":
    main()
