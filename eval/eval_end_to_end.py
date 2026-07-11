"""
eval/eval_end_to_end.py

End-to-end pipeline evaluation: stage 1 (AST v1) followed by stage 2 (the
calibrated 4-way ensemble). Produces a 5-class joint confusion matrix and
macroF1 over a single test universe.

Universe (5808 clips):
  - All 5645 no_drone clips from data/splits/test.csv (only place they exist).
  - The 163 drone clips that appear in BOTH data/splits/test.csv AND
    data/splits_stage2/test.csv (the URL-disjoint stage 2 test split).
  - The 1369 stage-1-drone clips not in stage 2 test are excluded - running
    stage 2 on them would risk URL leakage with stage 2 training.

Stage 1: AST v1 standalone (TTA argmax) per artifacts/ast_v1_stage1_tta_test_preds.csv.
Stage 2: 4-way calibrated ensemble per stage2_explained.txt:
  - Per-model temperature scaling fit on val by NLL minimisation.
  - Stage A gate: avg(AST_v7_cal[2_motors], PaSST_cal[2_motors]) > 0.5 -> 2_motors.
  - Stage B blend (over {4_motors, 6_motors, 8_motors}):
       0.0*AST_v7 + 0.9*PaSST + 0.0*XGB_2s + 0.1*LGBM_10s   -> argmax.

Joint label space: {no_drone, 2_motors, 4_motors, 6_motors, 8_motors}.

Usage:
    python eval/eval_end_to_end.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

S2_LABELS = ["2_motors", "4_motors", "6_motors", "8_motors"]
STAGE_B_LABELS = ["4_motors", "6_motors", "8_motors"]
JOINT_LABELS = ["no_drone", "2_motors", "4_motors", "6_motors", "8_motors"]
S2_PROB_COLS = [f"p_{l}" for l in S2_LABELS]


def fit_temperature(probs, y_true_idx, name="model"):
    eps = 1e-12
    log_probs = np.log(np.clip(probs, eps, 1.0))

    def nll(T):
        T = max(float(T), 1e-3)
        scaled = log_probs / T
        scaled -= scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        norm = e / e.sum(axis=1, keepdims=True)
        return -np.log(np.clip(norm[np.arange(len(y_true_idx)), y_true_idx], eps, 1.0)).mean()

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    T = float(res.x)
    print(f"  T_{name:<10s} = {T:.3f}")
    return T


def apply_temperature(probs, T):
    eps = 1e-12
    log_probs = np.log(np.clip(probs, eps, 1.0)) / T
    log_probs -= log_probs.max(axis=1, keepdims=True)
    e = np.exp(log_probs)
    return e / e.sum(axis=1, keepdims=True)


def load_tree_probs(features_path, model_path, feat_names_path, split, relpaths):
    """Return (N, 3) probs over STAGE_B_LABELS aligned to relpaths."""
    df = pd.read_parquet(features_path)
    df = df[df["split"] == split]
    feat_names = Path(feat_names_path).read_text().strip().splitlines()
    df = df.set_index("relpath").loc[relpaths].reset_index()
    X = df[feat_names].values.astype(np.float32)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    model = joblib.load(model_path)
    probs = model.predict_proba(X)
    assert probs.shape == (len(relpaths), 3)
    return probs


def compute_stage2_ensemble(args):
    """Run the calibrated 4-way ensemble on stage 2 val + test.
    Returns DataFrame with columns: relpath, true_label, pred_label.
    """
    df_ast_val = pd.read_csv(args.ast_v7_val)
    df_ast_test = pd.read_csv(args.ast_v7_test)
    df_passt_val = pd.read_csv(args.passt_val)
    df_passt_test = pd.read_csv(args.passt_test)

    # Align AST and PaSST by relpath (they should already be aligned).
    val_relpaths = df_ast_val["relpath"].tolist()
    test_relpaths = df_ast_test["relpath"].tolist()
    df_passt_val = df_passt_val.set_index("relpath").loc[val_relpaths].reset_index()
    df_passt_test = df_passt_test.set_index("relpath").loc[test_relpaths].reset_index()

    ast_val_p = df_ast_val[S2_PROB_COLS].values.astype(np.float64)
    ast_test_p = df_ast_test[S2_PROB_COLS].values.astype(np.float64)
    passt_val_p = df_passt_val[S2_PROB_COLS].values.astype(np.float64)
    passt_test_p = df_passt_test[S2_PROB_COLS].values.astype(np.float64)

    # Truth
    df_feat = pd.read_parquet(args.features_2s)
    df_val = df_feat[df_feat["split"] == "val"].set_index("relpath")
    df_test = df_feat[df_feat["split"] == "test"].set_index("relpath")
    y_val = df_val.loc[val_relpaths, "motor_label"].values
    y_test = df_test.loc[test_relpaths, "motor_label"].values

    label_to_idx_4 = {l: i for i, l in enumerate(S2_LABELS)}
    label_to_idx_3 = {l: i for i, l in enumerate(STAGE_B_LABELS)}
    y_val_idx_4 = np.array([label_to_idx_4[l] for l in y_val])

    # Trees
    p2s_val = load_tree_probs(args.features_2s, args.xgb_2s, args.feat_names_2s, "val", val_relpaths)
    p2s_test = load_tree_probs(args.features_2s, args.xgb_2s, args.feat_names_2s, "test", test_relpaths)
    p10_val = load_tree_probs(args.features_10s, args.lgbm_10s, args.feat_names_10s, "val", val_relpaths)
    p10_test = load_tree_probs(args.features_10s, args.lgbm_10s, args.feat_names_10s, "test", test_relpaths)

    # Trees are (N, 3) over Stage B; calibrate only on val Stage B subset
    val_b_mask = y_val != "2_motors"
    y_val_idx_3_b = np.array([label_to_idx_3[l] for l in y_val[val_b_mask]])

    print("\n[stage 2 ensemble] Temperature calibration on val:")
    T_AST = fit_temperature(ast_val_p, y_val_idx_4, name="AST_v7")
    T_PaSST = fit_temperature(passt_val_p, y_val_idx_4, name="PaSST")
    T_XGB = fit_temperature(p2s_val[val_b_mask], y_val_idx_3_b, name="XGB_2s")
    T_LGBM = fit_temperature(p10_val[val_b_mask], y_val_idx_3_b, name="LGBM_10s")

    ast_test_cal = apply_temperature(ast_test_p, T_AST)
    passt_test_cal = apply_temperature(passt_test_p, T_PaSST)
    p2s_test_cal = apply_temperature(p2s_test, T_XGB)
    p10_test_cal = apply_temperature(p10_test, T_LGBM)

    # Stage A gate: avg(AST, PaSST) p_2_motors > 0.5 -> 2_motors
    p2_avg = 0.5 * (ast_test_cal[:, 0] + passt_test_cal[:, 0])
    gate = p2_avg > 0.5

    # Stage B blend: 0.0*AST + 0.9*PaSST + 0.0*XGB + 0.1*LGBM on {4, 6, 8}
    # AST and PaSST need their probs renormalised over {4, 6, 8}.
    def renorm_b(p4):
        r = p4[:, 1:]
        return r / np.clip(r.sum(axis=1, keepdims=True), 1e-12, None)

    ast_b = renorm_b(ast_test_cal)
    passt_b = renorm_b(passt_test_cal)

    blend = 0.0 * ast_b + 0.9 * passt_b + 0.0 * p2s_test_cal + 0.1 * p10_test_cal
    stage_b_pred = np.array([STAGE_B_LABELS[i] for i in blend.argmax(axis=1)])
    pred = np.where(gate, "2_motors", stage_b_pred)

    # Standalone stage 2 metric on full stage 2 test
    mf1 = f1_score(y_test, pred, labels=S2_LABELS, average="macro", zero_division=0)
    print(f"\n[stage 2 ensemble] Standalone macroF1 on stage 2 test (1594 drones): {mf1:.4f}")

    return pd.DataFrame({
        "relpath": test_relpaths,
        "true_label": y_test,
        "pred_label": pred,
    })


def print_5class_report(y_true, y_pred, title):
    sep = "=" * 64
    print(f"\n{sep}\n  {title}\n{sep}")
    mf1_all = f1_score(y_true, y_pred, labels=JOINT_LABELS, average="macro", zero_division=0)
    # Exclude classes with zero support from the macro to avoid the
    # undefined-F1-pulled-to-zero artifact on small drone subsets.
    supported = [l for l in JOINT_LABELS if (np.array(y_true) == l).sum() > 0]
    mf1_sup = f1_score(y_true, y_pred, labels=supported, average="macro", zero_division=0)
    print(f"  macroF1 (all 5 classes):                   {mf1_all:.4f}")
    if len(supported) < len(JOINT_LABELS):
        missing = sorted(set(JOINT_LABELS) - set(supported))
        print(f"  macroF1 (supported classes only, n={len(supported)}):  {mf1_sup:.4f}   "
              f"# excludes zero-support classes: {missing}")
    print()
    print(classification_report(y_true, y_pred, labels=JOINT_LABELS, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=JOINT_LABELS)
    print("  Confusion matrix (rows=true, cols=pred):")
    print("  " + " ".join(f"{n:>10}" for n in JOINT_LABELS))
    for i, row in enumerate(cm):
        print(f"  {JOINT_LABELS[i]:10s}" + " ".join(f"{v:>10}" for v in row))
    return mf1_all, mf1_sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-test", default="artifacts/ast_v1_stage1_tta_test_preds.csv")
    ap.add_argument("--ast-v7-val", default="artifacts/ast_v7_tta_val_preds.csv")
    ap.add_argument("--ast-v7-test", default="artifacts/ast_v7_tta_test_preds.csv")
    ap.add_argument("--passt-val", default="artifacts/passt_v1_tta_val_preds.csv")
    ap.add_argument("--passt-test", default="artifacts/passt_v1_tta_test_preds.csv")
    ap.add_argument("--features-2s", default="features/stage2_features.parquet")
    ap.add_argument("--xgb-2s", default="artifacts/xgb_stage2/best_model.joblib")
    ap.add_argument("--feat-names-2s", default="artifacts/xgb_stage2/feature_names.txt")
    ap.add_argument("--features-10s", default="features/stage2_features_10s.parquet")
    ap.add_argument("--lgbm-10s", default="artifacts/xgb_stage2_10s/best_model.joblib")
    ap.add_argument("--feat-names-10s", default="artifacts/xgb_stage2_10s/feature_names.txt")
    args = ap.parse_args()

    # --- Load stage 1 predictions on the stage 1 test set ------------------
    df_s1 = pd.read_csv(args.stage1_test)
    print(f"[stage 1] Loaded {len(df_s1)} predictions on stage 1 test set.")

    # Stage 1 standalone metric on the full 7177-clip universe (for context)
    s1_mf1 = f1_score(
        df_s1["true_label"].values, df_s1["pred_label"].values,
        labels=["no_drone", "drone"], average="macro", zero_division=0,
    )
    s1_cm = confusion_matrix(
        df_s1["true_label"].values, df_s1["pred_label"].values,
        labels=["no_drone", "drone"],
    )
    print(f"[stage 1] Standalone binary macroF1 on full stage 1 test (7177 clips): {s1_mf1:.4f}")
    print(f"[stage 1] Confusion matrix [no_drone, drone]: {s1_cm.tolist()}")

    # --- Compute stage 2 ensemble predictions on stage 2 test --------------
    df_s2 = compute_stage2_ensemble(args)

    # --- Build the joint universe (intersection-based) ----------------------
    s2_relpaths = set(df_s2["relpath"].tolist())
    s1_drones_in_s2 = df_s1[(df_s1["true_label"] == "drone") & (df_s1["relpath"].isin(s2_relpaths))]
    s1_no_drones = df_s1[df_s1["true_label"] == "no_drone"]
    n_drone_overlap = len(s1_drones_in_s2)
    n_drone_only_s1 = (df_s1["true_label"] == "drone").sum() - n_drone_overlap
    n_no_drone = len(s1_no_drones)
    print(f"\n[joint universe] no_drones from stage 1 test: {n_no_drone}")
    print(f"[joint universe] drones in BOTH stage 1 + stage 2 test: {n_drone_overlap}")
    print(f"[joint universe] drones in stage 1 test only (excluded): {n_drone_only_s1}")
    print(f"[joint universe] total clips: {n_no_drone + n_drone_overlap}")

    # Stage 2 prediction lookup (relpath -> (true_motor_label, pred_motor_label))
    s2_lookup = df_s2.set_index("relpath")[["true_label", "pred_label"]].to_dict("index")

    # --- Compose end-to-end predictions on the joint universe --------------
    rows = []
    # No_drone clips: final = stage 1's pred (no_drone if correct, drone if FP).
    # If stage 1 said drone, we need a stage 2 prediction. We don't have one
    # for these clips (they're not in stage 2 test). Assign a sentinel
    # FP_sentinel that we'll handle two ways: BEST CASE (assume stage 2
    # would have called it 2_motors, the most common) and WORST CASE
    # (count as 4_motors, the most populous remaining class).
    for _, r in s1_no_drones.iterrows():
        if r["pred_label"] == "no_drone":
            rows.append({"relpath": r["relpath"], "true": "no_drone", "pred": "no_drone"})
        else:
            # Stage 1 false positive. Stage 2 was never run on these clips.
            rows.append({"relpath": r["relpath"], "true": "no_drone", "pred": "FP_no_stage2"})

    # Drone clips in the overlap: final = stage 2's pred if stage 1 said drone,
    # else "no_drone" (stage 1 false negative - but stage 1 has zero of these
    # on test, so this branch should never trigger).
    for _, r in s1_drones_in_s2.iterrows():
        if r["pred_label"] == "no_drone":
            # Stage 1 false negative - drone gated out
            true_motor = s2_lookup[r["relpath"]]["true_label"]
            rows.append({"relpath": r["relpath"], "true": true_motor, "pred": "no_drone"})
        else:
            s2 = s2_lookup[r["relpath"]]
            rows.append({"relpath": r["relpath"], "true": s2["true_label"], "pred": s2["pred_label"]})

    df_joint = pd.DataFrame(rows)

    # Count FP_no_stage2 cases (stage 1 false positives where we don't have
    # a stage 2 prediction). For the published metric, distribute them across
    # motor classes uniformly (worst-case for macroF1 since it spreads errors
    # across classes), but also report the best-case lower bound (treat them
    # as no_drone - i.e. assume stage 2 would have rejected them).
    n_fp = (df_joint["pred"] == "FP_no_stage2").sum()
    print(f"\n[end-to-end] Stage 1 false positives needing stage 2 prediction: {n_fp}")
    print(f"             (these clips are not in stage 2 test -> no stage 2 prediction)")

    if n_fp > 0:
        # Realistic bound: distribute the n_fp FPs uniformly across motor
        # classes. Each class gets n_fp/4 false positives in expectation.
        # We do this deterministically by round-robin assignment for
        # reproducibility.
        df_realistic = df_joint.copy()
        fp_idx = df_realistic.index[df_realistic["pred"] == "FP_no_stage2"].tolist()
        for i, idx in enumerate(fp_idx):
            df_realistic.at[idx, "pred"] = S2_LABELS[i % 4]

        # Best-case bound: assume stage 2 would have classified those FPs
        # somehow but we still pay the stage 1 false-positive cost. For an
        # upper bound, treat them as the single most populated motor class
        # (puts all errors in one class, helping macro average).
        df_best = df_joint.copy()
        df_best.loc[df_best["pred"] == "FP_no_stage2", "pred"] = "2_motors"
    else:
        df_realistic = df_joint
        df_best = df_joint

    # --- Reports ------------------------------------------------------------
    mf1_all_realistic, mf1_sup_realistic = print_5class_report(
        df_realistic["true"].values, df_realistic["pred"].values,
        title=f"END-TO-END (universe={len(df_joint)}; FPs distributed uniformly across motor classes)",
    )
    if n_fp > 0:
        mf1_all_best, mf1_sup_best = print_5class_report(
            df_best["true"].values, df_best["pred"].values,
            title=f"END-TO-END (FPs all -> 2_motors; macroF1 upper bound)",
        )

    s2_full_mf1 = f1_score(
        df_s2["true_label"].values, df_s2["pred_label"].values,
        labels=S2_LABELS, average="macro", zero_division=0,
    )

    print("\n---- Summary ----")
    print(f"  Stage 1 standalone (binary, 7177 clips):                       {s1_mf1:.4f}")
    print(f"  Stage 2 standalone (4-class, 1594 drones):                     {s2_full_mf1:.4f}")
    print(f"  End-to-end (5-class, {len(df_joint)} clips, uniform FPs, all 5):     {mf1_all_realistic:.4f}")
    print(f"  End-to-end (5-class, {len(df_joint)} clips, uniform FPs, supported): {mf1_sup_realistic:.4f}")
    if n_fp > 0:
        print(f"  End-to-end (5-class, {len(df_joint)} clips, FPs->2, all 5):          {mf1_all_best:.4f}")
        print(f"  End-to-end (5-class, {len(df_joint)} clips, FPs->2, supported):      {mf1_sup_best:.4f}")
    print(
        f"\n  Caveats:\n"
        f"  - 1369 drone clips from stage 1 test were excluded - they are not\n"
        f"    in stage 2's URL-disjoint test split, so running stage 2 on them\n"
        f"    would risk leakage with stage 2 training. Run a real inference\n"
        f"    pipeline (inference/pipeline.py - Step 2) for the full 7177.\n"
        f"  - The 163-drone overlap happens to contain ZERO true 6_motors\n"
        f"    clips, so F1(6_motors) is undefined and treated as 0 by the\n"
        f"    'all 5 classes' macroF1, dropping it ~0.18 vs the 'supported'\n"
        f"    macroF1. Use the per-class table + the standalone numbers as the\n"
        f"    operative reading; the joint number is best treated as a\n"
        f"    sanity check of error propagation, not the headline result."
    )


if __name__ == "__main__":
    main()
