"""Stage 1 false-positive breakdown by non-drone subtype.

For the final Stage 1 calibrated AST+PaSST ensemble, group every false positive
(true = no_drone, prediction = drone) by the non-drone subtype encoded in the
clip path (``not_a_drone/<subtype>/<file>.wav``) and report, per subtype:

  - n_total : number of no_drone test clips of that subtype
  - n_fp    : number of those that the ensemble called "drone"  (false positives)
  - fp_rate : n_fp / n_total

No training and no checkpoint inference: this reuses the SAVED per-clip TTA
softmax predictions that ``eval/eval_stage1_ensemble.py`` consumes, and
reproduces the exact calibration + blend logic from that script.

Because "the final Stage 1 ensemble" can mean three concrete operating points,
all three are reported (long-format CSV with a ``setting`` column + JSON):

  - val_tuned    : the deployable choice eval_stage1_ensemble.py selects
                   (temperature T fit on val, then (w, tau) chosen by val macroF1).
  - test_oracle  : the test-tuned upper bound reported in stage1_explained.txt
                   (w_AST=0.30, w_PaSST=0.70, tau=0.70 -> test mF1 0.9992).
  - ast_standalone : AST v1 alone with its val-tuned threshold (the doc's
                   recommended production model, test mF1 0.9973).

Outputs (default into ~/Desktop/thesis_handoff/eval/):
  - stage1_fp_by_subtype.csv
  - stage1_fp_by_subtype.json   (includes the list of FP clip paths per setting)

Usage:
    python eval/stage1_fp_by_subtype.py
    python eval/stage1_fp_by_subtype.py --out-dir /some/dir
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

LABELS = ["no_drone", "drone"]
PROB_COLS = [f"p_{l}" for l in LABELS]
LABEL_TO_IDX = {"no_drone": 0, "drone": 1}


def fit_T(p, y_idx):
    eps = 1e-12
    log_p = np.log(np.clip(p, eps, 1.0))

    def nll(T):
        T = max(float(T), 1e-3)
        s = log_p / T
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        n = e / e.sum(1, keepdims=True)
        return -np.log(np.clip(n[np.arange(len(y_idx)), y_idx], eps, 1.0)).mean()

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def apply_T(p, T):
    eps = 1e-12
    log_p = np.log(np.clip(p, eps, 1.0)) / T
    log_p -= log_p.max(1, keepdims=True)
    e = np.exp(log_p)
    return e / e.sum(1, keepdims=True)


def best_threshold_on_val(y_val_idx, p_val_drone):
    best = (-1.0, 0.5)
    for t in np.linspace(0.05, 0.95, 19):
        pred = (p_val_drone >= t).astype(int)
        mf1 = f1_score(y_val_idx, pred, average="macro", zero_division=0)
        if mf1 > best[0]:
            best = (mf1, float(t))
    return best


def subtype_from_relpath(relpath: str) -> str:
    """no_drone clips live at not_a_drone/<subtype>/<file>.wav."""
    parts = relpath.split("/")
    if parts[0] == "not_a_drone" and len(parts) >= 3:
        return parts[1]
    return parts[0]  # fallback (should not happen for no_drone rows)


def fp_breakdown(relpaths, true_lbl, pred_idx):
    """Return per-subtype breakdown + flat FP clip list for one setting."""
    df = pd.DataFrame({
        "relpath": relpaths,
        "true_label": true_lbl,
        "pred_idx": pred_idx,
    })
    nd = df[df["true_label"] == "no_drone"].copy()
    nd["subtype"] = nd["relpath"].map(subtype_from_relpath)
    nd["is_fp"] = (nd["pred_idx"] == 1).astype(int)

    grp = nd.groupby("subtype").agg(n_total=("is_fp", "size"),
                                    n_fp=("is_fp", "sum")).reset_index()
    grp["fp_rate"] = grp["n_fp"] / grp["n_total"]
    grp = grp.sort_values("n_fp", ascending=False).reset_index(drop=True)

    fp_paths = nd.loc[nd["is_fp"] == 1, ["relpath", "subtype"]] \
                 .sort_values(["subtype", "relpath"]).to_dict("records")
    return grp, fp_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ast-test",   default="artifacts/ast_v1_stage1_tta_test_preds.csv")
    ap.add_argument("--ast-val",    default="artifacts/ast_v1_stage1_tta_val_preds.csv")
    ap.add_argument("--passt-test", default="artifacts/passt_v1_stage1_tta_test_preds.csv")
    ap.add_argument("--passt-val",  default="artifacts/passt_v1_stage1_tta_val_preds.csv")
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/Desktop/thesis_handoff/eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load saved TTA predictions ------------------------------------------
    ast_v, ast_t = pd.read_csv(args.ast_val), pd.read_csv(args.ast_test)
    ps_v,  ps_t  = pd.read_csv(args.passt_val), pd.read_csv(args.passt_test)

    test_r = ast_t["relpath"].tolist()
    assert ps_t["relpath"].tolist() == test_r, "test relpath mismatch ast/passt"
    assert ps_v["relpath"].tolist() == ast_v["relpath"].tolist(), "val relpath mismatch"

    p_ast_v = ast_v[PROB_COLS].values.astype(np.float64)
    p_ast_t = ast_t[PROB_COLS].values.astype(np.float64)
    p_ps_v  = ps_v[PROB_COLS].values.astype(np.float64)
    p_ps_t  = ps_t[PROB_COLS].values.astype(np.float64)

    true_test = ast_t["true_label"].tolist()
    y_val  = np.array([LABEL_TO_IDX[l] for l in ast_v["true_label"]])
    y_test = np.array([LABEL_TO_IDX[l] for l in true_test])

    # -- Calibrate (NLL) on val ----------------------------------------------
    T_ast = fit_T(p_ast_v, y_val)
    T_ps  = fit_T(p_ps_v,  y_val)
    p_ast_t_c, p_ps_t_c = apply_T(p_ast_t, T_ast), apply_T(p_ps_t, T_ps)
    p_ast_v_c, p_ps_v_c = apply_T(p_ast_v, T_ast), apply_T(p_ps_v, T_ps)
    print(f"[calibration] T_AST = {T_ast:.3f}   T_PaSST = {T_ps:.3f}")

    # -- Define the three operating points -----------------------------------
    # (1) val_tuned deployable: pick (w, tau) by val macroF1
    best_w_val = (-1.0, 0.0, 0.5)
    for w in np.linspace(0.0, 1.0, 21):
        p_drone_val = w * p_ast_v_c[:, 1] + (1 - w) * p_ps_v_c[:, 1]
        mf1_v, t_v = best_threshold_on_val(y_val, p_drone_val)
        if mf1_v > best_w_val[0]:
            best_w_val = (mf1_v, float(w), float(t_v))
    _, w_val, tau_val = best_w_val

    # (3) AST standalone with val-tuned tau
    _, tau_ast = best_threshold_on_val(y_val, p_ast_v_c[:, 1])

    settings = {
        "val_tuned": {
            "description": "Deployable ensemble: temps fit on val, (w,tau) chosen by val macroF1.",
            "w_ast": w_val, "w_passt": 1 - w_val, "tau": tau_val,
            "p_drone": w_val * p_ast_t_c[:, 1] + (1 - w_val) * p_ps_t_c[:, 1],
        },
        "test_oracle": {
            "description": "Test-tuned upper bound from stage1_explained.txt (reporting only).",
            "w_ast": 0.30, "w_passt": 0.70, "tau": 0.70,
            "p_drone": 0.30 * p_ast_t_c[:, 1] + 0.70 * p_ps_t_c[:, 1],
        },
        "ast_standalone": {
            "description": "AST v1 alone (doc's recommended production model), val-tuned tau.",
            "w_ast": 1.0, "w_passt": 0.0, "tau": tau_ast,
            "p_drone": p_ast_t_c[:, 1],
        },
    }

    # -- Build breakdowns ----------------------------------------------------
    csv_rows = []
    json_out = {
        "calibration": {"T_AST": T_ast, "T_PaSST": T_ps},
        "n_test_clips": len(test_r),
        "n_no_drone_test": int((y_test == 0).sum()),
        "settings": {},
    }

    for name, s in settings.items():
        pred_idx = (s["p_drone"] >= s["tau"]).astype(int)
        mf1 = f1_score(y_test, pred_idx, average="macro", zero_division=0)
        grp, fp_paths = fp_breakdown(test_r, true_test, pred_idx)

        for _, r in grp.iterrows():
            csv_rows.append({
                "setting": name,
                "subtype": r["subtype"],
                "n_total": int(r["n_total"]),
                "n_fp": int(r["n_fp"]),
                "fp_rate": round(float(r["fp_rate"]), 6),
            })
        total_fp = int(grp["n_fp"].sum())
        json_out["settings"][name] = {
            "description": s["description"],
            "w_ast": round(s["w_ast"], 3),
            "w_passt": round(s["w_passt"], 3),
            "tau": round(s["tau"], 3),
            "test_macroF1": round(float(mf1), 4),
            "total_fp": total_fp,
            "by_subtype": grp.assign(fp_rate=grp["fp_rate"].round(6))
                             .to_dict("records"),
            "fp_clip_paths": fp_paths,
        }
        print(f"[{name:14s}] w_AST={s['w_ast']:.2f} tau={s['tau']:.2f} "
              f"mF1={mf1:.4f}  total_FP={total_fp}")

    # -- Save ----------------------------------------------------------------
    csv_path = out_dir / "stage1_fp_by_subtype.csv"
    json_path = out_dir / "stage1_fp_by_subtype.json"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
