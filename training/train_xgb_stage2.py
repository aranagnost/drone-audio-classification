"""
Train XGBoost and LightGBM classifiers on handcrafted features for Stage B
(4 / 6 / 8 motor classification — 3-class problem).

Usage:
    python models/train_xgb_stage2.py \\
        --features features/stage2_features.parquet \\
        --out_dir  artifacts/xgb_stage2

Outputs (saved in --out_dir):
    best_model.joblib   — best model (XGB or LGBM, whichever wins on val macroF1)
    xgb_model.joblib    — XGBoost model
    lgbm_model.joblib   — LightGBM model
    feature_names.txt   — ordered list of feature names used
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

MOTOR_LABELS  = ["4_motors", "6_motors", "8_motors"]
LABEL_ENCODER = LabelEncoder().fit(MOTOR_LABELS)   # 4→0, 6→1, 8→2

META_COLS = ["filepath", "relpath", "motor_label", "binary_label",
             "quality", "youtube_url", "split"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = set(META_COLS) | {"_ok", "_error"}
    return [c for c in df.columns if c not in drop]


def prepare_split(df: pd.DataFrame, split: str, feature_cols: list[str]):
    sub = df[(df["split"] == split) & (df["motor_label"].isin(MOTOR_LABELS))].copy()
    X = sub[feature_cols].values.astype(np.float32)
    y = LABEL_ENCODER.transform(sub["motor_label"].values)
    return X, y, sub


def print_results(tag: str, y_true, y_pred, label_names: list[str]):
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'═'*60}")
    print(f"  {tag}")
    print(f"{'═'*60}")
    print(f"  macroF1 : {mf1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    header = "  " + "  ".join(f"{n:>10}" for n in label_names)
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    print(header)
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:10s}" + "  ".join(f"{v:>10}" for v in row))
    return mf1


def print_feature_importance(model, feature_cols: list[str], top_n: int = 20):
    try:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        print(f"\n  Top {top_n} features:")
        for rank, i in enumerate(idx[:top_n], 1):
            print(f"    {rank:>2}. {feature_cols[i]:<35s} {importances[i]:.4f}")
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val):
    try:
        import xgboost as xgb
    except ImportError:
        print("[WARN] xgboost not installed — skipping XGB.")
        return None

    print("\n[XGB] Training XGBoost ...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"[XGB] Best iteration: {model.best_iteration}")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[WARN] lightgbm not installed — skipping LGBM.")
        return None

    print("\n[LGBM] Training LightGBM ...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        objective="multiclass",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False),
                 lgb.log_evaluation(period=-1)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    print(f"[LGBM] Best iteration: {model.best_iteration_}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train XGB/LGBM Stage B classifier (4/6/8 motors)")
    ap.add_argument("--features", default="features/stage2_features.parquet",
                    help="Path to features parquet/csv from extract_features.py")
    ap.add_argument("--out_dir",  default="artifacts/xgb_stage2",
                    help="Directory to save models and metadata")
    ap.add_argument("--top_n_features", type=int, default=20,
                    help="How many top features to print")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading features: {args.features}")
    df = load_features(args.features)
    feature_cols = get_feature_cols(df)
    print(f"[INFO] Features: {len(feature_cols)}  Rows: {len(df)}")
    print(f"[INFO] Splits:  {df['split'].value_counts().to_dict()}")
    print(f"[INFO] Classes: {df['motor_label'].value_counts().to_dict()}")

    # ── Prepare splits (4/6/8 motors only) ──────────────────────────────────
    X_tr, y_tr, df_tr = prepare_split(df, "train", feature_cols)
    X_va, y_va, df_va = prepare_split(df, "val",   feature_cols)
    X_te, y_te, df_te = prepare_split(df, "test",  feature_cols)

    print(f"\n[INFO] 3-class split sizes (4/6/8 motors only):")
    print(f"       Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

    # Handle NaN/inf values that might have slipped through
    for arr in [X_tr, X_va, X_te]:
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    label_names = list(LABEL_ENCODER.classes_)   # ["4_motors", "6_motors", "8_motors"]

    # ── Train ────────────────────────────────────────────────────────────────
    xgb_model  = train_xgboost(X_tr, y_tr, X_va, y_va)
    lgbm_model = train_lightgbm(X_tr, y_tr, X_va, y_va)

    # ── Evaluate on val ──────────────────────────────────────────────────────
    val_scores: dict[str, float] = {}

    if xgb_model is not None:
        p_va = xgb_model.predict(X_va)
        val_scores["xgb"] = print_results("XGBoost — Val", y_va, p_va, label_names)
        print_feature_importance(xgb_model, feature_cols, args.top_n_features)

    if lgbm_model is not None:
        p_va = lgbm_model.predict(X_va)
        val_scores["lgbm"] = print_results("LightGBM — Val", y_va, p_va, label_names)
        print_feature_importance(lgbm_model, feature_cols, args.top_n_features)

    # ── Evaluate on test ─────────────────────────────────────────────────────
    if xgb_model is not None:
        p_te = xgb_model.predict(X_te)
        print_results("XGBoost — Test", y_te, p_te, label_names)

    if lgbm_model is not None:
        p_te = lgbm_model.predict(X_te)
        print_results("LightGBM — Test", y_te, p_te, label_names)

    # ── Save ─────────────────────────────────────────────────────────────────
    saved_paths: dict[str, Path] = {}

    if xgb_model is not None:
        p = out_dir / "xgb_model.joblib"
        joblib.dump(xgb_model, p)
        saved_paths["xgb"] = p
        print(f"\n[SAVE] XGB model -> {p}")

    if lgbm_model is not None:
        p = out_dir / "lgbm_model.joblib"
        joblib.dump(lgbm_model, p)
        saved_paths["lgbm"] = p
        print(f"[SAVE] LGBM model -> {p}")

    # Save feature names so eval_cascade.py can reconstruct the same columns
    feat_path = out_dir / "feature_names.txt"
    feat_path.write_text("\n".join(feature_cols))
    print(f"[SAVE] Feature names -> {feat_path}")

    # Pick best model by val macroF1
    if val_scores:
        best_name = max(val_scores, key=val_scores.get)
        best_model = xgb_model if best_name == "xgb" else lgbm_model
        best_path = out_dir / "best_model.joblib"
        joblib.dump(best_model, best_path)
        print(f"\n[BEST] {best_name.upper()} wins val macroF1={val_scores[best_name]:.4f} -> {best_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
