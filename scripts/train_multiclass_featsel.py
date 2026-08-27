#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.core_train_multiclass_featsel import (
    TrainConfig,
    set_seed,
    load_npy,
    train_multiclass_task,
    predict_proba,
    predict_class
)
from utils.plots_multiclass import plot_overlays_multiclass
from utils.features import FEATURE_NAMES

# ── FEATURE SUBSET — EDIT HERE (or set DROP_FEATURES env var, comma-separated) ──
# Features to DROP from training. The remaining ones (original order) are trained.
# Same 6 as the binary 12-feat DY-vs-comb model (dy_comb_raw_st1_dropped).
DROP_FEATURES = [
    "rec_track_pos_x_st1", "rec_track_neg_x_st1",           # station-1 x
    "rec_track_pos_px_st1", "rec_track_neg_px_st1",         # station-1 px
    "rec_mu_dpt", "rec_dimu_mT",                            # dpt + transverse mass
]
_env = os.environ.get("DROP_FEATURES")
if _env is not None:
    DROP_FEATURES = [s.strip() for s in _env.split(",") if s.strip()]

# Input files — overridable via env vars for different data versions
JPSI_FILE = os.environ.get("JPSI_FILE", "mc_jpsi_tuned1_july16.npy")
PSIP_FILE = os.environ.get("PSIP_FILE", "mc_psip_tuned1_july26.npy")
DY_FILE   = os.environ.get("DY_FILE",   "tuned_dy_8d_energy_pyopt_full_1m.npy")
COMB_FILE = os.environ.get("COMB_FILE", "tuned_comb_deep12_8d_energy_1m.npy")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data" / "ml_input_final")))
OUT_DIR  = Path(os.environ.get("OUT_DIR",  str(REPO_ROOT / "models")))
USE_CLASS_WEIGHTS = bool(int(os.environ.get("USE_CLASS_WEIGHTS", "1")))

CFG = TrainConfig(
    epochs          = int(os.environ.get("EPOCHS",           "300")),
    lr              = float(os.environ.get("LR",             "5e-4")),
    lr_min          = float(os.environ.get("LR_MIN",         "1e-6")),
    batch_size      = int(os.environ.get("BATCH_SIZE",       "1024")),
    seed            = int(os.environ.get("BOOT_SEED", os.environ.get("SPLIT_SEED", "42"))),
    standardize     = bool(int(os.environ.get("STANDARDIZE", "1"))),
    hidden_dim      = int(os.environ.get("HIDDEN_DIM",       "512")),
    num_layers      = int(os.environ.get("NUM_LAYERS",       "4")),
    dropout_rate    = float(os.environ.get("DROPOUT",        "0.3")),
    flat            = bool(int(os.environ.get("FLAT",        "0"))),
    loss_type       = os.environ.get("LOSS_TYPE",            "ce"),
    focal_gamma     = float(os.environ.get("FOCAL_GAMMA",    "2.0")),
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING","0.0")),
    model_type      = os.environ.get("MODEL_TYPE",           "dnn"),
    optimizer_type  = os.environ.get("OPTIMIZER",            "adam"),
    scheduler_type  = os.environ.get("SCHEDULER",            "cosine"),
)


def main():
    set_seed(CFG.seed)

    run_name    = os.environ.get("RUN_NAME", "multiclass_12feat_st1_dropped")
    class_names = ["J/psi", "psi(2S)", "DY", "Combinatoric"]

    # ── feature selection ────────────────────────────────────────────────────
    unknown = [d for d in DROP_FEATURES if d not in FEATURE_NAMES]
    if unknown:
        raise SystemExit(f"[ERROR] DROP_FEATURES not in FEATURE_NAMES: {unknown}\n"
                         f"        valid: {FEATURE_NAMES}")
    keep_idx   = [i for i, n in enumerate(FEATURE_NAMES) if n not in DROP_FEATURES]
    kept_names = [FEATURE_NAMES[i] for i in keep_idx]
    print(f"[INFO] DROPPING {len(DROP_FEATURES)}: {DROP_FEATURES}")
    print(f"[INFO] TRAINING on {len(kept_names)}: {kept_names}")

    run_dir = OUT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading data from {DATA_DIR}")
    X_jpsi = load_npy(DATA_DIR / JPSI_FILE)
    X_psip = load_npy(DATA_DIR / PSIP_FILE)
    X_dy   = load_npy(DATA_DIR / DY_FILE)
    X_comb = load_npy(DATA_DIR / COMB_FILE)

    print("[INFO] Data shapes (18-col input):")
    print(f"  J/psi       : {X_jpsi.shape}")
    print(f"  psi(2S)     : {X_psip.shape}")
    print(f"  DY          : {X_dy.shape}")
    print(f"  Combinatoric: {X_comb.shape}")

    # slice to the kept columns BEFORE training / plotting
    Xs = [X[:, keep_idx] for X in (X_jpsi, X_psip, X_dy, X_comb)]
    print(f"[INFO] After feature selection: {[X.shape for X in Xs]}")
    mode = "all events + class-weighted loss" if USE_CLASS_WEIGHTS else "balanced downsample to smallest class"
    print(f"[INFO] Training sample mode: {mode}")

    print("[INFO] Generating feature overlay plots ...")
    plot_overlays_multiclass(
        Xs=Xs,
        class_names=class_names,
        feature_names=kept_names,
        run_name=run_name,
        out_dir=run_dir,
        bins=100,
        density=True,
        fontsize=14,
        show_stats=True,
        legend_all=False,
        feature_ranges={"rec_dimu_M": (2.0, 6.0)},
        save=True,
        show=False,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[INFO] Training multiclass classifier ...")
    out = train_multiclass_task(
        Xs=Xs,
        cfg=CFG,
        out_dir=run_dir,
        run_name=run_name,
        class_names=class_names,
        use_class_weights=USE_CLASS_WEIGHTS,
        feature_names=kept_names,
    )

    # ── Predictions on test set ──────────────────────────────────────────────
    y_proba = predict_proba(out["model"], out["X_test"], CFG.device)
    y_pred  = predict_class(out["model"], out["X_test"], CFG.device)

    # ── Save test bundle ─────────────────────────────────────────────────────
    np.savez_compressed(
        run_dir / f"{run_name}.test_bundle.npz",
        X_test=out["X_test"].astype(np.float32),
        y_test=np.asarray(out["y_test"]).astype(np.int64),
        y_proba=np.asarray(y_proba).astype(np.float32),
        y_pred=np.asarray(y_pred).astype(np.int64),
        class_names=class_names,
        feature_names=kept_names,
        run_name=run_name,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Results saved to : {run_dir}")
    print(f"[INFO] Best checkpoint  : {out['summary']['best_ckpt']}")
    print(f"[INFO] Test accuracy    : {out['summary']['test_metrics']['acc']:.4f}")
    print(f"[INFO] Test macro F1    : {out['summary']['test_metrics']['macro_f1']:.4f}")

    print(f"\n[INFO] Test Confusion Matrix:")
    cm = np.array(out['summary']['test_metrics']['confusion_matrix'])
    print("Predicted ->")
    header = "True | " + " | ".join([f"{name:>12s}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = (f"{name:>4s} | " +
               " | ".join([f"{cm[i, j]:>12d}" for j in range(len(class_names))]))
        print(row)


if __name__ == "__main__":
    main()
