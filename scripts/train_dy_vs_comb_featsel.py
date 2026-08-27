#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.core_train_binary_featsel import TrainConfig, set_seed, load_npy, train_binary_task, predict_prob
from utils.plots_binary import plot_confusion_binary, plot_prob_hists, plot_overlays
from utils.features import FEATURE_NAMES

# ── FEATURE SUBSET — EDIT HERE (or set DROP_FEATURES env var, comma-separated) ──
# Features to DROP from training. The remaining ones (original order) are trained
DROP_FEATURES = [
    "rec_track_pos_x_st1", "rec_track_neg_x_st1",           # station-1 x
    "rec_track_pos_px_st1", "rec_track_neg_px_st1",         # station-1 px
    "rec_mu_dpt", "rec_dimu_mT",                            # dpt + transverse mass
]
_env = os.environ.get("DROP_FEATURES")
if _env is not None:
    DROP_FEATURES = [s.strip() for s in _env.split(",") if s.strip()]

# POS = DY (untuned)
# NEG = raw combinatoric
# Defaults match the Itr16_12feat checkpoint's data; override via env vars to
# point at a different DY/Comb iteration without editing this file.
POS_FILE = os.environ.get("DY_FILE", "mc_raw_dy_compact_july_09_2026.npy")
NEG_FILE = os.environ.get("COMB_FILE", "tuned_comb_Itr16_12feat.npy")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data" / "ml_input")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "models")))


CFG = TrainConfig(
    epochs=int(os.environ.get("EPOCHS", "3")),
    lr=float(os.environ.get("LR", "5e-4")),
    batch_size=int(os.environ.get("BATCH_SIZE", "1024")),
    seed=int(os.environ.get("BOOT_SEED", os.environ.get("SPLIT_SEED", "42"))),
    standardize=bool(int(os.environ.get("STANDARDIZE", "0"))),
)

THR = float(os.environ.get("THRESHOLD", "0.90"))

def main():
    set_seed(CFG.seed)

    pos_label = "DY"
    neg_label = "COMB"

    # ── feature selection ─────────────────────────────────────────────────────
    unknown = [d for d in DROP_FEATURES if d not in FEATURE_NAMES]
    if unknown:
        raise SystemExit(f"[ERROR] DROP_FEATURES not in FEATURE_NAMES: {unknown}\n"
                         f"        valid: {FEATURE_NAMES}")
    keep_idx   = [i for i, n in enumerate(FEATURE_NAMES) if n not in DROP_FEATURES]
    kept_names = [FEATURE_NAMES[i] for i in keep_idx]
    run_name = os.environ.get("RUN_NAME", f"dy_comb_{len(kept_names)}feat")
    print(f"[INFO] data_dir={DATA_DIR}")
    print(f"[INFO] POS (DY)   file: {POS_FILE}")
    print(f"[INFO] NEG (COMB) file: {NEG_FILE}")
    print(f"[INFO] DROPPING {len(DROP_FEATURES)}: {DROP_FEATURES}")
    print(f"[INFO] TRAINING on {len(kept_names)}: {kept_names}")

    #run_dir = OUT_DIR / run_name
    run_dir = OUT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    X_pos = load_npy(DATA_DIR / POS_FILE)
    X_neg = load_npy(DATA_DIR / NEG_FILE)
    # slice to the kept columns BEFORE training / plotting
    X_pos = X_pos[:, keep_idx]
    X_neg = X_neg[:, keep_idx]

    plot_overlays(
        X_pos, X_neg, kept_names,
        pos_label=pos_label,
        neg_label=neg_label,
        run_name=run_name,
        out_dir=run_dir,
        bins=100,
        density=True,
        fontsize=18,
        show_stats=True,
        legend_all=False,
        save=True,
        show=False,
    )

    out = train_binary_task(X_pos, X_neg, CFG, run_dir, run_name, feature_names=kept_names)

    y_prob = predict_prob(out["model"], out["X_test"], CFG.device)

    np.savez_compressed(
        run_dir / f"{run_name}.test_bundle.npz",
        X_test=out["X_test"].astype(np.float32),
        y_test=np.asarray(out["y_test"]).astype(np.int64),
        y_prob=np.asarray(y_prob).astype(np.float32),
        pos_label=pos_label,
        neg_label=neg_label,
        run_name=run_name,
        threshold=float(THR),
    )

    plot_confusion_binary(
        out["y_test"], y_prob,
        threshold=THR,
        run_name=run_name,
        pos_label=pos_label,
        neg_label=neg_label,
        out_dir=run_dir,
        save=True,
        show=False,
    )

    plot_prob_hists(
        out["y_test"], y_prob,
        run_name=run_name,
        pos_label=pos_label,
        neg_label=neg_label,
        out_dir=run_dir,
        save=True,
        show=False,
    )


if __name__ == "__main__":
    main()

