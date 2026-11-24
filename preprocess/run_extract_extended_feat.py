#!/usr/bin/env python3
"""
Extract the extended (27-column) feature set from the existing ROOT inputs in
the old spinquest-combinatoric-bkg area, reading them in place (no copying).

Usage (from anywhere, after `source preprocess/setup.sh`):
    python3 run_extract_extended_feat.py                 # all four classes
    python3 run_extract_extended_feat.py --only jpsi dy  # a subset
"""
from pathlib import Path
import argparse
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.extract_dimu_features import extract_features, FEATURE_NAMES

# Where the ROOT files already live
DATA_ROOT = Path("/project/ptgroup/Forhad/spinquest-combinatoric-bkg")

SOURCES = {
    "jpsi": DATA_ROOT / "data/raw_input/Jpsi_tuned_sep05_8d.root",
    "psip": DATA_ROOT / "data/raw_input/tuned_psip_multiclass_12feat_itr2_clip05.root",
    "dy":   DATA_ROOT / "data/raw_input/tuned_dy_8d_energy_pyopt_full_1m.root",
    "comb": DATA_ROOT / "data/raw_input/tuned_comb_deep12_8d_energy_1m.root",
}

OUT_DIR = REPO_ROOT / "data" / "ml_input_extended"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(SOURCES), help="Process only these classes")
    p.add_argument("--tree", default="tree")
    p.add_argument("--mass-min", type=float, default=1.5)
    p.add_argument("--mass-max", type=float, default=6.0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    names = args.only or list(SOURCES)
    missing = [str(SOURCES[n]) for n in names if not SOURCES[n].exists()]
    if missing:
        raise SystemExit("[ERROR] Missing input ROOT files:\n  " + "\n  ".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(FEATURE_NAMES)} features: {FEATURE_NAMES}")

    for i, name in enumerate(names, start=1):
        src = SOURCES[name]
        out = args.out_dir / f"{src.stem}_{len(FEATURE_NAMES)}feat.npy"
        print(f"\n[{i}/{len(names)}] {name}: {src}")
        X, _, meta = extract_features(
            src,
            tree_name=args.tree,
            output_path=None,
            mass_min=args.mass_min,
            mass_max=args.mass_max,
            verbose_every=50000,
        )
        np.save(out, X)
        print(f"Saved: {out}  shape={X.shape}  "
              f"(pass mass {meta['n_entries_pass_mass']}/{meta['n_entries_total']})")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
