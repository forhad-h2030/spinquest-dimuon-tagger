# Training SLURM scripts

Production training jobs only. Sweep/variant scripts stay on the HPC copy
(`dgy5cd@login.hpc.virginia.edu:/project/ptgroup/Forhad/spinquest-combinatoric-bkg`).

| script | driver | feature toggle | depends on |
|---|---|---|---|
| `multi_class_final_12feat.sh` | `scripts/train_multiclass_featsel.py` | `DROP_FEATURES` (unset=12, `=` empty=18) | `utils/core_train_multiclass_featsel.py`, `utils/plots_multiclass.py`, `utils/features.py` |
| `submit_train_dy_vs_comb.sh` | `scripts/train_dy_vs_comb_featsel.py` | `FEATURE_SET=12\|18` | `utils/core_train_binary_featsel.py`, `utils/plots_binary.py`, `utils/features.py` |
| `submit_train_jpsi_psip.sh` | `scripts/train_jpsi_vs_nonjpsi.py`, `scripts/train_psip_vs_nonpsip.py` | none (fixed feature set) | `utils/core_train_binary.py`, `utils/plots_binary.py`, `utils/features.py` |

- Default is 12-feat everywhere it's supported. `submit_train_dy_vs_comb.sh` retrains on the current tuned comb MC each iteration -> `checkpoints/Itr16_12feat/dy_comb_raw_st1_dropped.best.pth`.
- `submit_train_jpsi_psip.sh` has no 12/18 toggle (uses `core_train_binary.py`, not `_featsel`); adding one would need new code, not just a copy.
- No checkpoints/data files tracked here -- only scripts.
