<img width="1502" height="619" alt="Itr-dnn-nf" src="https://github.com/user-attachments/assets/b9572991-0b85-40c6-b573-2ddb11fa0d26" />
# spinquest-combinatoric-bkg

DNN classifiers (binary: J/ψ, ψ′, DY-vs-comb; 4-class multiclass) for tagging
dimuon events in SpinQuest analyses. ROOT → NumPy preprocessing, training,
and application to experimental data, on Rivanna (Slurm + GPU).

---
## Repository Structure

```text
spinquest-combinatoric-bkg/
├── preprocess/       # ROOT -> NumPy feature extraction (see data/README.md)
├── slurm/            # production training jobs (see slurm/README.md)
├── scripts/          # training & inference drivers
├── utils/            # model/training/plotting code, feature definitions
└── data/             # ROOT inputs & generated NumPy features
```

## Running

1. **Preprocess**: `cd preprocess && source setup.sh && python3 run_extract_final.py`
2. **Train**: `sbatch slurm/<script>.sh` -- see `slurm/README.md` for which script trains what.
3. **Apply to experimental data**: `python3 scripts/tag_exp_root_with_ml.py`
