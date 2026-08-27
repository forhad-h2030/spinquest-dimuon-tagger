#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

# Binary DY-vs-Comb retrain (produces checkpoints/Itr16_12feat). POS = DY,
# NEG = current tuned comb MC. FEATURE_SET: 12 (default) drops station-1
# x/px for both muons, dpt, mT; 18 trains on the full feature set.
#
# sbatch submit_train_dy_vs_comb.sh
# sbatch --export=ALL,FEATURE_SET=18 submit_train_dy_vs_comb.sh
BOOT_SEED=42
BOOT_TAG="boot_000"

export FEATURE_SET="${FEATURE_SET:-12}"
if [[ "$FEATURE_SET" == "18" ]]; then
  export DROP_FEATURES=""
else
  export DROP_FEATURES="rec_track_pos_x_st1,rec_track_neg_x_st1,rec_track_pos_px_st1,rec_track_neg_px_st1,rec_mu_dpt,rec_dimu_mT"
fi

export OUT_ROOT="${OUT_ROOT:-outputs_dy_vs_comb_itr16_${SLURM_JOB_ID}}"
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"
export STANDARDIZE="${STANDARDIZE:-1}"

export DATA_DIR="${SLURM_SUBMIT_DIR}/data/ml_input"
export DY_FILE="mc_raw_dy_compact_july_09_2026.npy"
export COMB_FILE="tuned_comb_Itr16_12feat.npy"

RUN_NAME="${RUN_NAME:-dy_comb_${FEATURE_SET}feat_itr16}"
export RUN_NAME
SCRIPT="scripts/train_dy_vs_comb_featsel.py"

export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
mkdir -p "$RUN_DIR"
export OUT_DIR="$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Redirect all output into the model directory
exec 1>"$RUN_DIR/slurm_${SLURM_JOB_ID}.out" \
     2>"$RUN_DIR/slurm_${SLURM_JOB_ID}.err"

export SPLIT_SEED="$BOOT_SEED"
export BOOT_SEED="$BOOT_SEED"

echo "[INFO] job_id=${SLURM_JOB_ID}  seed=${BOOT_SEED}  tag=${BOOT_TAG}"
echo "[INFO] feature_set=${FEATURE_SET}  drop_features=${DROP_FEATURES:-<none>}"
echo "[INFO] data_dir=${DATA_DIR}"
echo "[INFO] dy_file=${DY_FILE}"
echo "[INFO] comb_file=${COMB_FILE}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] epochs=${EPOCHS}  lr=${LR}  batch=${BATCH_SIZE}"

: "${CONTAINERDIR:?CONTAINERDIR is not set}"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"
if [[ ! -f "$SIF" ]]; then
  echo "[ERROR] container not found: $SIF"
  exit 2
fi

apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR="$MPLCONFIGDIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env BOOT_SEED="$BOOT_SEED" \
  --env SPLIT_SEED="$SPLIT_SEED" \
  --env EPOCHS="$EPOCHS" \
  --env LR="$LR" \
  --env BATCH_SIZE="$BATCH_SIZE" \
  --env STANDARDIZE="$STANDARDIZE" \
  --env DROP_FEATURES="$DROP_FEATURES" \
  --env RUN_NAME="$RUN_NAME" \
  --env DATA_DIR="$DATA_DIR" \
  --env DY_FILE="$DY_FILE" \
  --env COMB_FILE="$COMB_FILE" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 "$SCRIPT"

echo "[INFO] done. results in: $RUN_DIR"
