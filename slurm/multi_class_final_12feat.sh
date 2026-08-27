#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=3:30:00
#SBATCH --array=0-19
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# 12-feat (default) / 18-feat multiclass production run, 20-seed array job.
# Driver: scripts/train_multiclass_featsel.py. USE_CLASS_WEIGHTS=1 (full data,
# loss reweighted per class). DROP_FEATURES hardcoded in the driver for
# 12-feat; leave unset for 12-feat, export empty for 18-feat.
#
# sbatch multi_class_final_12feat.sh
# sbatch --export=ALL,DROP_FEATURES= multi_class_final_12feat.sh   # 18-feat

set -euo pipefail

module purge
module load apptainer pytorch/2.7.0

cd "${SLURM_SUBMIT_DIR}"

# ── Distinct seeds for independent seeded replicas ───────────────────────────
SEEDS=(
  43 124 457 789 1011
  1213 1415 1617 1819 2021
  2223 2425 2627 2829 3031
  3233 3435 3637 3839 4041
)
IDX="${SLURM_ARRAY_TASK_ID}"
if (( IDX < 0 || IDX >= ${#SEEDS[@]} )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX} outside seed list length ${#SEEDS[@]}"
  exit 3
fi
BOOT_SEED="${SEEDS[$IDX]}"
BOOT_TAG=$(printf "boot_%03d" "$IDX")

# ── Model config: matches multi_class_final_classweighted.sh exactly ─────────
export OUT_ROOT="${OUT_ROOT:-outputs_final_12feat_classweighted_${SLURM_ARRAY_JOB_ID}}"
export EPOCHS="${EPOCHS:-250}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export LR="${LR:-5e-4}"
export LR_MIN="${LR_MIN:-1e-6}"
export STANDARDIZE="${STANDARDIZE:-1}"
export OPTIMIZER="adamw"
export SCHEDULER="onecycle"
export LOSS_TYPE="ce_ls"
export LABEL_SMOOTHING="0.05"
export FOCAL_GAMMA="2.0"
export DROPOUT="0.1"
export FLAT="1"
export MODEL_TYPE="dnn"
export HIDDEN_DIM="512"
export NUM_LAYERS="4"
export USE_CLASS_WEIGHTS="${USE_CLASS_WEIGHTS:-1}"

export DATA_DIR="${DATA_DIR:-${SLURM_SUBMIT_DIR}/data/ml_input_final}"
# Same production files as multi_class_final.sh (the 18-feat final run) --
# updated from the July 29 defaults (mc_jpsi_tuned1_july16.npy /
# mc_psip_tuned1_july26.npy, 987,357 / 1,366,284 events) which predate the
# July 31 2M-event J/psi + psi(2S) production files used by muticlass_final_reg.
export JPSI_FILE="${JPSI_FILE:-tuned_jpsi_multiclass_12feat_itr2_July31_2026_2M.npy}"
export PSIP_FILE="${PSIP_FILE:-tuned_psip_multiclass_12feat_itr2_July31_2026_2M.npy}"
export DY_FILE="${DY_FILE:-tuned_dy_8d_energy_pyopt_full_1m.npy}"
export COMB_FILE="${COMB_FILE:-tuned_comb_deep12_8d_energy_1m.npy}"
RUN_NAME="${RUN_NAME:-multiclass_dnn_12feat_classweighted_20seed}"
SCRIPT="scripts/train_multiclass_featsel.py"

export RUN_DIR="${OUT_ROOT}/${RUN_NAME}/${BOOT_TAG}"
mkdir -p "$RUN_DIR"
export OUT_DIR="$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Redirect all output into the model directory
exec 1>"$RUN_DIR/slurm_${SLURM_ARRAY_JOB_ID}_${IDX}.out" \
     2>"$RUN_DIR/slurm_${SLURM_ARRAY_JOB_ID}_${IDX}.err"

export SPLIT_SEED="$BOOT_SEED"
export BOOT_SEED="$BOOT_SEED"

echo "[INFO] job_id=${SLURM_JOB_ID}  task=${IDX}  seed=${BOOT_SEED}  tag=${BOOT_TAG}"
echo "[INFO] model=DNN flat 512x4 (12 features)  replicas=${#SEEDS[@]}  optimizer=adamw  scheduler=onecycle  loss=ce_ls"
echo "[INFO] use_class_weights=${USE_CLASS_WEIGHTS}  balanced_per_class=$((1 - USE_CLASS_WEIGHTS))"
echo "[INFO] data_dir=${DATA_DIR}"
echo "[INFO] jpsi_file=${JPSI_FILE}"
echo "[INFO] psip_file=${PSIP_FILE}"
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

# only forward DROP_FEATURES if the user set it (empty string would mean "drop nothing")
EXTRA_ENV=()
if [[ -n "${DROP_FEATURES:-}" ]]; then
  EXTRA_ENV+=(--env DROP_FEATURES="$DROP_FEATURES")
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
  --env LR_MIN="$LR_MIN" \
  --env BATCH_SIZE="$BATCH_SIZE" \
  --env STANDARDIZE="$STANDARDIZE" \
  --env MODEL_TYPE="$MODEL_TYPE" \
  --env HIDDEN_DIM="$HIDDEN_DIM" \
  --env NUM_LAYERS="$NUM_LAYERS" \
  --env USE_CLASS_WEIGHTS="$USE_CLASS_WEIGHTS" \
  --env DROPOUT="$DROPOUT" \
  --env FLAT="$FLAT" \
  --env LOSS_TYPE="$LOSS_TYPE" \
  --env FOCAL_GAMMA="$FOCAL_GAMMA" \
  --env LABEL_SMOOTHING="$LABEL_SMOOTHING" \
  --env OPTIMIZER="$OPTIMIZER" \
  --env SCHEDULER="$SCHEDULER" \
  --env RUN_NAME="$RUN_NAME" \
  --env DATA_DIR="$DATA_DIR" \
  --env JPSI_FILE="$JPSI_FILE" \
  --env PSIP_FILE="$PSIP_FILE" \
  --env DY_FILE="$DY_FILE" \
  --env COMB_FILE="$COMB_FILE" \
  "${EXTRA_ENV[@]}" \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY= \
  --env QT_PLUGIN_PATH= \
  --env QT_QPA_PLATFORM_PLUGIN_PATH= \
  --env XDG_RUNTIME_DIR=/tmp \
  "$SIF" \
  python3 "$SCRIPT"

echo "[INFO] done. results in: $RUN_DIR"
