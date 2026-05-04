#!/bin/bash
# Batch-evaluate every checkpoint in a tracking sweep directory.
#
# Usage:
#   bash eval_tracking_sweep.sh \
#     --sweep-dir logs/rsl_rl/g1_tracking_dance1s2_sweep \
#     --motion-file /path/to/motion.npz \
#     [--task Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation] \
#     [--iters "3200 3300 3400 3500 3600 3700 3800 3899"] \
#     [--num-envs 512] \
#     [--output-dir eval_results] \
#     [--gpu 0]

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# Defaults.
TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
ITERS="3200 3300 3400 3500 3600 3700 3800 3899"
NUM_ENVS=512
OUTPUT_DIR="eval_results"
GPU=0
SWEEP_DIR=""
MOTION_FILE=""

usage() {
  echo "Usage: $0 --sweep-dir DIR --motion-file PATH [options]"
  echo "  --sweep-dir DIR        Sweep root, e.g. logs/rsl_rl/g1_tracking_dance1s2_sweep"
  echo "  --motion-file PATH     Motion .npz used during training"
  echo "  --task TASK_ID         Task id (default: $TASK)"
  echo "  --iters \"3200 3300\"    Iterations to evaluate (default: dense save iters)"
  echo "  --num-envs N           Parallel envs per eval (default: $NUM_ENVS)"
  echo "  --output-dir DIR       Where to write per-checkpoint JSON (default: $OUTPUT_DIR)"
  echo "  --gpu ID               GPU id (default: $GPU)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep-dir)   SWEEP_DIR="$2"; shift 2 ;;
    --motion-file) MOTION_FILE="$2"; shift 2 ;;
    --task)        TASK="$2"; shift 2 ;;
    --iters)       ITERS="$2"; shift 2 ;;
    --num-envs)    NUM_ENVS="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --gpu)         GPU="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "[ERROR] Unknown arg: $1"; usage ;;
  esac
done

[[ -n "$SWEEP_DIR" && -n "$MOTION_FILE" ]] || usage
[[ -d "$SWEEP_DIR" ]]   || { echo "[ERROR] Sweep dir not found: $SWEEP_DIR"; exit 1; }
[[ -f "$MOTION_FILE" ]] || { echo "[ERROR] Motion file not found: $MOTION_FILE"; exit 1; }

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"

mkdir -p "$OUTPUT_DIR"

FAILED=()
TOTAL=0
OK=0
SKIPPED=0

for run_dir in "$SWEEP_DIR"/*/; do
  run_name=$(basename "$run_dir")
  for it in $ITERS; do
    ckpt="${run_dir}model_${it}.pt"
    out="${OUTPUT_DIR}/${run_name}__model_${it}.json"
    TOTAL=$((TOTAL + 1))

    if [[ ! -f "$ckpt" ]]; then
      echo "[skip] missing: $ckpt"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    if [[ -f "$out" ]]; then
      echo "[skip] already evaluated: $out"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo "========================================"
    echo "[eval] $run_name  iter=$it  $(date)"
    echo "========================================"
    uv run python -m mjlab.tasks.tracking.scripts.evaluate "$TASK" \
      --checkpoint-file "$ckpt" \
      --motion-file "$MOTION_FILE" \
      --num-envs "$NUM_ENVS" \
      --output-file "$out"
    status=$?

    if (( status == 0 )); then
      OK=$((OK + 1))
    else
      echo "[eval] FAILED (exit=$status): $ckpt"
      FAILED+=("$ckpt")
    fi
  done
done

echo "========================================"
echo "[eval] Summary: total=$TOTAL ok=$OK skipped=$SKIPPED failed=${#FAILED[@]}"
if (( ${#FAILED[@]} > 0 )); then
  printf '  failed: %s\n' "${FAILED[@]}"
  exit 1
fi
