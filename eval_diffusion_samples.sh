#!/bin/bash
# Batch-evaluate diffusion-sampled policy .pt files using mjlab tracking metrics.
#
# The diffusion samples share the rsl_rl checkpoint format (actor_state_dict +
# placeholder critic/optimizer dicts), so they load via evaluate.py's
# --checkpoint-file mode unchanged.
#
# Usage:
#   bash eval_diffusion_samples.sh \
#     --sample-dir /path/to/diff_weight/sample/outputs/g1dance1s2/<run> \
#     --motion-file /path/to/motion.npz \
#     [--task Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation] \
#     [--num-envs 512] \
#     [--output-dir eval_results_diffusion] \
#     [--gpu 0] \
#     [--pattern "sample_*.pt"]

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# Defaults.
TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
NUM_ENVS=512
OUTPUT_DIR=""  # default: <sample-dir>/eval
GPU=2
PATTERN="sample_*.pt"
MAX_SAMPLES=100  # 0 = no limit
SAMPLE_DIR="/home/huixuan_cheng/diff_weight/sample/outputs/g1dance1s2/NL8_EMBED1024_PATCH16_PREDsample_BS256_LR2E04_EMAon_LNzscore_final"
MOTION_FILE="/home/huixuan_cheng/mjlab/artifacts/dance1s2:v0/motion.npz"

usage() {
  echo "Usage: $0 --sample-dir DIR --motion-file PATH [options]"
  echo "  --sample-dir DIR       Directory containing diffusion sample .pt files"
  echo "  --motion-file PATH     Motion .npz used during the original training"
  echo "  --task TASK_ID         Task id (default: $TASK)"
  echo "  --num-envs N           Parallel envs per eval (default: $NUM_ENVS)"
  echo "  --output-dir DIR       Where to write per-sample JSON (default: <sample-dir>/eval)"
  echo "  --gpu ID               GPU id (default: $GPU)"
  echo "  --pattern GLOB         Filename glob inside sample-dir (default: $PATTERN)"
  echo "  --max-samples N        Only evaluate the first N matched samples (0 = all)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-dir)  SAMPLE_DIR="$2"; shift 2 ;;
    --motion-file) MOTION_FILE="$2"; shift 2 ;;
    --task)        TASK="$2"; shift 2 ;;
    --num-envs)    NUM_ENVS="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --gpu)         GPU="$2"; shift 2 ;;
    --pattern)     PATTERN="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "[ERROR] Unknown arg: $1"; usage ;;
  esac
done

[[ -n "$SAMPLE_DIR" && -n "$MOTION_FILE" ]] || usage
[[ -d "$SAMPLE_DIR" ]]  || { echo "[ERROR] Sample dir not found: $SAMPLE_DIR"; exit 1; }
[[ -f "$MOTION_FILE" ]] || { echo "[ERROR] Motion file not found: $MOTION_FILE"; exit 1; }

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"

# Default output is an "eval" subdir inside the sample dir itself; explicit
# --output-dir overrides and is used as-is.
if [[ -z "$OUTPUT_DIR" ]]; then
  OUT_SUBDIR="${SAMPLE_DIR}/eval"
else
  OUT_SUBDIR="$OUTPUT_DIR"
fi
mkdir -p "$OUT_SUBDIR"

shopt -s nullglob
samples=( "$SAMPLE_DIR"/$PATTERN )
shopt -u nullglob

if (( ${#samples[@]} == 0 )); then
  echo "[ERROR] No files matching '$PATTERN' under $SAMPLE_DIR"
  exit 1
fi

# Sort to make "first N" deterministic (lexicographic = numeric for sample_NNN.pt).
IFS=$'\n' samples=($(printf '%s\n' "${samples[@]}" | sort))
unset IFS

found_total=${#samples[@]}
if (( MAX_SAMPLES > 0 && MAX_SAMPLES < found_total )); then
  samples=( "${samples[@]:0:$MAX_SAMPLES}" )
  echo "[eval] Found $found_total samples, limiting to first $MAX_SAMPLES."
else
  echo "[eval] Found $found_total samples in $SAMPLE_DIR"
fi
echo "[eval] Writing results to $OUT_SUBDIR"

FAILED=()
TOTAL=0
OK=0
SKIPPED=0

for ckpt in "${samples[@]}"; do
  name=$(basename "$ckpt" .pt)
  out="${OUT_SUBDIR}/${name}.json"
  TOTAL=$((TOTAL + 1))

  if [[ -f "$out" ]]; then
    echo "[skip] already evaluated: $out"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "========================================"
  echo "[eval] $name  $(date)"
  echo "========================================"
  uv run python -m mjlab.tasks.tracking.scripts.evaluate "$TASK" \
    --checkpoint-file "$ckpt" \
    --motion-file "$MOTION_FILE" \
    --num-envs "$NUM_ENVS" \
    --device "cuda:$GPU" \
    --output-file "$out"
  status=$?

  if (( status == 0 )); then
    OK=$((OK + 1))
  else
    echo "[eval] FAILED (exit=$status): $ckpt"
    FAILED+=("$ckpt")
  fi
done

echo "========================================"
echo "[eval] Summary: total=$TOTAL ok=$OK skipped=$SKIPPED failed=${#FAILED[@]}"
echo "[eval] Results dir: $OUT_SUBDIR"
if (( ${#FAILED[@]} > 0 )); then
  printf '  failed: %s\n' "${FAILED[@]}"
  exit 1
fi
