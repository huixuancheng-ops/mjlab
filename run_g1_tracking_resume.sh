#!/bin/bash
# Resume G1 tracking training from a specific checkpoint .pt file.
#
# Auto-derives --agent.experiment-name / --agent.load-run / --agent.load-checkpoint
# from the checkpoint path. The .pt must live under the standard layout:
#   logs/rsl_rl/<experiment_name>/<run_dir>/<model_*.pt>
#
# Usage (minimal):
#   bash run_g1_tracking_resume.sh \
#     --ckpt /home/huixuan_cheng/mjlab/logs/rsl_rl/g1_tracking_dance1s2_sweep/2026-04-11_21-56-33_seed_600/model_3899.pt \
#     --motion-file /home/huixuan_cheng/mjlab/artifacts/dance1s2:v0/motion.npz
#
# Usage (full):
#   bash run_g1_tracking_resume.sh \
#     --ckpt PATH --motion-file PATH \
#     [--task TASK_ID] [--max-iterations N] [--num-envs N] [--gpu ID] [--run-suffix STR]

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# Defaults.
TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
NUM_ENVS=4096
MAX_ITERATIONS=5500
SAVE_INTERVAL=2000
DENSE_SAVE_ITERS="(4200,4400,4600,4800,4900,5000,5100,5200,5300,5400,5499)"
GPU=4
RUN_SUFFIX="resumed"
CKPT=""
MOTION_FILE=""

# wandb resilience (same as the from-scratch sweep).
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=60
export WANDB__SERVICE_WAIT=300
export WANDB_RESUME=allow

usage() {
  echo "Usage: $0 --ckpt PATH --motion-file PATH [options]"
  echo "  --ckpt PATH            .pt to resume from (path encodes experiment/run/ckpt)"
  echo "  --motion-file PATH     Motion .npz used during the original training"
  echo "  --task TASK_ID         Task id (default: $TASK)"
  echo "  --max-iterations N     Target total iterations (default: $MAX_ITERATIONS)"
  echo "  --num-envs N           Parallel envs (default: $NUM_ENVS)"
  echo "  --gpu ID               GPU id (default: $GPU)"
  echo "  --run-suffix STR       Suffix appended to run-name (default: $RUN_SUFFIX)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)           CKPT="$2"; shift 2 ;;
    --motion-file)    MOTION_FILE="$2"; shift 2 ;;
    --task)           TASK="$2"; shift 2 ;;
    --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
    --num-envs)       NUM_ENVS="$2"; shift 2 ;;
    --gpu)            GPU="$2"; shift 2 ;;
    --run-suffix)     RUN_SUFFIX="$2"; shift 2 ;;
    -h|--help)        usage ;;
    *) echo "[ERROR] Unknown arg: $1"; usage ;;
  esac
done

[[ -n "$CKPT" && -n "$MOTION_FILE" ]] || usage
[[ -f "$CKPT" ]]        || { echo "[ERROR] Checkpoint not found: $CKPT"; exit 1; }
[[ -f "$MOTION_FILE" ]] || { echo "[ERROR] Motion file not found: $MOTION_FILE"; exit 1; }

# Derive experiment_name / load_run / load_checkpoint from the .pt path.
LOAD_CHECKPOINT=$(basename "$CKPT")
RUN_DIR=$(dirname "$CKPT")
LOAD_RUN=$(basename "$RUN_DIR")
EXPERIMENT_NAME=$(basename "$(dirname "$RUN_DIR")")

# Try to recover the original seed from the run dir name (e.g. "..._seed_600" -> 600).
ORIG_SEED=""
if [[ "$LOAD_RUN" =~ _seed_([0-9]+) ]]; then
  ORIG_SEED="${BASH_REMATCH[1]}"
fi
SEED="${ORIG_SEED:-0}"

NEW_RUN_NAME="${LOAD_RUN}_${RUN_SUFFIX}"

echo "========================================"
echo "[resume] checkpoint:    $CKPT"
echo "[resume] experiment:    $EXPERIMENT_NAME"
echo "[resume] load-run:      $LOAD_RUN"
echo "[resume] load-ckpt:     $LOAD_CHECKPOINT"
echo "[resume] new run name:  $NEW_RUN_NAME"
echo "[resume] seed:          $SEED"
echo "[resume] max-iters:     $MAX_ITERATIONS"
echo "[resume] motion-file:   $MOTION_FILE"
echo "[resume] gpu:           $GPU"
echo "========================================"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"

uv run train "$TASK" \
  --agent.resume True \
  --agent.experiment-name "$EXPERIMENT_NAME" \
  --agent.load-run "$LOAD_RUN" \
  --agent.load-checkpoint "$LOAD_CHECKPOINT" \
  --agent.run-name "$NEW_RUN_NAME" \
  --agent.seed "$SEED" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.save-interval "$SAVE_INTERVAL" \
  --agent.dense-save-iterations "$DENSE_SAVE_ITERS" \
  --agent.actor.hidden-dims "(128,128)" \
  --agent.actor.obs-normalization False \
  --agent.critic.obs-normalization False \
  --env.scene.num-envs "$NUM_ENVS" \
  --env.commands.motion.motion-file "$MOTION_FILE" \
  --env.commands.motion.sampling-mode uniform \
  --gpu-ids "[$GPU]"
