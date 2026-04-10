#!/bin/bash
# Sweep G1 tracking (No State Estimation) over seeds.
#
# Usage:
#   bash run_g1_tracking_sweep.sh --local /path/to/motion.npz
#   bash run_g1_tracking_sweep.sh --wandb org/registry/name

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# ── GPU ──────────────────────────────────────────────
GPU=1

# ── Task ─────────────────────────────────────────────
TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

# ── Training hyperparameters ─────────────────────────
NUM_ENVS=4096
MAX_ITERATIONS=3900
SAVE_INTERVAL=2000
EXPERIMENT_NAME="g1_tracking_dance1s2_sweep"

# ── Logging ──────────────────────────────────────────
LOGGER="wandb"
WANDB_PROJECT_NAME="mjlab tracking dance1s2"
VIDEO=True
VIDEO_INTERVAL=5000
VIDEO_LENGTH=300

# ── Parse motion source ──────────────────────────────
usage() {
  echo "Usage: $0 --local /path/to/motion.npz"
  echo "       $0 --wandb org/registry/name"
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

MOTION_FLAG=""
case "$1" in
  --local)
    if [[ ! -f "$2" ]]; then
      echo "[ERROR] Motion file not found: $2"
      exit 1
    fi
    MOTION_FLAG="--env.commands.motion.motion-file $2"
    ;;
  --wandb)
    MOTION_FLAG="--registry-name $2"
    ;;
  *)
    usage
    ;;
esac

# ── Environment ──────────────────────────────────────
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$GPU

echo "[sweep] Using GPU $GPU"

for SEED in $(seq 100 199); do
  echo "========================================"
  echo "[sweep] seed=$SEED  $(date)"
  echo "========================================"
  uv run train "$TASK" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$MAX_ITERATIONS" \
    --agent.save-interval "$SAVE_INTERVAL" \
    --agent.dense-save-iterations "(3200,3300,3400,3500,3600,3700,3800,3899)" \
    --agent.actor.hidden-dims "(128,128)" \
    --agent.actor.obs-normalization False \
    --agent.critic.obs-normalization False \
    --agent.seed "$SEED" \
    --agent.experiment-name "$EXPERIMENT_NAME" \
    --agent.run-name "seed_${SEED}" \
    --agent.logger "$LOGGER" \
    --agent.wandb-project "$WANDB_PROJECT_NAME" \
    --env.commands.motion.sampling-mode uniform \
    --video "$VIDEO" \
    --video-interval "$VIDEO_INTERVAL" \
    --video-length "$VIDEO_LENGTH" \
    --gpu-ids "[$GPU]" \
    $MOTION_FLAG
done

echo "[sweep] All seeds done."
