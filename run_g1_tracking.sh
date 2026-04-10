#!/bin/bash
# Train G1 tracking (No State Estimation) — for sim-to-real.
#
# Usage:
#   # Local motion file:
#   bash run_g1_tracking.sh --local /path/to/motion.npz
#
#   # WandB registry:
#   bash run_g1_tracking.sh --wandb your-org/registry/motion-name

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# ── GPU ──────────────────────────────────────────────
GPU=6

# ── Task ─────────────────────────────────────────────
TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

# ── Training hyperparameters ─────────────────────────
NUM_ENVS=4096
MAX_ITERATIONS=3900
SAVE_INTERVAL=2000
SEED=0
EXPERIMENT_NAME="g1_tracking_dance1s2"
RUN_NAME="seed_${SEED}"

# ── Logging ──────────────────────────────────────────
LOGGER="wandb"          # "wandb" or "tensorboard"
WANDB_PROJECT_NAME="mjlab tracking dance1s2"   # wandb project name
VIDEO=True
VIDEO_INTERVAL=5000
VIDEO_LENGTH=300

# ── Parse motion source ──────────────────────────────
MOTION_SOURCE=""
MOTION_ARG=""

usage() {
  echo "Usage: $0 --local /path/to/motion.npz"
  echo "       $0 --wandb org/registry/name"
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

case "$1" in
  --local)
    MOTION_SOURCE="local"
    MOTION_ARG="$2"
    if [[ ! -f "$MOTION_ARG" ]]; then
      echo "[ERROR] Motion file not found: $MOTION_ARG"
      exit 1
    fi
    ;;
  --wandb)
    MOTION_SOURCE="wandb"
    MOTION_ARG="$2"
    ;;
  *)
    usage
    ;;
esac

# ── Environment ──────────────────────────────────────
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$GPU

# ── Build command ────────────────────────────────────
CMD=(
  uv run train "$TASK"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$MAX_ITERATIONS"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.seed "$SEED"
  --agent.experiment-name "$EXPERIMENT_NAME"
  --agent.run-name "$RUN_NAME"
  --agent.logger "$LOGGER"
  --agent.wandb-project "$WANDB_PROJECT_NAME"
  --video "$VIDEO"
  --video-interval "$VIDEO_INTERVAL"
  --video-length "$VIDEO_LENGTH"
  --gpu-ids "[$GPU]"
  --agent.actor.obs-normalization False
  --agent.critic.obs-normalization False
  --agent.actor.hidden-dims "(128, 128)"
  --agent.dense-save-iterations "(3200,3300,3400,3500,3600,3700,3800,3899)"
  --env.commands.motion.sampling-mode uniform
)

if [[ "$MOTION_SOURCE" == "local" ]]; then
  CMD+=(--env.commands.motion.motion-file "$MOTION_ARG")
else
  CMD+=(--registry-name "$MOTION_ARG")
fi

echo "========================================"
echo "[tracking] Task:   $TASK"
echo "[tracking] Motion: $MOTION_SOURCE → $MOTION_ARG"
echo "[tracking] GPU:    $GPU"
echo "[tracking] Envs:   $NUM_ENVS"
echo "[tracking] Iters:  $MAX_ITERATIONS"
echo "[tracking] $(date)"
echo "========================================"

"${CMD[@]}"

echo "[tracking] Done."
