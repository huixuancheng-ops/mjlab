#!/bin/bash
# Sweep Mjlab-Velocity-Flat-Unitree-G1 over seeds 0-499.
# Seeds run sequentially on a single GPU.

export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# Auto-select the GPU with the least memory usage.
# EMPTY_GPU=$(nvidia-smi --query-gpu=index,memory.used \
#   --format=csv,noheader,nounits \
#   | awk -F, '{print $2+0, $1}' | sort -n | head -1 | awk '{print $2}' | tr -d ' ')
EMPTY_GPU=0
echo "[sweep] Using GPU $EMPTY_GPU"

# Headless EGL rendering for video recording.
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$EMPTY_GPU

for SEED in $(seq 500 999); do
  echo "========================================"
  echo "[sweep] seed=$SEED  $(date)"
  echo "========================================"
  uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 3500 \
    --agent.save-interval 1000 \
    --agent.dense-save-iterations "(3000,3100,3200,3300,3400,3499)" \
    --agent.actor.hidden-dims "(128,128)" \
    --agent.seed "$SEED" \
    --agent.experiment-name "g1_flat_velocity_sweep" \
    --agent.run-name "seed_${SEED}" \
    --agent.logger wandb \
    --video True \
    --video-interval 2000 \
    --video-length 200 \
    --gpu-ids "[$EMPTY_GPU]"
done

echo "[sweep] All seeds done."
