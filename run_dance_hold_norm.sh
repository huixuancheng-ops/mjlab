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
     GPU=0

     # ── Task ─────────────────────────────────────────────
     TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

     # ── Training hyperparameters ─────────────────────────
     NUM_ENVS=4096
     MAX_ITERATIONS=4500
     SAVE_INTERVAL=2000
     EXPERIMENT_NAME="g1_tracking_dance1s2_hold_sweep"

     # ── Logging ──────────────────────────────────────────
     LOGGER="wandb"
     WANDB_PROJECT_NAME="mjlab tracking dance1s2 hold"
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

     # wandb: tolerate slow networks so a single flaky init doesn't kill the sweep.
     export WANDB_INIT_TIMEOUT=300
     export WANDB_HTTP_TIMEOUT=60
     export WANDB__SERVICE_WAIT=300
     export WANDB_RESUME=allow

     MAX_RETRIES=3
     FAILED_SEEDS=()

     echo "[sweep] Using GPU $GPU"

     for SEED in $(seq 133 133); do
       echo "========================================"
       echo "[sweep] seed=$SEED  $(date)"
       echo "========================================"

       attempt=1
       while (( attempt <= MAX_RETRIES )); do
         echo "[sweep] seed=$SEED attempt=$attempt/$MAX_RETRIES"
         # Disable -e locally so a failed run doesn't terminate the whole sweep.
         set +e
         uv run train "$TASK" \
           --env.scene.num-envs "$NUM_ENVS" \
           --agent.max-iterations "$MAX_ITERATIONS" \
           --agent.save-interval "$SAVE_INTERVAL" \
           --agent.dense-save-iterations "(3800,3900,4000,4100,4200,4300,4400,4499)" \
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
         status=$?
         set -e

         if (( status == 0 )); then
           echo "[sweep] seed=$SEED OK"
           break
         fi

         echo "[sweep] seed=$SEED failed (exit=$status) on attempt $attempt"
         if (( attempt == MAX_RETRIES )); then
           echo "[sweep] seed=$SEED giving up after $MAX_RETRIES attempts"
           FAILED_SEEDS+=("$SEED")
           break
         fi
         # Backoff before retry (10s, 30s, 90s, ...).
         backoff=$(( 10 * (3 ** (attempt - 1)) ))
         echo "[sweep] retrying seed=$SEED in ${backoff}s..."
         sleep "$backoff"
         attempt=$(( attempt + 1 ))
       done
     done

     if (( ${#FAILED_SEEDS[@]} > 0 )); then
       echo "[sweep] Completed with failures. Failed seeds: ${FAILED_SEEDS[*]}"
       exit 1
     fi
     echo "[sweep] All seeds done."