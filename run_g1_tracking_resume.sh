#!/bin/bash
# Batch-resume every seed in a tracking sweep against a new motion file.
#
# How it works:
#   1. For each seed in [SEED_START, SEED_END], find the source run directory
#      under logs/rsl_rl/<source-experiment>/.
#   2. Symlink it into logs/rsl_rl/<resume-experiment>/ so train.py's
#      --agent.load-run resolver finds the source ckpt.
#   3. Resume training with --agent.experiment-name <resume-experiment>;
#      the new run directory lives alongside the symlink in the resume dir.
#
# This is idempotent: a seed is skipped if the resume run already produced
# a final-iteration ckpt, so the script can be safely re-run after an
# interruption.
#
# Parallelize across GPUs by slicing seeds:
#   bash run_g1_tracking_resume.sh --gpu 0 --seed-start 0   --seed-end 249 ...
#   bash run_g1_tracking_resume.sh --gpu 1 --seed-start 250 --seed-end 499 ...
#   bash run_g1_tracking_resume.sh --gpu 2 --seed-start 500 --seed-end 749 ...

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# Defaults.
TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
SOURCE_EXPERIMENT="g1_tracking_dance1s2_sweep"
RESUME_EXPERIMENT="g1_tracking_dance1s2_sweep_resume"
MOTION_FILE="/home/huixuan_cheng/mjlab/artifacts/dance1s2_ramp05_hold2s/motion.npz"
NUM_ENVS=4096
MAX_ITERATIONS=805
SAVE_INTERVAL=2000 
DENSE_SAVE_ITERS="(4000,4100,4200,4300,4400,4500,4600,4700)"
LOAD_CHECKPOINT_REGEX="model_.*.pt"  # picks highest-iter ckpt
GPU=2
RUN_SUFFIX="hold2s_finetune"
WANDB_PROJECT_NAME="mjlab tracking dance1s2 resume"
SEED_START=0
SEED_END=249
MAX_RETRIES=2

# wandb resilience.
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=60
export WANDB__SERVICE_WAIT=300
export WANDB_RESUME=allow

usage() {
  echo "Usage: $0 [options]"
  echo "  --motion-file PATH       Motion .npz to finetune against (default: $MOTION_FILE)"
  echo "  --source-experiment NAME Original experiment dir name (default: $SOURCE_EXPERIMENT)"
  echo "  --resume-experiment NAME New experiment dir to write into (default: $RESUME_EXPERIMENT)"
  echo "  --seed-start N           First seed to resume (default: $SEED_START)"
  echo "  --seed-end N             Last seed to resume, inclusive (default: $SEED_END)"
  echo "  --gpu ID                 GPU id (default: $GPU)"
  echo "  --num-envs N             Parallel envs (default: $NUM_ENVS)"
  echo "  --max-iterations N       Target total iterations (default: $MAX_ITERATIONS)"
  echo "  --run-suffix STR         Suffix appended to each new run name (default: $RUN_SUFFIX)"
  echo "  --task TASK_ID           Task id (default: $TASK)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-file)        MOTION_FILE="$2"; shift 2 ;;
    --source-experiment)  SOURCE_EXPERIMENT="$2"; shift 2 ;;
    --resume-experiment)  RESUME_EXPERIMENT="$2"; shift 2 ;;
    --seed-start)         SEED_START="$2"; shift 2 ;;
    --seed-end)           SEED_END="$2"; shift 2 ;;
    --gpu)                GPU="$2"; shift 2 ;;
    --num-envs)           NUM_ENVS="$2"; shift 2 ;;
    --max-iterations)     MAX_ITERATIONS="$2"; shift 2 ;;
    --run-suffix)         RUN_SUFFIX="$2"; shift 2 ;;
    --task)               TASK="$2"; shift 2 ;;
    -h|--help)            usage ;;
    *) echo "[ERROR] Unknown arg: $1"; usage ;;
  esac
done

[[ -f "$MOTION_FILE" ]] || { echo "[ERROR] Motion file not found: $MOTION_FILE"; exit 1; }

SOURCE_DIR="logs/rsl_rl/$SOURCE_EXPERIMENT"
RESUME_DIR="logs/rsl_rl/$RESUME_EXPERIMENT"
[[ -d "$SOURCE_DIR" ]] || { echo "[ERROR] Source dir not found: $SOURCE_DIR"; exit 1; }
mkdir -p "$RESUME_DIR"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"

FINAL_ITER=$((MAX_ITERATIONS - 1))

echo "========================================"
echo "[batch-resume] source:        $SOURCE_DIR"
echo "[batch-resume] resume target: $RESUME_DIR"
echo "[batch-resume] motion:        $MOTION_FILE"
echo "[batch-resume] seeds:         $SEED_START..$SEED_END"
echo "[batch-resume] gpu:           $GPU"
echo "[batch-resume] target iters:  $MAX_ITERATIONS  (final ckpt: model_$FINAL_ITER.pt)"
echo "[batch-resume] run suffix:    $RUN_SUFFIX"
echo "========================================"

OK=0
SKIPPED_NO_SOURCE=0
SKIPPED_DONE=0
FAILED=()

for SEED in $(seq "$SEED_START" "$SEED_END"); do
  # Find the source run directory for this seed (timestamped: <date>_seed_<N>).
  shopt -s nullglob
  candidates=( "$SOURCE_DIR"/*_seed_${SEED} )
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    SKIPPED_NO_SOURCE=$((SKIPPED_NO_SOURCE + 1))
    continue
  fi
  # If multiple matches, pick the most recent by sorted name (timestamp prefix).
  IFS=$'\n' sorted=( $(printf '%s\n' "${candidates[@]}" | sort) )
  unset IFS
  SOURCE_RUN_PATH="${sorted[-1]}"
  SOURCE_RUN_NAME=$(basename "$SOURCE_RUN_PATH")

  # Skip if a finetuned run for this seed already produced the final ckpt.
  shopt -s nullglob
  done_ckpts=( "$RESUME_DIR"/*_seed_${SEED}_${RUN_SUFFIX}/model_${FINAL_ITER}.pt )
  shopt -u nullglob
  if (( ${#done_ckpts[@]} > 0 )); then
    echo "[skip] seed=$SEED already finetuned (${done_ckpts[0]})"
    SKIPPED_DONE=$((SKIPPED_DONE + 1))
    continue
  fi

  # Ensure a symlink exists in the resume dir so train.py's load-run resolver
  # finds the source run when --agent.experiment-name = RESUME_EXPERIMENT.
  LINK="$RESUME_DIR/$SOURCE_RUN_NAME"
  if [[ ! -e "$LINK" ]]; then
    ln -s "$(realpath "$SOURCE_RUN_PATH")" "$LINK"
  fi

  NEW_RUN_NAME="${SOURCE_RUN_NAME}_${RUN_SUFFIX}"

  echo "========================================"
  echo "[batch-resume] seed=$SEED  source=$SOURCE_RUN_NAME  $(date)"
  echo "========================================"

  attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "[batch-resume] seed=$SEED attempt=$attempt/$MAX_RETRIES"
    set +e
    uv run train "$TASK" \
      --agent.resume True \
      --agent.experiment-name "$RESUME_EXPERIMENT" \
      --agent.load-run "^${SOURCE_RUN_NAME}\$" \
      --agent.load-checkpoint "$LOAD_CHECKPOINT_REGEX" \
      --agent.run-name "$NEW_RUN_NAME" \
      --agent.seed "$SEED" \
      --agent.max-iterations "$MAX_ITERATIONS" \
      --agent.save-interval "$SAVE_INTERVAL" \
      --agent.dense-save-iterations "$DENSE_SAVE_ITERS" \
      --agent.actor.hidden-dims "(128,128)" \
      --agent.actor.obs-normalization False \
      --agent.critic.obs-normalization False \
      --agent.wandb-project "$WANDB_PROJECT_NAME" \
      --env.scene.num-envs "$NUM_ENVS" \
      --env.commands.motion.motion-file "$MOTION_FILE" \
      --env.commands.motion.sampling-mode uniform \
      --gpu-ids "[$GPU]"
    status=$?
    set -e

    if (( status == 0 )); then
      echo "[batch-resume] seed=$SEED OK"
      OK=$((OK + 1))
      break
    fi
    echo "[batch-resume] seed=$SEED failed (exit=$status) on attempt $attempt"
    if (( attempt == MAX_RETRIES )); then
      FAILED+=("$SEED")
      break
    fi
    backoff=$(( 30 * attempt ))
    echo "[batch-resume] retrying seed=$SEED in ${backoff}s..."
    sleep "$backoff"
    attempt=$(( attempt + 1 ))
  done
done

echo "========================================"
echo "[batch-resume] Summary"
echo "  OK:                  $OK"
echo "  Skipped (no source): $SKIPPED_NO_SOURCE"
echo "  Skipped (done):      $SKIPPED_DONE"
echo "  Failed:              ${#FAILED[@]}"
if (( ${#FAILED[@]} > 0 )); then
  printf '  failed seed: %s\n' "${FAILED[@]}"
  exit 1
fi
