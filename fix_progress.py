#!/usr/bin/env python3
"""Correct under-normalized `progress_mean` in existing tracking eval JSONs.

Old (buggy) per-env progress  = time_steps / motion_total
New (correct) per-env progress = time_steps / min(motion_total, max_env_steps)

Since per-env values transform by a constant factor 1/cap (cap = max_env_steps /
motion_total) AND torch.clamp at 1.0 is a no-op given time_steps <= max_env_steps,
the per-sample mean and the across-samples mean/std all transform exactly by 1/cap:

    new_mean = old_mean / cap
    new_std  = old_std  / cap

So no re-eval is needed for tracking motions whose motion length > episode length.

For tracking motions whose motion length <= episode length, cap >= 1.0 and the
correction is a no-op.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--dir", required=True, help="Directory of eval JSONs.")
  ap.add_argument(
    "--motion-total",
    type=int,
    required=True,
    help="Total frames in motion.npz (= command.motion.time_step_total).",
  )
  ap.add_argument(
    "--max-env-steps",
    type=int,
    default=500,
    help="Max env steps per episode (= ceil(episode_length_s / step_dt)). "
    "Tracking default is 500 (10 s / 0.02 s).",
  )
  ap.add_argument("--limit", type=int, default=100, help="Max JSONs to read.")
  args = ap.parse_args()

  ceiling = min(args.motion_total, args.max_env_steps)
  cap = ceiling / args.motion_total  # max value of old per-env progress

  files = sorted(Path(args.dir).glob("*.json"))[: args.limit]
  samples = [json.loads(f.read_text()) for f in files]
  old_progs = [s["progress_mean"] for s in samples if "progress_mean" in s]

  if not old_progs:
    print(f"[error] no 'progress_mean' field found in {args.dir}")
    return

  # Per-sample exact correction. Clamp guards against tiny numerical overshoot.
  new_progs = [min(p / cap, 1.0) for p in old_progs]

  old_mean = statistics.mean(old_progs)
  old_std = statistics.stdev(old_progs) if len(old_progs) > 1 else 0.0
  new_mean = statistics.mean(new_progs)
  new_std = statistics.stdev(new_progs) if len(new_progs) > 1 else 0.0

  print(f"dir            : {args.dir}")
  print(f"n_samples      : {len(old_progs)}")
  print(f"motion_total   : {args.motion_total}")
  print(f"max_env_steps  : {args.max_env_steps}")
  print(f"ceiling        : {ceiling}")
  print(f"cap (old max)  : {cap:.4f}")
  print(f"OLD progress   : {old_mean:.4f} ± {old_std:.4f}")
  print(f"NEW progress   : {new_mean:.4f} ± {new_std:.4f}")


if __name__ == "__main__":
  main()
