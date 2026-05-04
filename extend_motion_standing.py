"""Extend a tracking motion .npz by holding its final pose for extra seconds.

Use case: the input motion already ends with a transition into a standing pose
(e.g. dance1s2 with a 0.5s interpolation to standing). To make the policy hold
that pose longer, this script appends frames that repeat the last pose with
zero velocity (so the reference truly represents "stand still here").

Optional `--ramp-seconds` inserts a smoothing segment between the original
motion and the held pose: positions are held at the last pose, while velocity
fields linearly decay from their last-frame values down to zero. This avoids
a sharp velocity discontinuity at the boundary when the input motion still has
non-zero residual velocity at its final frame.

Layout matches mjlab tracking motion.npz:
  fps              (1,)
  joint_pos        (T, n_joints)
  joint_vel        (T, n_joints)
  body_pos_w       (T, n_bodies, 3)
  body_quat_w      (T, n_bodies, 4)
  body_lin_vel_w   (T, n_bodies, 3)
  body_ang_vel_w   (T, n_bodies, 3)

Usage:
  uv run python extend_motion_standing.py \
    --input  /home/.../dance1s2:v0/motion.npz \
    --output /home/.../dance1s2_hold2s/motion.npz \
    --hold-seconds 2.0 \
    --ramp-seconds 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

POS_KEYS: tuple[str, ...] = ("joint_pos", "body_pos_w", "body_quat_w")
VEL_KEYS: tuple[str, ...] = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")


def _build_segments(
  arr: np.ndarray,
  ramp_frames: int,
  hold_frames: int,
  is_velocity: bool,
) -> tuple[np.ndarray, np.ndarray]:
  """Construct ramp and hold segments to append after `arr`.

  Position fields: both segments hold the last pose constant.
  Velocity fields: ramp linearly decays last value to zero, hold is zero.
  """
  trailing = arr.shape[1:]
  last = arr[-1]

  if is_velocity:
    if ramp_frames > 0:
      # blend goes from (1 - 1/N) down to 0: smooth but starts already slightly
      # below the original last value to avoid a flat plateau at the boundary.
      blend = np.linspace(1.0, 0.0, ramp_frames + 1, dtype=arr.dtype)[1:]
      ramp = last * blend.reshape((-1,) + (1,) * (arr.ndim - 1))
    else:
      ramp = np.empty((0,) + trailing, dtype=arr.dtype)
    hold = np.zeros((hold_frames,) + trailing, dtype=arr.dtype)
  else:
    held = np.broadcast_to(last, (1,) + trailing)
    ramp = np.broadcast_to(held, (ramp_frames,) + trailing).astype(arr.dtype, copy=True)
    hold = np.broadcast_to(held, (hold_frames,) + trailing).astype(arr.dtype, copy=True)
  return ramp, hold


def extend_motion(
  input_path: Path,
  output_path: Path,
  hold_seconds: float,
  ramp_seconds: float,
) -> None:
  data = np.load(str(input_path), allow_pickle=True)
  fps = float(data["fps"][0])
  hold_frames = int(round(hold_seconds * fps))
  ramp_frames = int(round(ramp_seconds * fps))
  if hold_frames < 0 or ramp_frames < 0:
    raise ValueError("hold_seconds and ramp_seconds must be non-negative")
  if hold_frames + ramp_frames == 0:
    raise ValueError("Nothing to do: both hold_seconds and ramp_seconds are zero")

  out: dict[str, np.ndarray] = {"fps": np.array([fps], dtype=np.float64)}

  for key in POS_KEYS + VEL_KEYS:
    if key not in data.files:
      raise KeyError(f"Missing key in input: {key}")
    arr = np.asarray(data[key])
    ramp, hold = _build_segments(arr, ramp_frames, hold_frames, key in VEL_KEYS)
    out[key] = np.concatenate([arr, ramp, hold], axis=0).astype(arr.dtype, copy=False)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(str(output_path), **out)

  in_T = int(data["joint_pos"].shape[0])
  out_T = int(out["joint_pos"].shape[0])
  print(f"[INFO] Input:  {input_path}  ({in_T} frames, {in_T / fps:.2f}s @ {fps}fps)")
  print(
    f"[INFO] Output: {output_path}  ({out_T} frames, {out_T / fps:.2f}s @ {fps}fps)"
  )
  print(
    f"[INFO] Appended {ramp_frames} ramp frames ({ramp_seconds:.2f}s, velocity "
    f"decays to 0) and {hold_frames} hold frames ({hold_seconds:.2f}s, zero "
    "velocity)."
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, required=True, help="Input motion .npz")
  parser.add_argument("--output", type=str, required=True, help="Output motion .npz")
  parser.add_argument(
    "--hold-seconds",
    type=float,
    default=1.0,
    help="Seconds of zero-velocity hold to append (default: 1.0)",
  )
  parser.add_argument(
    "--ramp-seconds",
    type=float,
    default=0.0,
    help="Seconds of velocity-decay ramp inserted before the hold (default: 0.0)",
  )
  args = parser.parse_args()

  in_path = Path(args.input)
  if not in_path.is_file():
    sys.exit(f"Input not found: {in_path}")
  extend_motion(in_path, Path(args.output), args.hold_seconds, args.ramp_seconds)


if __name__ == "__main__":
  main()
