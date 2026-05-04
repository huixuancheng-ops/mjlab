"""Upload an existing motion .npz to W&B as an artifact and link it into the
"motions" registry, matching the layout that mjlab tracking tasks consume via
`--registry-name <entity>/motions/<name>:<version>`.

This mirrors what `csv_to_npz.py` does at the end of its pipeline, but for a
.npz that already exists on disk (e.g. the output of `extend_motion_standing.py`).

Usage:
  uv run python upload_motion_to_wandb.py \
    --motion-file /path/to/motion.npz \
    --name dance1s2_ramp05_hold2s \
    [--project csv_to_npz] \
    [--description "..."] \
    [--registry motions]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb


def upload(
  motion_file: Path,
  name: str,
  project: str,
  registry: str,
  description: str | None,
) -> None:
  run = wandb.init(project=project, name=name, job_type="upload-motion")
  if run is None:
    raise RuntimeError("wandb.init returned None — check WANDB_API_KEY / network")
  print(f"[INFO] W&B run: {run.url}")

  # Always store the file inside the artifact as "motion.npz" — mjlab's loader
  # (e.g. evaluate.py / train.py) does `Path(artifact.download()) / "motion.npz"`.
  artifact = wandb.Artifact(name=name, type=registry, description=description)
  artifact.add_file(str(motion_file), name="motion.npz")
  run.log_artifact(artifact)
  artifact.wait()  # Block until upload completes so link_artifact has a version.

  target = f"wandb-registry-{registry}/{name}"
  run.link_artifact(artifact=artifact, target_path=target)

  print(f"[INFO] Uploaded {motion_file} -> registry: {registry}/{name}")
  print(f"[INFO] Linked at: {target}")
  if run.entity:
    print(f"[INFO] Use in train: --registry-name {run.entity}/{registry}/{name}:latest")
  run.finish()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--motion-file", type=str, required=True, help="Path to motion .npz to upload"
  )
  parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Collection / artifact name in the registry (e.g. 'dance1s2_ramp05_hold2s')",
  )
  parser.add_argument(
    "--project",
    type=str,
    default="csv_to_npz",
    help="W&B project for the uploader run (default: csv_to_npz, matches mjlab convention)",
  )
  parser.add_argument(
    "--registry",
    type=str,
    default="motions",
    help="Registry name (default: motions, what mjlab tracking expects)",
  )
  parser.add_argument(
    "--description",
    type=str,
    default=None,
    help="Optional description recorded with the artifact",
  )
  args = parser.parse_args()

  motion_file = Path(args.motion_file)
  if not motion_file.is_file():
    sys.exit(f"Motion file not found: {motion_file}")
  if motion_file.suffix != ".npz":
    print(f"[WARN] {motion_file} does not end with .npz — proceeding anyway.")

  upload(
    motion_file=motion_file,
    name=args.name,
    project=args.project,
    registry=args.registry,
    description=args.description,
  )


if __name__ == "__main__":
  main()
