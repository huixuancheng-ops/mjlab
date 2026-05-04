"""Upload selected g1_flat_velocity_sweep_no_norm checkpoints to HuggingFace.

Renames files from {seed_dir}/model_{iter}.pt to seed_{seed}_model_{iter}.pt
and uploads only the iterations listed in REQUIRED_ITERS.
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError

SOURCE = Path("logs/rsl_rl/g1_flat_velocity_sweep_no_norm")
REQUIRED_ITERS = (4000, 4100, 4200, 4300, 4400, 4499)
SEED_DIR_RE = re.compile(
  r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_seed_(?P<seed>\d+)$"
)


def find_complete_dirs(root: Path) -> dict[int, Path]:
  """For each seed, return the latest directory that has all required ckpts."""
  by_seed: dict[int, list[tuple[str, Path]]] = defaultdict(list)
  for d in root.iterdir():
    if not d.is_dir():
      continue
    m = SEED_DIR_RE.match(d.name)
    if not m:
      continue
    seed = int(m.group("seed"))
    ts = m.group("ts")
    if all((d / f"model_{it}.pt").exists() for it in REQUIRED_ITERS):
      by_seed[seed].append((ts, d))

  return {seed: max(entries)[1] for seed, entries in by_seed.items()}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-id", required=True, help="e.g. username/diff_weight")
  parser.add_argument(
    "--subfolder",
    default="walking",
    help="Subfolder inside the repo for these checkpoints",
  )
  parser.add_argument(
    "--private", action="store_true", help="Create private dataset repo"
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would upload without actually uploading",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=100,
    help="Number of files per commit (higher = fewer commits, larger requests)",
  )
  args = parser.parse_args()

  if not SOURCE.exists():
    sys.exit(f"Source directory not found: {SOURCE}")

  complete = find_complete_dirs(SOURCE)
  if not complete:
    sys.exit("No complete seed directories found.")

  total_files = len(complete) * len(REQUIRED_ITERS)
  print(f"Found {len(complete)} complete seeds, {total_files} files to upload.")

  prefix = args.subfolder.strip("/") + "/" if args.subfolder else ""

  if args.dry_run:
    print(f"  Target: {args.repo_id}/{prefix}")
    for seed in sorted(complete)[:3]:
      print(f"  seed {seed} -> {complete[seed].name}")
      for it in REQUIRED_ITERS:
        print(f"    {prefix}seed_{seed}_model_{it}.pt")
    print(f"  ... ({len(complete) - 3} more seeds)")
    return

  api = HfApi()
  create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

  # Skip files already on HF for resumable uploads.
  existing = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
  print(f"  {len(existing)} files already on HF, skipping those that match.")

  operations: list[CommitOperationAdd] = []
  for seed in sorted(complete):
    src_dir = complete[seed]
    for it in REQUIRED_ITERS:
      path_in_repo = f"{prefix}seed_{seed}_model_{it}.pt"
      if path_in_repo in existing:
        continue
      operations.append(
        CommitOperationAdd(
          path_in_repo=path_in_repo,
          path_or_fileobj=str(src_dir / f"model_{it}.pt"),
        )
      )

  total_files = len(operations)
  print(f"  Remaining to upload: {total_files} files.")
  if total_files == 0:
    print("Nothing to upload.")
    return

  def commit_with_retry(batch, batch_idx):
    while True:
      try:
        api.create_commit(
          repo_id=args.repo_id,
          repo_type="dataset",
          operations=batch,
          commit_message=f"Upload checkpoints batch {batch_idx}",
        )
        return
      except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 429:
          wait = 15 * 60  # commit rate limit is hourly; sleep 15 min then retry
          print(f"  Rate limited (429). Sleeping {wait}s then retrying...")
          time.sleep(wait)
        else:
          raise

  done = 0
  for i in range(0, len(operations), args.batch_size):
    batch = operations[i : i + args.batch_size]
    commit_with_retry(batch, i // args.batch_size + 1)
    done += len(batch)
    print(
      f"  [{done}/{total_files}] committed batch "
      f"{i // args.batch_size + 1}/{(total_files + args.batch_size - 1) // args.batch_size}"
    )

  print(f"Done. Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
  main()
