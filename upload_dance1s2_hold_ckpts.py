"""Upload dance-hold checkpoints from g1_tracking_dance1s2_hold_sweep to HF.

For each seed directory `{ts}_seed_{N}`, only seeds that contain ALL of
REQUIRED_ITERS are uploaded. Files are renamed to `seed_{N}_model_{it}.pt`
under the target subfolder. Already-uploaded files are skipped (resumable).
Uploads stream directly from disk via CommitOperationAdd (no local staging).

Usage:
  uv run python upload_dance1s2_hold_ckpts.py \
    --repo-id huixuanc/diff_weight \
    --subfolder dance_hold
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError

SOURCE = Path("logs/rsl_rl/g1_tracking_dance1s2_hold_sweep")
REQUIRED_ITERS = (3800, 3900, 4000, 4100, 4200, 4300, 4400, 4499)
SEED_RE = re.compile(r"_seed_(?P<seed>\d+)$")


def find_complete_dirs(root: Path) -> dict[int, Path]:
  """For each seed, return the (latest) dir containing all required ckpts."""
  by_seed: dict[int, list[tuple[str, Path]]] = defaultdict(list)
  for d in root.iterdir():
    if not d.is_dir():
      continue
    m = SEED_RE.search(d.name)
    if not m:
      continue
    seed = int(m.group("seed"))
    if all((d / f"model_{it}.pt").exists() for it in REQUIRED_ITERS):
      by_seed[seed].append((d.name, d))
  return {seed: max(entries)[1] for seed, entries in by_seed.items()}


def commit_with_retry(api, repo_id, batch, idx, message):
  """Commit a batch with backoff on rate limit / transient network errors."""
  backoff = 30
  while True:
    try:
      api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=batch,
        commit_message=message,
      )
      return
    except HfHubHTTPError as e:
      code = e.response.status_code if e.response is not None else None
      if code == 429:
        print(f"  [batch {idx}] rate limited (429), sleeping 15min...")
        time.sleep(15 * 60)
      elif code is not None and 500 <= code < 600:
        print(f"  [batch {idx}] server error {code}, sleeping {backoff}s...")
        time.sleep(backoff)
        backoff = min(backoff * 2, 600)
      else:
        raise
    except (ConnectionError, TimeoutError, OSError) as e:
      print(f"  [batch {idx}] network error ({e}), sleeping {backoff}s...")
      time.sleep(backoff)
      backoff = min(backoff * 2, 600)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-id", default="huixuanc/diff_weight")
  parser.add_argument("--subfolder", default="dance_hold")
  parser.add_argument("--private", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--batch-size", type=int, default=50)
  args = parser.parse_args()

  if not SOURCE.exists():
    sys.exit(f"Source directory not found: {SOURCE}")

  complete = find_complete_dirs(SOURCE)
  if not complete:
    sys.exit("No seeds have all required iterations yet.")

  total_files = len(complete) * len(REQUIRED_ITERS)
  print(
    f"Found {len(complete)} complete seeds, {total_files} files candidate for upload."
  )

  prefix = args.subfolder.strip("/") + "/" if args.subfolder else ""

  if args.dry_run:
    print(f"  Target: {args.repo_id}/{prefix}")
    for seed in sorted(complete)[:3]:
      print(f"  seed {seed} <- {complete[seed].name}")
      for it in REQUIRED_ITERS:
        print(f"    {prefix}seed_{seed}_model_{it}.pt")
    if len(complete) > 3:
      print(f"  ... ({len(complete) - 3} more seeds)")
    return

  api = HfApi()
  create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

  existing = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
  print(f"  {len(existing)} files already on HF, will skip duplicates.")

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

  total = len(operations)
  print(f"  Remaining to upload: {total} files.")
  if total == 0:
    print("Nothing to upload.")
    return

  n_batches = (total + args.batch_size - 1) // args.batch_size
  done = 0
  for i in range(0, total, args.batch_size):
    batch = operations[i : i + args.batch_size]
    idx = i // args.batch_size + 1
    commit_with_retry(
      api,
      args.repo_id,
      batch,
      idx,
      f"Upload dance_hold ckpts batch {idx}/{n_batches}",
    )
    done += len(batch)
    print(f"  [{done}/{total}] committed batch {idx}/{n_batches}")

  print(f"Done. Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
  main()
