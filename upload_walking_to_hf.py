"""Upload g1_flat_velocity_sweep iter-3499 checkpoints to HuggingFace.

Companion to upload_dance_hold_norm_to_hf.py, but for the walking sweep
(g1_flat_velocity_sweep). Differences:

  * Only the iter-3499 checkpoint per seed.
  * Uploads to a separate subfolder (default: analysis/walking).
  * Caps total uploads at --max-seeds (default 100) distinct seeds. If the
    sweep contains more, the lowest-numbered seeds are selected.

Uses batch commits (CommitOperationAdd + create_commit) to stay under HF's
per-account commit rate limit. Resume semantics:

  * Cross-run: every invocation re-lists the repo and rebuilds the operations
    list, skipping files already present. Ctrl-C / OOM / process death is
    recovered by simply rerunning the script.
  * Within a batch: create_commit pre-uploads LFS objects by SHA256. If the
    network drops mid-upload, the next attempt resumes from the last completed
    multipart chunk; if all LFS uploads finished but the final commit RPC
    failed, the retry reuses the already-uploaded objects and only re-submits
    the commit. This is native huggingface_hub behavior, not a wrapper.
  * Per-batch retry: exponential backoff on 5xx / connection errors, 15-min
    sleep on 429, unlimited attempts (the script will keep trying until the
    batch either succeeds or hits a non-retryable error).
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError

SOURCE = Path("/home/huixuan_cheng/mjlab/logs/rsl_rl/g1_flat_velocity_sweep")
REQUIRED_ITERS = (3499,)
DEFAULT_MAX_SEEDS = 100
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


def commit_with_retry(api, repo_id, batch, idx, message):
  """Commit a batch with backoff on rate limit / transient network errors."""
  backoff = 30
  attempt = 0
  while True:
    attempt += 1
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
        print(
          f"  [batch {idx}] 429 rate-limited (attempt {attempt}), sleeping 15min..."
        )
        time.sleep(15 * 60)
      elif code is not None and 500 <= code < 600:
        print(
          f"  [batch {idx}] HTTP {code} (attempt {attempt}), sleeping {backoff}s..."
        )
        time.sleep(backoff)
        backoff = min(backoff * 2, 600)
      else:
        raise
    except (ConnectionError, TimeoutError, OSError) as e:
      print(
        f"  [batch {idx}] network error ({e}) (attempt {attempt}), sleeping {backoff}s..."
      )
      time.sleep(backoff)
      backoff = min(backoff * 2, 600)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--repo-id",
    default="huixuanc/diff_weight",
    help="e.g. username/diff_weight",
  )
  parser.add_argument(
    "--source",
    default=str(SOURCE),
    help=(
      "Local sweep directory to scan "
      "(default: /home/huixuan_cheng/mjlab/logs/rsl_rl/g1_flat_velocity_sweep)"
    ),
  )
  parser.add_argument(
    "--subfolder",
    default="analysis/walking",
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
    default=25,
    help="Files per commit (smaller = finer-grained resume; larger = fewer commits)",
  )
  parser.add_argument(
    "--max-seeds",
    type=int,
    default=DEFAULT_MAX_SEEDS,
    help=(
      "Maximum number of distinct seeds to upload (default 100). "
      "If the sweep contains more, the lowest-numbered seeds are chosen."
    ),
  )
  args = parser.parse_args()

  source = Path(args.source)
  if not source.exists():
    sys.exit(f"Source directory not found: {source}")

  complete = find_complete_dirs(source)
  if not complete:
    sys.exit(f"No seed dirs in {source} contain all of {REQUIRED_ITERS}.")

  if len(complete) > args.max_seeds:
    selected_seeds = sorted(complete)[: args.max_seeds]
    print(
      f"Found {len(complete)} complete seeds; capping at {args.max_seeds} "
      f"(lowest-numbered): seeds {selected_seeds[0]}..{selected_seeds[-1]}."
    )
    complete = {seed: complete[seed] for seed in selected_seeds}

  total_files = len(complete) * len(REQUIRED_ITERS)
  print(f"Selected {len(complete)} seeds, {total_files} files to upload.")

  prefix = args.subfolder.strip("/") + "/" if args.subfolder else ""

  if args.dry_run:
    print(f"  Target: {args.repo_id}/{prefix}")
    for seed in sorted(complete):
      print(f"  seed {seed} -> {complete[seed].name}")
      for it in REQUIRED_ITERS:
        print(f"    {prefix}seed_{seed}_model_{it}.pt")
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
      f"Upload walking ckpts batch {idx}/{n_batches}",
    )
    done += len(batch)
    print(f"  [{done}/{total}] committed batch {idx}/{n_batches}")

  print(f"Done. Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
  main()
