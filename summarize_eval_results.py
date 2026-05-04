"""Aggregate per-sample evaluation JSONs into a single summary report.

For each metric (mpkpe, r_mpkpe, joint_vel_error, ee_pos_error, ee_ori_error,
success_rate), computes mean / std / median / p90 / p99, plus min / max with the
sample names that produced them. Output is a plain-text report.

Usage:
  uv run python summarize_eval_results.py <results_dir> [--output FILE]
                                           [--rank-by mpkpe]
                                           [--top N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

METRICS = (
  "mpkpe",
  "r_mpkpe",
  "joint_vel_error",
  "ee_pos_error",
  "ee_ori_error",
  "success_rate",
)
# Metrics where higher = better; everything else lower = better.
HIGHER_IS_BETTER = {"success_rate"}


def load_results(results_dir: Path) -> list[tuple[str, dict[str, float]]]:
  files = sorted(results_dir.glob("*.json"))
  results = []
  for f in files:
    try:
      with open(f) as fh:
        data = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
      print(f"[WARN] Skipping {f.name}: {e}", file=sys.stderr)
      continue
    results.append((f.stem, data))
  return results


def percentile(arr: np.ndarray, p: float) -> float:
  return float(np.percentile(arr, p))


def format_stats_table(
  results: list[tuple[str, dict[str, float]]],
) -> tuple[str, dict[str, np.ndarray]]:
  rows = []
  header = f"{'metric':<18} {'mean':>10} {'std':>10} {'median':>10} {'p90':>10} {'p99':>10} {'min':>10} {'max':>10}"
  rows.append(header)
  rows.append("-" * len(header))
  arrays: dict[str, np.ndarray] = {}
  for m in METRICS:
    vals = np.array([r[m] for _, r in results if m in r], dtype=np.float64)
    if vals.size == 0:
      continue
    arrays[m] = vals
    rows.append(
      f"{m:<18} {vals.mean():>10.4f} {vals.std():>10.4f} "
      f"{percentile(vals, 50):>10.4f} {percentile(vals, 90):>10.4f} "
      f"{percentile(vals, 99):>10.4f} {vals.min():>10.4f} {vals.max():>10.4f}"
    )
  return "\n".join(rows), arrays


def format_extremes(
  results: list[tuple[str, dict[str, float]]],
  arrays: dict[str, np.ndarray],
) -> str:
  names = [n for n, _ in results]
  lines = []
  for m, vals in arrays.items():
    best_idx = int(vals.argmax() if m in HIGHER_IS_BETTER else vals.argmin())
    worst_idx = int(vals.argmin() if m in HIGHER_IS_BETTER else vals.argmax())
    direction = "(higher better)" if m in HIGHER_IS_BETTER else "(lower better)"
    lines.append(f"{m} {direction}:")
    lines.append(f"  best  = {vals[best_idx]:.4f}  ->  {names[best_idx]}")
    lines.append(f"  worst = {vals[worst_idx]:.4f}  ->  {names[worst_idx]}")
  return "\n".join(lines)


def _ranked_rows(
  results: list[tuple[str, dict[str, float]]],
  rank_by: str,
) -> tuple[list[str], list[tuple[float, str, dict[str, float]]]]:
  metric_cols = [m for m in METRICS if any(m in r for _, r in results)]
  reverse = rank_by in HIGHER_IS_BETTER
  scored = [(r.get(rank_by, float("nan")), n, r) for n, r in results]
  scored = [s for s in scored if not np.isnan(s[0])]
  scored.sort(key=lambda x: x[0], reverse=reverse)
  return metric_cols, scored


def format_top_n(
  results: list[tuple[str, dict[str, float]]],
  rank_by: str,
  top: int,
) -> str:
  if rank_by not in METRICS:
    return f"[WARN] Unknown rank-by metric: {rank_by}"
  metric_cols, scored = _ranked_rows(results, rank_by)
  reverse = rank_by in HIGHER_IS_BETTER
  lines = [f"Top {top} samples by {rank_by} ({'desc' if reverse else 'asc'}):"]
  header = f"{'rank':>4}  {'sample':<40}  " + "  ".join(f"{m:>12}" for m in metric_cols)
  lines.append(header)
  lines.append("-" * len(header))
  for i, (_, name, r) in enumerate(scored[:top], 1):
    cells = "  ".join(f"{r.get(m, float('nan')):>12.4f}" for m in metric_cols)
    lines.append(f"{i:>4}  {name:<40}  {cells}")
  return "\n".join(lines)


def format_full_table(
  results: list[tuple[str, dict[str, float]]],
  rank_by: str,
) -> str:
  if rank_by not in METRICS:
    rank_by = "mpkpe"
  metric_cols, scored = _ranked_rows(results, rank_by)
  lines = [f"All samples (sorted by {rank_by}):"]
  header = f"{'rank':>4}  {'sample':<40}  " + "  ".join(f"{m:>12}" for m in metric_cols)
  lines.append(header)
  lines.append("-" * len(header))
  for i, (_, name, r) in enumerate(scored, 1):
    cells = "  ".join(f"{r.get(m, float('nan')):>12.4f}" for m in metric_cols)
    lines.append(f"{i:>4}  {name:<40}  {cells}")
  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("results_dir", type=str, help="Directory of per-sample JSONs")
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output txt path (default: <results_dir>/summary_rank_by_<metric>_top<N>.txt)",
  )
  parser.add_argument(
    "--rank-by",
    type=str,
    default="mpkpe",
    help="Metric to rank top-N samples by (default: mpkpe)",
  )
  parser.add_argument(
    "--top", type=int, default=10, help="How many top samples to list (default: 10)"
  )
  parser.add_argument(
    "--cleanup",
    action="store_true",
    help="Delete all per-sample *.json files in results-dir after writing summary.",
  )
  args = parser.parse_args()

  results_dir = Path(args.results_dir)
  if not results_dir.is_dir():
    sys.exit(f"Not a directory: {results_dir}")

  results = load_results(results_dir)
  if not results:
    sys.exit(f"No JSON results found in {results_dir}")

  default_name = f"summary_rank_by_{args.rank_by}_top{args.top}.txt"
  output_path = Path(args.output) if args.output else results_dir / default_name

  stats_table, arrays = format_stats_table(results)
  extremes = format_extremes(results, arrays)
  top_n = format_top_n(results, args.rank_by, args.top)
  full_table = format_full_table(results, args.rank_by)

  report = "\n\n".join(
    [
      f"Evaluation summary: {results_dir}",
      f"Samples: {len(results)}",
      "Per-metric statistics:",
      stats_table,
      "Best / worst sample per metric:",
      extremes,
      top_n,
      full_table,
    ]
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(report + "\n")
  print(report)
  print(f"\n[INFO] Saved to {output_path}")

  if args.cleanup:
    deleted = 0
    for f in results_dir.glob("*.json"):
      f.unlink()
      deleted += 1
    print(f"[INFO] Cleanup: removed {deleted} per-sample JSON files in {results_dir}")


if __name__ == "__main__":
  main()
