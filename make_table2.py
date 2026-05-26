#!/usr/bin/env python3
"""Generate paper Table 2: Ablation (embed_dim sweep) — Markdown or LaTeX.

Rows: motions.
Columns: embed_dim groups, each containing two cells: Prog and TE.

Cells: mean +/- std over diffusion samples per (motion, dim).
Bold: per-row per-metric winner across dims (max Prog, min TE).

Progress definition (so the column is comparable across task types):
  - Tracking : `progress_mean` JSON field (fraction of motion frames completed).
  - Walking  : `mean_episode_length_s / WALKING_MAX_EP_LEN_S`
               (fraction of max episode duration survived; default max = 20 s).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TaskType = Literal["tracking", "walking"]

WALKING_MAX_EP_LEN_S_DEFAULT = 20.0

# task_type -> (error_field, error_decimals)
TASK_SPEC: dict[TaskType, tuple[str, int]] = {
  "tracking": ("mpkpe", 3),
  "walking": ("error_vel_xy", 3),
}

PROG_DECIMALS = 3


@dataclass(frozen=True)
class MotionEntry:
  name: str
  task_type: TaskType
  dirs: tuple[Path, ...]  # one per dim, same order as dim_labels


@dataclass(frozen=True)
class DimStats:
  prog_mean: float
  prog_std: float
  err_mean: float
  err_std: float
  n: int


@dataclass(frozen=True)
class RowData:
  name: str
  is_walking: bool
  err_decimals: int
  per_dim: list[DimStats]
  best_prog_idx: int | None  # index where prog is max across dims; None if all NaN
  best_err_idx: int | None


def load_jsons(d: Path, limit: int) -> list[dict]:
  if not d.is_dir():
    return []
  files = sorted(d.glob("*.json"))
  if limit > 0:
    files = files[:limit]
  out: list[dict] = []
  for f in files:
    try:
      out.append(json.loads(f.read_text()))
    except json.JSONDecodeError:
      print(f"[warn] malformed JSON skipped: {f}", file=sys.stderr)
  return out


def stat_list(vals: list[float]) -> tuple[float, float, int]:
  if not vals:
    return float("nan"), float("nan"), 0
  if len(vals) == 1:
    return vals[0], 0.0, 1
  return statistics.mean(vals), statistics.stdev(vals), len(vals)


def progress_values(
  samples: list[dict], task_type: TaskType, walking_max: float
) -> list[float]:
  out: list[float] = []
  for s in samples:
    if task_type == "tracking":
      v = s.get("progress_mean")
    else:
      ep = s.get("mean_episode_length_s")
      v = ep / walking_max if ep is not None else None
    if v is not None and not math.isnan(v):
      out.append(v)
  return out


def error_values(samples: list[dict], field: str) -> list[float]:
  return [s[field] for s in samples if field in s]


def fmt(v: float, d: int) -> str:
  return "—" if v != v else f"{v:.{d}f}"


def compute_rows(
  entries: list[MotionEntry],
  dim_labels: list[str],
  limit: int,
  walking_max: float,
) -> list[RowData]:
  rows: list[RowData] = []
  for e in entries:
    err_field, err_d = TASK_SPEC[e.task_type]
    per_dim: list[DimStats] = []
    for dim_label, dir_ in zip(dim_labels, e.dirs):
      samples = load_jsons(dir_, limit)
      print(
        f"[info] {e.name:>10} embed={dim_label:<5} n={len(samples):>3}",
        file=sys.stderr,
      )
      prog_vals = progress_values(samples, e.task_type, walking_max)
      err_vals = error_values(samples, err_field)
      prog_m, prog_s, _ = stat_list(prog_vals)
      err_m, err_s, n = stat_list(err_vals)
      per_dim.append(
        DimStats(prog_mean=prog_m, prog_std=prog_s, err_mean=err_m, err_std=err_s, n=n)
      )

    prog_means = [d.prog_mean for d in per_dim]
    err_means = [d.err_mean for d in per_dim]
    valid_prog = [(i, v) for i, v in enumerate(prog_means) if v == v]
    valid_err = [(i, v) for i, v in enumerate(err_means) if v == v]
    best_prog_idx = max(valid_prog, key=lambda x: x[1])[0] if valid_prog else None
    best_err_idx = min(valid_err, key=lambda x: x[1])[0] if valid_err else None

    rows.append(
      RowData(
        name=e.name,
        is_walking=(e.task_type == "walking"),
        err_decimals=err_d,
        per_dim=per_dim,
        best_prog_idx=best_prog_idx,
        best_err_idx=best_err_idx,
      )
    )
  return rows


# --- Markdown rendering ---


def md_cell_pm(mean: float, std: float, n: int, d: int, bold: bool) -> str:
  if mean != mean:
    return "—"
  s = fmt(mean, d) if n <= 1 else f"{fmt(mean, d)} ± {fmt(std, d)}"
  return f"**{s}**" if bold else s


def render_markdown(rows: list[RowData], dim_labels: list[str]) -> str:
  header_top = [""] + sum(([f"embed={lbl}", ""] for lbl in dim_labels), [])
  header_bot = ["Motion"] + sum((["Prog ↑", "TE ↓"] for _ in dim_labels), [])
  data_rows: list[list[str]] = []
  for r in rows:
    name = r.name + (" ⋆" if r.is_walking else "")
    cells = [name]
    for i, ds in enumerate(r.per_dim):
      bold_p = i == r.best_prog_idx
      bold_e = i == r.best_err_idx
      cells.append(md_cell_pm(ds.prog_mean, ds.prog_std, ds.n, PROG_DECIMALS, bold_p))
      cells.append(md_cell_pm(ds.err_mean, ds.err_std, ds.n, r.err_decimals, bold_e))
    data_rows.append(cells)

  all_rows = [header_top, header_bot, *data_rows]
  widths = [max(len(r[i]) for r in all_rows) for i in range(len(header_bot))]
  fmt_row = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
  sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"

  out = [
    fmt_row(header_top),
    fmt_row(header_bot),
    sep,
    *(fmt_row(r) for r in data_rows),
  ]
  caption = (
    "\n**Caption.** All cells are mean ± std over diffusion samples. **Bold** marks "
    "the best dim per row per metric (max Prog, min TE). Prog = task progress "
    "fraction in [0, 1]: for tracking, fraction of reference motion frames "
    "completed; for walking (⋆), fraction of max episode duration (20 s) survived. "
    "TE = MPKPE (m) for tracking; mean linear velocity command error (m/s) for walking."
  )
  return "\n".join(out) + "\n" + caption


# --- LaTeX rendering (booktabs) ---


def tex_cell_pm(mean: float, std: float, n: int, d: int, bold: bool) -> str:
  if mean != mean:
    return "---"
  inner = fmt(mean, d) if n <= 1 else f"{fmt(mean, d)} \\pm {fmt(std, d)}"
  if bold:
    return f"$\\boldsymbol{{{inner}}}$"
  return f"${inner}$"


def render_latex(rows: list[RowData], dim_labels: list[str]) -> str:
  n_dims = len(dim_labels)
  col_spec = "l " + " ".join(["cc"] * n_dims)
  groups = " & ".join(rf"\multicolumn{{2}}{{c}}{{embed={lbl}}}" for lbl in dim_labels)
  cmid_pairs = [(2 + 2 * i, 3 + 2 * i) for i in range(n_dims)]
  cmid_line = " ".join(rf"\cmidrule(lr){{{a}-{b}}}" for a, b in cmid_pairs)
  metric_header = " & ".join([r"Prog $\uparrow$ & TE $\downarrow$"] * n_dims)

  lines = [
    r"% Requires \usepackage{booktabs, amsmath} in the preamble.",
    rf"\begin{{tabular}}{{{col_spec}}}",
    r"\toprule",
    rf" & {groups} \\",
    cmid_line,
    rf"Motion & {metric_header} \\",
    r"\midrule",
  ]
  for r in rows:
    name_tex = f"{r.name}$^{{\\star}}$" if r.is_walking else r.name
    cells = [name_tex]
    for i, ds in enumerate(r.per_dim):
      bold_p = i == r.best_prog_idx
      bold_e = i == r.best_err_idx
      cells.append(tex_cell_pm(ds.prog_mean, ds.prog_std, ds.n, PROG_DECIMALS, bold_p))
      cells.append(tex_cell_pm(ds.err_mean, ds.err_std, ds.n, r.err_decimals, bold_e))
    lines.append(" & ".join(cells) + r" \\")
  lines.extend([r"\bottomrule", r"\end{tabular}"])
  caption = (
    "% Caption suggestion: All cells are mean $\\pm$ std over diffusion samples. "
    "\\textbf{Bold} marks the best dim per row per metric (max Prog, min TE). "
    "Prog = task progress fraction in $[0, 1]$: for tracking, fraction of "
    "reference motion frames completed; for walking ($^{\\star}$), fraction of "
    "max episode duration (20 s) survived. TE = MPKPE (m) for tracking; "
    "mean linear velocity command error (m/s) for walking."
  )
  return "\n".join(lines) + "\n" + caption


def main():
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument(
    "--dim-labels",
    nargs="+",
    required=True,
    help="Column dim labels in order, e.g. --dim-labels 512 768 1024.",
  )
  ap.add_argument(
    "--motion",
    action="append",
    required=True,
    help=(
      "Repeatable. Format: 'name:tracking|walking:dir_dim1:dir_dim2:...'. "
      "Number of dirs must match --dim-labels count."
    ),
  )
  ap.add_argument(
    "--limit",
    type=int,
    default=100,
    help="Max JSONs per source (lexicographically first N). 0 = no limit. Default 100.",
  )
  ap.add_argument(
    "--walking-max-ep-len-s",
    type=float,
    default=WALKING_MAX_EP_LEN_S_DEFAULT,
    help="Max walking episode length (s) used to normalize Prog. Default 20.",
  )
  ap.add_argument(
    "--latex", action="store_true", help="Output LaTeX (booktabs) instead of Markdown."
  )
  args = ap.parse_args()

  n_dims = len(args.dim_labels)
  entries: list[MotionEntry] = []
  for s in args.motion:
    parts = s.split(":")
    if len(parts) != n_dims + 2:
      ap.error(
        f"--motion {s!r}: expected {n_dims + 2} colon-separated fields "
        f"(name, type, {n_dims} dirs), got {len(parts)}"
      )
    name, t, *dirs = parts
    if t not in ("tracking", "walking"):
      ap.error(f"--motion {s!r}: type must be tracking|walking, got {t!r}")
    entries.append(
      MotionEntry(name=name, task_type=t, dirs=tuple(Path(d) for d in dirs))
    )

  rows = compute_rows(
    entries,
    args.dim_labels,
    limit=args.limit,
    walking_max=args.walking_max_ep_len_s,
  )
  print(
    render_latex(rows, args.dim_labels)
    if args.latex
    else render_markdown(rows, args.dim_labels)
  )


if __name__ == "__main__":
  main()
