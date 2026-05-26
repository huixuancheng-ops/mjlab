#!/usr/bin/env python3
"""Generate paper Table 1: Dataset vs DiT (mean / best) — Markdown or LaTeX.

Cells:
  - Dataset           : mean +/- std over dataset eval JSONs (multiple oracle ckpts)
  - DiT-1024 (mean)   : mean +/- std over 100 diffusion samples
  - DiT-1024 (best)   : per-metric extreme over 100 samples (max for succ, min for err)
                        Bolded when it beats Dataset-mean.

Tracking error metric : MPKPE (m)            for tracking motions
                      : error_vel_xy (m/s)   for walking (velocity task)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TaskType = Literal["tracking", "walking"]

# task_type -> (success_field, error_field, error_decimals)
TASK_SPEC: dict[TaskType, tuple[str, str, int]] = {
  "tracking": ("success_rate", "mpkpe", 3),
  "walking": ("success_rate", "error_vel_xy", 3),
}

SUCC_DECIMALS = 3


@dataclass(frozen=True)
class MotionEntry:
  name: str
  task_type: TaskType
  dataset_dir: Path
  dit_dir: Path


@dataclass(frozen=True)
class RowData:
  name: str
  is_walking: bool
  err_decimals: int
  ds_succ_m: float
  ds_succ_s: float
  ds_n: int
  ds_err_m: float
  ds_err_s: float
  dit_succ_m: float
  dit_succ_s: float
  dit_n: int
  dit_err_m: float
  dit_err_s: float
  dit_succ_best: float
  succ_beats: bool
  dit_err_best: float
  err_beats: bool


def load_jsons(d: Path, limit: int) -> list[dict]:
  if not d.is_dir():
    return []
  files = sorted(d.glob("*.json"))
  if limit > 0:
    files = files[:limit]
  out = []
  for f in files:
    try:
      out.append(json.loads(f.read_text()))
    except json.JSONDecodeError:
      print(f"[warn] malformed JSON skipped: {f}", file=sys.stderr)
  return out


def stats(samples: list[dict], field: str) -> tuple[float, float, int]:
  vals = [s[field] for s in samples if field in s]
  if not vals:
    return float("nan"), float("nan"), 0
  if len(vals) == 1:
    return vals[0], 0.0, 1
  return statistics.mean(vals), statistics.stdev(vals), len(vals)


def best(samples: list[dict], field: str, higher_is_better: bool) -> float:
  vals = [s[field] for s in samples if field in s]
  if not vals:
    return float("nan")
  return max(vals) if higher_is_better else min(vals)


def fmt(v: float, d: int) -> str:
  return "—" if v != v else f"{v:.{d}f}"


def compute_rows(entries: list[MotionEntry], limit: int) -> list[RowData]:
  rows: list[RowData] = []
  for e in entries:
    succ_f, err_f, err_d = TASK_SPEC[e.task_type]
    ds = load_jsons(e.dataset_dir, limit)
    dit = load_jsons(e.dit_dir, limit)
    print(
      f"[info] {e.name:>10}: dataset={len(ds):>3}  dit={len(dit):>3}", file=sys.stderr
    )

    ds_succ_m, ds_succ_s, ds_n = stats(ds, succ_f)
    ds_err_m, ds_err_s, _ = stats(ds, err_f)
    dit_succ_m, dit_succ_s, dit_n = stats(dit, succ_f)
    dit_err_m, dit_err_s, _ = stats(dit, err_f)

    dit_succ_best = best(dit, succ_f, higher_is_better=True)
    dit_err_best = best(dit, err_f, higher_is_better=False)

    succ_beats = (ds_succ_m == ds_succ_m) and (dit_succ_best > ds_succ_m)
    err_beats = (ds_err_m == ds_err_m) and (dit_err_best < ds_err_m)

    rows.append(
      RowData(
        name=e.name,
        is_walking=(e.task_type == "walking"),
        err_decimals=err_d,
        ds_succ_m=ds_succ_m,
        ds_succ_s=ds_succ_s,
        ds_n=ds_n,
        ds_err_m=ds_err_m,
        ds_err_s=ds_err_s,
        dit_succ_m=dit_succ_m,
        dit_succ_s=dit_succ_s,
        dit_n=dit_n,
        dit_err_m=dit_err_m,
        dit_err_s=dit_err_s,
        dit_succ_best=dit_succ_best,
        succ_beats=succ_beats,
        dit_err_best=dit_err_best,
        err_beats=err_beats,
      )
    )
  return rows


# --- Markdown rendering ---


def md_cell_pm(mean: float, std: float, n: int, d: int) -> str:
  if mean != mean:
    return "—"
  if n <= 1:
    return fmt(mean, d)
  return f"{fmt(mean, d)} ± {fmt(std, d)}"


def md_cell_best(v: float, d: int, bold: bool) -> str:
  if v != v:
    return "—"
  s = fmt(v, d)
  return f"**{s}**" if bold else s


def render_markdown(rows: list[RowData]) -> str:
  header_top = ["", "Dataset", "", "DiT-1024 (mean)", "", "DiT-1024 (best)", ""]
  header_bot = ["Motion", "Succ ↑", "TE ↓", "Succ ↑", "TE ↓", "Succ ↑", "TE ↓"]
  data_rows: list[list[str]] = []
  for r in rows:
    name = r.name + (" ⋆" if r.is_walking else "")
    data_rows.append(
      [
        name,
        md_cell_pm(r.ds_succ_m, r.ds_succ_s, r.ds_n, SUCC_DECIMALS),
        md_cell_pm(r.ds_err_m, r.ds_err_s, r.ds_n, r.err_decimals),
        md_cell_pm(r.dit_succ_m, r.dit_succ_s, r.dit_n, SUCC_DECIMALS),
        md_cell_pm(r.dit_err_m, r.dit_err_s, r.dit_n, r.err_decimals),
        md_cell_best(r.dit_succ_best, SUCC_DECIMALS, r.succ_beats),
        md_cell_best(r.dit_err_best, r.err_decimals, r.err_beats),
      ]
    )

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
    "\n**Caption.** Dataset and DiT-mean cells are reported as mean ± std. "
    "DiT-best (best of 100 samples) is per-metric extreme. **Bold** marks DiT-best "
    "beating Dataset-mean. TE = Tracking Error: MPKPE (m) for tracking motions; "
    "mean linear-velocity command error (m/s) for walking (⋆)."
  )
  return "\n".join(out) + "\n" + caption


# --- LaTeX rendering (booktabs) ---


def tex_cell_pm(mean: float, std: float, n: int, d: int) -> str:
  if mean != mean:
    return "---"
  if n <= 1:
    return f"${fmt(mean, d)}$"
  return f"${fmt(mean, d)} \\pm {fmt(std, d)}$"


def tex_cell_best(v: float, d: int, bold: bool) -> str:
  if v != v:
    return "---"
  s = fmt(v, d)
  return f"\\textbf{{{s}}}" if bold else s


def render_latex(rows: list[RowData]) -> str:
  lines = [
    r"% Requires \usepackage{booktabs} in the preamble.",
    r"\begin{tabular}{l cc cc cc}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Dataset} & \multicolumn{2}{c}{DiT-1024 (mean)} & \multicolumn{2}{c}{DiT-1024 (best)} \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
    r"Motion & Succ $\uparrow$ & TE $\downarrow$ & Succ $\uparrow$ & TE $\downarrow$ & Succ $\uparrow$ & TE $\downarrow$ \\",
    r"\midrule",
  ]
  for r in rows:
    name_tex = f"{r.name}$^{{\\star}}$" if r.is_walking else r.name
    cells = [
      name_tex,
      tex_cell_pm(r.ds_succ_m, r.ds_succ_s, r.ds_n, SUCC_DECIMALS),
      tex_cell_pm(r.ds_err_m, r.ds_err_s, r.ds_n, r.err_decimals),
      tex_cell_pm(r.dit_succ_m, r.dit_succ_s, r.dit_n, SUCC_DECIMALS),
      tex_cell_pm(r.dit_err_m, r.dit_err_s, r.dit_n, r.err_decimals),
      tex_cell_best(r.dit_succ_best, SUCC_DECIMALS, r.succ_beats),
      tex_cell_best(r.dit_err_best, r.err_decimals, r.err_beats),
    ]
    lines.append(" & ".join(cells) + r" \\")
  lines.extend([r"\bottomrule", r"\end{tabular}"])
  caption = (
    "% Caption suggestion: Dataset and DiT-mean cells are reported as mean $\\pm$ std. "
    "DiT-best (best of 100 samples) is per-metric extreme. \\textbf{Bold} marks DiT-best "
    "beating Dataset-mean. TE = Tracking Error: MPKPE (m) for tracking motions; "
    "mean linear-velocity command error (m/s) for walking ($^{\\star}$)."
  )
  return "\n".join(lines) + "\n" + caption


def parse_entry(s: str) -> MotionEntry:
  parts = s.split(":")
  if len(parts) != 4:
    raise argparse.ArgumentTypeError(
      f"Expected 'name:tracking|walking:dataset_dir:dit_dir', got {s!r}"
    )
  name, t, ds, dit = parts
  if t not in ("tracking", "walking"):
    raise argparse.ArgumentTypeError(f"task type must be tracking|walking, got {t!r}")
  return MotionEntry(name=name, task_type=t, dataset_dir=Path(ds), dit_dir=Path(dit))


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "--motion",
    action="append",
    required=True,
    type=parse_entry,
    help="Repeatable. Format: 'name:tracking|walking:dataset_dir:dit_dir'",
  )
  ap.add_argument(
    "--limit",
    type=int,
    default=100,
    help="Max JSONs per source (lexicographically first N). 0 = no limit. Default 100.",
  )
  ap.add_argument(
    "--latex",
    action="store_true",
    help="Output LaTeX (booktabs) instead of Markdown.",
  )
  args = ap.parse_args()
  rows = compute_rows(args.motion, limit=args.limit)
  print(render_latex(rows) if args.latex else render_markdown(rows))


if __name__ == "__main__":
  main()
