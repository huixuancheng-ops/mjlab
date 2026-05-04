"""Convert mjlab actor checkpoints (.pt) into ONNX by cloning a reference ONNX
and surgically replacing the policy MLP weights.

Why clone-and-swap (surgical):
  Tracking ONNX files bundle motion reference data, time_step input, and a
  multi-output graph. Recreating that from Python would require loading the
  full mjlab env. Cloning the reference ONNX preserves everything (motion
  buffers, graph topology, metadata, IO names) and only the policy MLP
  weights get overwritten with the .pt sample's weights.

This works for any reference ONNX produced by mjlab (walking or tracking).

The .pt file must contain `actor_state_dict` with `mlp.{0,2,4,...}.{weight,bias}`.
The reference ONNX initializers must match either `mlp.*` (walking) or
`policy.mlp.*` (tracking) and have identical shapes to the .pt MLP.

Usage (single):
  uv run python convert_pt_to_onnx.py \
    --pt-file /path/to/sample.pt \
    --reference-onnx /path/to/reference.onnx \
    --output /path/to/out.onnx

Usage (batch):
  uv run python convert_pt_to_onnx.py \
    --pt-dir /path/to/samples_dir \
    --pattern "sample_*.pt" \
    --reference-onnx /path/to/reference.onnx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

POLICY_KEY_RE = re.compile(r"^(?:policy\.)?(mlp\.\d+\.(?:weight|bias))$")


def find_policy_initializers(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
  """Return {pt_key: initializer} for every MLP policy tensor in the ONNX graph.

  Strips the optional `policy.` prefix so keys match the .pt actor_state_dict.
  """
  found: dict[str, onnx.TensorProto] = {}
  for init in model.graph.initializer:
    m = POLICY_KEY_RE.match(init.name)
    if m:
      pt_key = m.group(1)
      if pt_key in found:
        raise ValueError(
          f"Duplicate policy key in reference ONNX: {init.name} vs already-found "
          f"{pt_key}. Cannot disambiguate."
        )
      found[pt_key] = init
  if not found:
    raise ValueError(
      "No `mlp.*.weight/bias` (or `policy.mlp.*`) initializers found in reference."
    )
  return found


def replace_initializers(
  model: onnx.ModelProto,
  policy_inits: dict[str, onnx.TensorProto],
  actor_sd: dict,
) -> None:
  """Overwrite each policy initializer's data with the .pt actor weights.

  Validates shapes and dtypes; raises if any policy initializer has no
  corresponding .pt key, or vice versa.
  """
  pt_keys = {k for k in actor_sd if POLICY_KEY_RE.match(k)}
  onnx_keys = set(policy_inits.keys())
  missing = onnx_keys - pt_keys
  extra = pt_keys - onnx_keys
  if missing:
    raise ValueError(
      f".pt is missing keys present in reference ONNX: {sorted(missing)}"
    )
  if extra:
    raise ValueError(
      f".pt has extra keys not present in reference ONNX: {sorted(extra)}. "
      "Architectures differ — cannot do a 1:1 swap."
    )

  for pt_key, init in policy_inits.items():
    tensor = actor_sd[pt_key]
    if not isinstance(tensor, torch.Tensor):
      raise TypeError(f"actor_state_dict['{pt_key}'] is not a torch.Tensor")
    arr = tensor.detach().cpu().numpy()
    if tuple(arr.shape) != tuple(init.dims):
      raise ValueError(
        f"Shape mismatch for {pt_key}: .pt={arr.shape} vs ref={tuple(init.dims)}"
      )
    new_init = numpy_helper.from_array(np.ascontiguousarray(arr), name=init.name)
    init.CopyFrom(new_init)


def convert_one(
  pt_path: Path,
  output_path: Path,
  reference_path: Path,
) -> None:
  """Clone the reference ONNX, swap in .pt's policy weights, save to output."""
  ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
  if not isinstance(ckpt, dict) or "actor_state_dict" not in ckpt:
    raise ValueError(f"{pt_path} has no 'actor_state_dict'.")
  actor_sd = ckpt["actor_state_dict"]

  model = onnx.load(str(reference_path))
  policy_inits = find_policy_initializers(model)
  replace_initializers(model, policy_inits, actor_sd)
  onnx.checker.check_model(model)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  onnx.save(model, str(output_path))
  print(
    f"  wrote {output_path}  (swapped {len(policy_inits)} policy tensors, "
    f"motion data + metadata preserved)"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  src = parser.add_mutually_exclusive_group(required=True)
  src.add_argument("--pt-file", type=str, help="Single .pt to convert")
  src.add_argument("--pt-dir", type=str, help="Directory of .pt files for batch")
  parser.add_argument(
    "--pattern", type=str, default="*.pt", help="Glob inside --pt-dir (default: *.pt)"
  )
  parser.add_argument(
    "--reference-onnx",
    type=str,
    required=True,
    help="Reference ONNX file. Its policy MLP weights get replaced; everything "
    "else (motion data, metadata, time_step input, etc.) is preserved.",
  )
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output .onnx path (only with --pt-file). Default: alongside the .pt.",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory (only with --pt-dir). Default: <pt-dir>/onnx",
  )
  args = parser.parse_args()

  ref_path = Path(args.reference_onnx)
  if not ref_path.is_file():
    sys.exit(f"Reference ONNX not found: {ref_path}")

  ref = onnx.load(str(ref_path))
  policy_inits = find_policy_initializers(ref)
  motion_inits = [i for i in ref.graph.initializer if not POLICY_KEY_RE.match(i.name)]
  has_motion = any(
    i.name.startswith(("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"))
    for i in motion_inits
  )
  print(
    f"[INFO] Reference: {ref_path}\n"
    f"  policy tensors:  {len(policy_inits)} (will be replaced)\n"
    f"  motion tensors:  {'yes (preserved)' if has_motion else 'none'}\n"
    f"  metadata fields: {len(ref.metadata_props)} (preserved)"
  )

  if args.pt_file:
    pt_path = Path(args.pt_file)
    if not pt_path.is_file():
      sys.exit(f".pt not found: {pt_path}")
    out_path = Path(args.output) if args.output else pt_path.with_suffix(".onnx")
    convert_one(pt_path, out_path, ref_path)
    return

  pt_dir = Path(args.pt_dir)
  if not pt_dir.is_dir():
    sys.exit(f"--pt-dir not found: {pt_dir}")
  out_dir = Path(args.output_dir) if args.output_dir else pt_dir / "onnx"
  pt_files = sorted(pt_dir.glob(args.pattern))
  if not pt_files:
    sys.exit(f"No files matching '{args.pattern}' under {pt_dir}")
  print(f"[INFO] Found {len(pt_files)} .pt files. Output dir: {out_dir}")

  failed: list[tuple[Path, str]] = []
  for pt_path in pt_files:
    out_path = out_dir / (pt_path.stem + ".onnx")
    if out_path.exists():
      print(f"  skip {out_path.name} (already exists)")
      continue
    try:
      convert_one(pt_path, out_path, ref_path)
    except Exception as e:
      print(f"  FAILED {pt_path.name}: {e}")
      failed.append((pt_path, str(e)))

  print(
    f"[INFO] Done. ok={len(pt_files) - len(failed)} failed={len(failed)} "
    f"out_dir={out_dir}"
  )
  if failed:
    sys.exit(1)


if __name__ == "__main__":
  main()
