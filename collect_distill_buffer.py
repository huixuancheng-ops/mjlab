"""Collect (actor_obs, teacher_action_mean) buffer for BC distillation.

Runs a trained mjlab actor in parallel envs with corruption OFF, records every
control-step (obs, action_mean) pair, and dumps the result as a torch .pt file
to be loaded by `diff_weight/plots/distill_il/distill_actor.py`.

Why this file lives in mjlab/: distillation downstream lives in diff_weight,
but mjlab is the only repo with the simulator + rsl-rl actor wrapper. Buffer
files are just torch tensors and load fine from either venv.

Run via mjlab's venv:
  /home/huixuan_cheng/mjlab/.venv/bin/python collect_distill_buffer.py \
    --task-id Mjlab-Velocity-Flat-Unitree-G1 \
    --checkpoint /home/huixuan_cheng/diff_weight/dataset/walking/checkpoint/seed_0_model_4499.pt \
    --out-path /home/huixuan_cheng/diff_weight/plots/distill_il/buffers/walking.pt \
    --num-envs 1024 --num-steps 500 --num-replays 1
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class Config:
  task_id: str
  """Mjlab task ID, e.g. Mjlab-Velocity-Flat-Unitree-G1 or
  Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation."""
  checkpoint: str
  """Local teacher .pt path (RSL-RL wrapped: actor_state_dict + critic + ...)."""
  out_path: str
  """Where to write the buffer .pt file (absolute path)."""
  num_envs: int = 1024
  """Number of parallel envs."""
  num_steps: int = 500
  """Steps per replay. For tracking, set to motion frame count."""
  num_replays: int = 1
  """Total replays; env.reset() called between replays for fresh DR draws."""
  motion_file: str | None = None
  """Path to motion .npz. Required for tracking tasks."""
  device: str | None = None
  """Defaults to cuda:0 if available."""
  seed: int = 0
  """Torch seed for action sampling/DR. Doesn't fully determine env state."""


def _check_path(p: str | Path, kind: str) -> Path:
  path = Path(p)
  if not path.is_file():
    raise FileNotFoundError(f"{kind} not found: {path}")
  return path


def main(cfg: Config) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)

  ckpt_path = _check_path(cfg.checkpoint, "checkpoint")
  out_path = Path(cfg.out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  # ── Build env cfg ──────────────────────────────────────────────
  env_cfg = load_env_cfg(cfg.task_id, play=False)
  agent_cfg = load_rl_cfg(cfg.task_id)

  # Roll out exactly like deploy: clean obs, no pushes, deterministic.
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  motion_cmd = env_cfg.commands.get("motion")
  if isinstance(motion_cmd, MotionCommandCfg):
    if cfg.motion_file is None:
      raise ValueError(f"task {cfg.task_id} is a tracking task; pass --motion-file")
    motion_path = _check_path(cfg.motion_file, "motion file")
    motion_cmd.motion_file = str(motion_path)
    motion_cmd.sampling_mode = "start"  # always start from frame 0
    # Disable reference-state-init randomization for repeatable replays.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
  elif cfg.motion_file is not None:
    print(f"[warn] --motion-file ignored for non-tracking task {cfg.task_id}")

  # ── Build env and load teacher ────────────────────────────────
  print(f"[buffer] task: {cfg.task_id}")
  print(f"[buffer] teacher: {ckpt_path}")
  print(
    f"[buffer] num_envs={cfg.num_envs} steps={cfg.num_steps} "
    f"replays={cfg.num_replays} -> total {cfg.num_envs * cfg.num_steps * cfg.num_replays:,} samples"
  )

  # Detect whether the teacher checkpoint was trained with an obs_normalizer
  # and patch agent_cfg.actor.obs_normalization accordingly. The task's
  # default rl_cfg may not match the actual training-time setting.
  probe = torch.load(str(ckpt_path.resolve()), map_location="cpu", weights_only=True)
  asd = probe.get("actor_state_dict", {}) if isinstance(probe, dict) else {}
  teacher_has_norm = any(k.startswith("obs_normalizer.") for k in asd)
  cfg_has_norm = getattr(agent_cfg.actor, "obs_normalization", False)
  if teacher_has_norm != cfg_has_norm:
    print(
      f"[buffer] patching agent_cfg.actor.obs_normalization "
      f"{cfg_has_norm} -> {teacher_has_norm} to match teacher ckpt"
    )
    agent_cfg.actor.obs_normalization = teacher_has_norm

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(ckpt_path.resolve()), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  # ── Probe shapes ──────────────────────────────────────────────
  obs = env.get_observations()
  actor_obs0 = obs["actor"]
  with torch.no_grad():
    act0 = policy(obs)
  in_dim = int(actor_obs0.shape[-1])
  out_dim = int(act0.shape[-1])
  print(f"[buffer] in_dim={in_dim}  out_dim={out_dim}")

  total_steps = cfg.num_steps * cfg.num_replays
  total_samples = total_steps * cfg.num_envs
  est_mb = total_samples * (in_dim + out_dim) * 4 / (1024**2)
  print(f"[buffer] CPU buffer estimate: {est_mb:.1f} MB")

  # Preallocate on CPU pinned memory for fast device->host transfer.
  obs_buf = torch.empty(total_samples, in_dim, dtype=torch.float32, pin_memory=True)
  act_buf = torch.empty(total_samples, out_dim, dtype=torch.float32, pin_memory=True)

  # ── Rollout ──────────────────────────────────────────────────
  t_start = time.time()
  write_idx = 0
  for replay in range(cfg.num_replays):
    if replay > 0:
      env.reset()
      obs = env.get_observations()
    print(f"[buffer] replay {replay + 1}/{cfg.num_replays}")
    for step in range(cfg.num_steps):
      with torch.no_grad():
        action = policy(obs)  # deterministic mean (stochastic_output=False)
      # Stash (obs[t], action[t]) — the pair the deployed teacher commits to.
      n = cfg.num_envs
      obs_buf[write_idx : write_idx + n].copy_(obs["actor"], non_blocking=True)
      act_buf[write_idx : write_idx + n].copy_(action, non_blocking=True)
      write_idx += n
      obs, _, _, _ = env.step(action)
      if (step + 1) % 100 == 0 or step == cfg.num_steps - 1:
        elapsed = time.time() - t_start
        print(
          f"[buffer]   step {step + 1:>5d}/{cfg.num_steps}  "
          f"({write_idx:,}/{total_samples:,} samples, {elapsed:.1f}s)"
        )
  assert write_idx == total_samples, (write_idx, total_samples)

  # ── Save ─────────────────────────────────────────────────────
  meta = {
    "task_id": cfg.task_id,
    "teacher_path": str(ckpt_path.resolve()),
    "motion_file": str(_check_path(cfg.motion_file, "motion").resolve())
    if cfg.motion_file
    else None,
    "num_envs": cfg.num_envs,
    "num_steps": cfg.num_steps,
    "num_replays": cfg.num_replays,
    "total_samples": int(write_idx),
    "in_dim": in_dim,
    "out_dim": out_dim,
    "corruption_enabled": False,
    "seed": cfg.seed,
  }
  torch.save({"obs": obs_buf, "action": act_buf, "meta": meta}, out_path)
  print(f"[buffer] saved -> {out_path}")
  print(f"[buffer] meta: {json.dumps(meta, indent=2)}")
  print(f"[buffer] total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
  main(tyro.cli(Config))
