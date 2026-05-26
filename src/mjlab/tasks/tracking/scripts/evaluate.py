"""Evaluate a trained tracking policy and compute metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
import wandb

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for policy evaluation."""

  wandb_run_path: str | None = None
  """W&B run path 'entity/project/run_id'. Mutually exclusive with checkpoint_file."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  """Local checkpoint .pt path. Requires --motion-file. Mutually exclusive with W&B."""
  motion_file: str | None = None
  """Local motion .npz path. Required when --checkpoint-file is used."""
  num_envs: int = 1024
  """Number of parallel environments (= number of episodes to evaluate)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_file: str | None = None
  """Optional path to save metrics as JSON."""


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  """Run policy evaluation and compute metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load configs.
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")

  local_ckpt: Path | None = None
  if cfg.checkpoint_file is not None:
    if cfg.wandb_run_path is not None:
      raise ValueError("Pass either --wandb-run-path or --checkpoint-file, not both.")
    if cfg.motion_file is None:
      raise ValueError("--checkpoint-file requires --motion-file.")
    local_ckpt = Path(cfg.checkpoint_file)
    motion_path = Path(cfg.motion_file)
    if not local_ckpt.is_file():
      raise FileNotFoundError(f"Checkpoint not found: {local_ckpt}")
    if not motion_path.is_file():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd.motion_file = str(motion_path)
  else:
    if cfg.wandb_run_path is None:
      raise ValueError("Pass either --wandb-run-path or --checkpoint-file.")
    api = wandb.Api()
    run = api.run(cfg.wandb_run_path)
    art = next((a for a in run.used_artifacts() if a.type == "motions"), None)
    if art is None:
      raise RuntimeError("No motion artifact found in the run.")
    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  # Evaluation config.
  motion_cmd.sampling_mode = "start"
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if local_ckpt is not None:
    resume_path = local_ckpt.resolve()
  else:
    assert cfg.wandb_run_path is not None
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    resume_path, _ = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  ee_body_names = env_cfg.terminations["ee_body_pos"].params["body_names"]
  motion_total = int(command.motion.time_step_total)
  # Envs can survive at most min(motion_total, episode_time_out_steps). Normalize
  # progress by THAT ceiling so success (env survived its time-out) maps to
  # progress=1.0, regardless of whether the motion is longer than the episode.
  max_env_steps = int(env.unwrapped.max_episode_length)
  progress_ceiling = min(motion_total, max_env_steps)
  print(f"[INFO] End effector bodies: {ee_body_names}")
  print(
    f"[INFO] Motion length: {motion_total} frames; "
    f"episode max steps: {max_env_steps}; "
    f"progress ceiling: {progress_ceiling}"
  )

  # Metric accumulators.
  all_mpkpe: list[torch.Tensor] = []
  all_r_mpkpe: list[torch.Tensor] = []
  all_joint_vel_error: list[torch.Tensor] = []
  all_ee_pos_error: list[torch.Tensor] = []
  all_ee_ori_error: list[torch.Tensor] = []

  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  # Policy steps survived per env before reset. NOT command.time_steps —
  # that's the (looped) motion phase, which wraps when motion < episode and
  # gives bogus partial progress on otherwise-successful envs.
  progress_frames = torch.zeros(cfg.num_envs, device=device)

  obs = env.get_observations()
  env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    ep_len_pre = env.unwrapped.episode_length_buf.clone()
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    # Compute metrics for active envs.
    active = ~done_envs
    if active.any():
      all_mpkpe.append(torch.where(active, compute_mpkpe(command), 0.0))
      all_r_mpkpe.append(torch.where(active, compute_root_relative_mpkpe(command), 0.0))
      all_joint_vel_error.append(
        torch.where(active, compute_joint_velocity_error(command), 0.0)
      )
      all_ee_pos_error.append(
        torch.where(active, compute_ee_position_error(command, ee_body_names), 0.0)
      )
      all_ee_ori_error.append(
        torch.where(active, compute_ee_orientation_error(command, ee_body_names), 0.0)
      )

    # Track completions.
    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs

    if newly_done.any():
      completed = (ep_len_pre + 1).clamp(max=progress_ceiling).float()
      progress_frames = torch.where(newly_done, completed, progress_frames)
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{cfg.num_envs} episodes completed "
        f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
        f"terminated={(newly_done & terminated).sum().item()})"
      )
    step += 1

  # Compute mean metrics.
  stacks = [
    all_mpkpe,
    all_r_mpkpe,
    all_joint_vel_error,
    all_ee_pos_error,
    all_ee_ori_error,
  ]
  stacks = [torch.stack(s, dim=0) for s in stacks]
  active_steps = (stacks[0] != 0).sum(dim=0).float().clamp(min=1)
  means = [s.sum(dim=0) / active_steps for s in stacks]

  # Per-env fraction of episode-reachable motion completed before reset, in [0, 1].
  # Divisor = min(motion_total, max_env_steps): an env that survives its episode
  # without falling reaches progress = 1.0 even if the motion is longer than the
  # episode length.
  progress_frac = progress_frames / float(progress_ceiling)
  q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95], device=device)
  p_values = torch.quantile(progress_frac, q)

  metrics = {
    "success_rate": success.float().mean().item(),
    "mpkpe": means[0].mean().item(),
    "r_mpkpe": means[1].mean().item(),
    "joint_vel_error": means[2].mean().item(),
    "ee_pos_error": means[3].mean().item(),
    "ee_ori_error": means[4].mean().item(),
    "motion_total_frames": motion_total,
    "max_env_steps": max_env_steps,
    "progress_ceiling": progress_ceiling,
    "progress_mean": progress_frac.mean().item(),
    "progress_std": progress_frac.std().item(),
    "progress_min": progress_frac.min().item(),
    "progress_max": progress_frac.max().item(),
    "progress_p10": p_values[0].item(),
    "progress_p25": p_values[1].item(),
    "progress_p50": p_values[2].item(),
    "progress_p75": p_values[3].item(),
    "progress_p90": p_values[4].item(),
    "progress_p95": p_values[5].item(),
  }

  print("\n" + "=" * 50)
  print("Evaluation Results")
  print("=" * 50)
  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)

  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {output_path}")

  env.close()
  return metrics


def main():
  import mjlab.tasks  # noqa: F401

  tracking_tasks = [t for t in list_tasks() if "Tracking" in t]
  if not tracking_tasks:
    print("No tracking tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tracking_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
