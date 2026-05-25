"""Evaluate a trained velocity-walking policy on built-in tracking metrics.

Reports the same `error_vel_xy` / `error_vel_yaw` quantities accumulated by
`UniformVelocityCommand._update_metrics` (linear and yaw command-tracking error
in body frame), plus survival rate and mean episode length.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for velocity policy evaluation."""

  checkpoint_file: str
  """Local checkpoint .pt path."""
  num_envs: int = 1024
  """Number of parallel environments (= number of episodes to evaluate)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_file: str | None = None
  """Optional path to save metrics as JSON."""
  command_name: str = "twist"
  """Name of the UniformVelocityCommand term in the env config."""


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  """Run policy evaluation and compute built-in velocity-tracking metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  cmd_cfg = env_cfg.commands.get(cfg.command_name)
  if not isinstance(cmd_cfg, UniformVelocityCommandCfg):
    raise ValueError(
      f"Task {task_id} has no UniformVelocityCommand named "
      f"'{cfg.command_name}'. Available: {list(env_cfg.commands)}"
    )

  ckpt_path = Path(cfg.checkpoint_file).resolve()
  if not ckpt_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  # Match tracking evaluate.py: keep actor noise, drop random push.
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  print(f"[INFO] Loading checkpoint: {ckpt_path}")
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  cmd_term = cast(
    UniformVelocityCommand,
    env.unwrapped.command_manager.get_term(cfg.command_name),
  )
  robot = cmd_term.robot
  step_dt = env.unwrapped.step_dt

  # Per-env accumulators. Same error formula as UniformVelocityCommand._update_metrics.
  sum_err_xy = torch.zeros(cfg.num_envs, device=device)
  sum_err_yaw = torch.zeros(cfg.num_envs, device=device)
  active_steps = torch.zeros(cfg.num_envs, device=device)
  episode_lengths = torch.zeros(cfg.num_envs, device=device)
  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

  obs = env.get_observations()
  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    # Snapshot the command BEFORE stepping — this is what the policy is
    # actually trying to track on this step. env.step internally calls
    # command_manager.compute which may resample the command in place.
    cmd_for_step = cmd_term.command.clone()

    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    active = ~done_envs
    if active.any():
      lin_vel_b = robot.data.root_link_lin_vel_b
      ang_vel_b = robot.data.root_link_ang_vel_b
      err_xy = torch.norm(cmd_for_step[:, :2] - lin_vel_b[:, :2], dim=-1)
      err_yaw = torch.abs(cmd_for_step[:, 2] - ang_vel_b[:, 2])
      sum_err_xy = sum_err_xy + torch.where(active, err_xy, 0.0)
      sum_err_yaw = sum_err_yaw + torch.where(active, err_yaw, 0.0)
      active_steps = active_steps + active.float()

    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs
    if newly_done.any():
      episode_lengths = torch.where(newly_done, active_steps, episode_lengths)
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{cfg.num_envs} episodes completed "
        f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
        f"terminated={(newly_done & terminated).sum().item()})"
      )
    step += 1

  per_env_steps = active_steps.clamp(min=1)
  metrics = {
    "success_rate": success.float().mean().item(),
    "error_vel_xy": (sum_err_xy / per_env_steps).mean().item(),
    "error_vel_yaw": (sum_err_yaw / per_env_steps).mean().item(),
    "mean_episode_length_s": (episode_lengths.mean() * step_dt).item(),
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

  velocity_tasks = [t for t in list_tasks() if "Velocity" in t]
  if not velocity_tasks:
    print("No velocity tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(velocity_tasks),
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
