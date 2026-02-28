"""Record video from a trained checkpoint without launching a viewer."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401 — populate registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task_id", type=str)
  parser.add_argument("--checkpoint-file", type=str, required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--video-length", type=int, default=1000)
  parser.add_argument("--output-dir", type=str, default="videos")
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--actor-hidden-dims", type=int, nargs="+", default=None,
    help="Override actor hidden dims, e.g. --actor-hidden-dims 256 128",
  )
  args = parser.parse_args()

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task_id, play=True)
  agent_cfg = load_rl_cfg(args.task_id)
  env_cfg.scene.num_envs = args.num_envs
  if args.actor_hidden_dims is not None:
    agent_cfg.actor.hidden_dims = tuple(args.actor_hidden_dims)

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=args.device, render_mode="rgb_array"
  )

  output_dir = Path(args.output_dir)
  env = VideoRecorder(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=args.video_length,
    disable_logger=False,
  )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(
    env, asdict(agent_cfg), device=args.device
  )
  runner.load(
    args.checkpoint_file,
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  obs = env.get_observations()
  for _ in range(args.video_length):
    actions = policy(obs)
    obs, _, _, _ = env.step(actions)

  env.close()
  print(f"[Done] Video saved to {output_dir}/")


if __name__ == "__main__":
  main()
