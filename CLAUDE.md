# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Development Workflow

**Always use `uv run`, not python**.

```sh
# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories.

## Style Guidelines

- Line length limit is 88 columns. This applies to code, comments, and
  docstrings.
- Indent width is 2 spaces (configured in ruff).
- Avoid local imports unless strictly necessary (e.g. circular imports).
- Tests: use functions and fixtures, not test classes. Favor targeted tests
  over exhaustive edge-case coverage. Prefer running individual tests.

## Architecture Overview

mjlab combines Isaac Lab's manager-based API with MuJoCo Warp for
GPU-accelerated robot learning. Simulation runs on GPU via `mujoco_warp`
(Warp arrays), with zero-copy conversion to PyTorch tensors.

### Key Modules (`src/mjlab/`)

- **`sim/`** — Physics simulation wrapper around `mujoco_warp`. `Simulation`
  manages `mjwarp.Model`/`mjwarp.Data` on GPU. `WarpBridge`/`TorchArray`
  provide zero-copy Warp-to-PyTorch conversion.
- **`envs/`** — `ManagerBasedRlEnv`: the core vectorized RL environment.
  Orchestrates simulation + all managers. Configured via
  `ManagerBasedRlEnvCfg` dataclass.
- **`managers/`** — Each manager handles one RL concern: `ActionManager`,
  `ObservationManager`, `RewardManager`, `TerminationManager`,
  `CommandManager`, `EventManager` (domain randomization/resets),
  `CurriculumManager`, `MetricsManager`. Managers are composed of
  pluggable **terms** (functions or classes) specified in config dicts.
- **`tasks/`** — Task definitions and registry. Each task registers via
  `register_mjlab_task()` in `tasks/registry.py` with an env config, RL
  config, and optional custom runner. Task-specific MDP logic (rewards,
  observations, terminations, commands) lives under `tasks/<task>/mdp/`.
- **`rl/`** — RSL-RL integration. `RslRlVecEnvWrapper` adapts
  `ManagerBasedRlEnv` for rsl_rl. `MjlabOnPolicyRunner` extends rsl_rl's
  runner with ONNX export and W&B integration. Config dataclasses:
  `RslRlOnPolicyRunnerCfg`, `RslRlModelCfg`, `RslRlPpoAlgorithmCfg`.
- **`scene/`** — Builds MuJoCo spec from `SceneCfg` (terrain + entities +
  sensors), then compiles to `MjModel`.
- **`entity/`** — Robot/object definitions via `EntityCfg`.
- **`asset_zoo/`** — Bundled robots: `unitree_g1` (humanoid),
  `unitree_go1` (quadruped), `i2rt_yam` (manipulator).
- **`sensor/`** — Contact, raycast, camera, and builtin (IMU-like) sensors.
- **`actuator/`** — PD, DC motor, delayed, and learned actuator models.
- **`terrains/`** — Procedural terrain generation with curriculum support.
- **`viewer/`** — Native MuJoCo viewer, Viser web viewer, offscreen
  renderer.

### CLI Entry Points (defined in `pyproject.toml [project.scripts]`)

```sh
uv run train <TASK_ID> [OPTIONS]     # Train RL agent (PPO via rsl_rl)
uv run play <TASK_ID> [OPTIONS]      # Visualize policy or dummy agent
uv run list_envs                     # List all registered task IDs
uv run demo                          # Interactive pretrained demo
```

### Training Data Flow

```
CLI (train.py) → registry.load_env_cfg/load_rl_cfg
  → ManagerBasedRlEnv(env_cfg)
    → Scene (terrain + entities + sensors → MjModel)
    → Simulation (mjwarp on GPU)
    → Managers (action, obs, reward, termination, command, event)
  → RslRlVecEnvWrapper
  → MjlabOnPolicyRunner (PPO)
  → Logs to logs/rsl_rl/{experiment_name}/{timestamp}/
```

### Configuration Patterns

- All configs are `@dataclass`-based, overridable via CLI (`tyro`).
  Example: `--agent.actor.hidden-dims "(256, 128)"`.
- Tasks use factory functions to build base configs, then customize
  per-robot (e.g., `make_velocity_env_cfg()` → add G1-specific rewards).
- Manager terms are specified as `dict[str, TermCfg]` mapping term name
  to a callable + params + manager-specific fields (e.g., `weight` for
  rewards).

### Headless Rendering

On servers without a display, set `MUJOCO_GL=egl` for offscreen rendering
(video recording, `play --video`). The train script sets this automatically.
