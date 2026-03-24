"""Checkpoint compatibility helpers for local scripts."""

from __future__ import annotations

import torch


def _adapt_state_dict(loaded: dict, target: dict) -> dict:
  state_dict = dict(loaded)

  # Bridge newer GaussianDistribution checkpoints to older scalar-std models.
  if "distribution.std_param" in state_dict and "std" in target:
    state_dict["std"] = state_dict.pop("distribution.std_param")
  elif "std" in state_dict and "distribution.std_param" in target:
    state_dict["distribution.std_param"] = state_dict.pop("std")

  allowed_keys = set(target.keys())
  return {key: value for key, value in state_dict.items() if key in allowed_keys}


def load_runner_checkpoint_compat(
  runner,
  path: str,
  load_cfg: dict | None = None,
  strict: bool = True,
  map_location: str | None = None,
):
  """Load a checkpoint while adapting small model format differences."""
  loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

  loaded_dict["actor_state_dict"] = _adapt_state_dict(
    loaded_dict["actor_state_dict"], runner.alg.actor.state_dict()
  )
  loaded_dict["critic_state_dict"] = _adapt_state_dict(
    loaded_dict["critic_state_dict"], runner.alg.critic.state_dict()
  )

  load_iteration = runner.alg.load(loaded_dict, load_cfg, strict)
  if load_iteration:
    runner.current_learning_iteration = loaded_dict["iter"]

  infos = loaded_dict.get("infos")
  if infos and "env_state" in infos:
    runner.env.unwrapped.common_step_counter = infos["env_state"][
      "common_step_counter"
    ]
  return infos
