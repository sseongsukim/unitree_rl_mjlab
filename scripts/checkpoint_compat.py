"""Checkpoint compatibility helpers for local scripts."""

from __future__ import annotations

import torch


def _adapt_tensor_shape(
  key: str, loaded_value: torch.Tensor, target_value: torch.Tensor
):
  if loaded_value.shape == target_value.shape:
    return loaded_value
  if loaded_value.ndim != target_value.ndim:
    return loaded_value

  normalizer_key = any(
    token in key.lower()
    for token in ("normalizer", "normalization", "running", "variance", "var")
  )
  adapted = target_value.clone() if normalizer_key else torch.zeros_like(target_value)
  slices = tuple(
    slice(0, min(loaded_dim, target_dim))
    for loaded_dim, target_dim in zip(loaded_value.shape, target_value.shape)
  )
  adapted[slices] = loaded_value[slices].to(device=adapted.device, dtype=adapted.dtype)
  return adapted


def _adapt_state_dict(loaded: dict, target: dict) -> dict:
  state_dict = dict(loaded)

  # Bridge newer GaussianDistribution checkpoints to older scalar-std models.
  if "distribution.std_param" in state_dict and "std" in target:
    state_dict["std"] = state_dict.pop("distribution.std_param")
  elif "std" in state_dict and "distribution.std_param" in target:
    state_dict["distribution.std_param"] = state_dict.pop("std")

  allowed_keys = set(target.keys())
  adapted = {}
  for key, value in state_dict.items():
    if key not in allowed_keys:
      continue
    target_value = target[key]
    if isinstance(value, torch.Tensor) and isinstance(target_value, torch.Tensor):
      value = _adapt_tensor_shape(key, value, target_value)
    adapted[key] = value
  return adapted


def load_runner_checkpoint_compat(
  runner,
  path: str,
  load_cfg: dict | None = None,
  strict: bool = True,
  map_location: str | None = None,
  set_iteration: bool = True,
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
  if load_iteration and set_iteration:
    runner.current_learning_iteration = loaded_dict["iter"]

  infos = loaded_dict.get("infos")
  if infos and "env_state" in infos:
    runner.env.unwrapped.common_step_counter = infos["env_state"][
      "common_step_counter"
    ]
  return infos
