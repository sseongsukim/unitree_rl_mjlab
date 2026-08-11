from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .threat_command import ThreatAwareVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ThreatStage(TypedDict):
  step: int
  threat_prob: float


def threat_exposure(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[ThreatStage],
) -> dict[str, torch.Tensor]:
  """Ramp threat_prob up over training, mirroring the ``commands_vel`` curriculum
  in the base velocity task -- mutates the command term's cfg in place, which
  ``ThreatAwareVelocityCommand._resample_command`` reads every resample."""
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  cfg = cast(ThreatAwareVelocityCommandCfg, command_term.cfg)
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      cfg.threat_prob = stage["threat_prob"]
  return {"threat_prob": torch.tensor([cfg.threat_prob])}
