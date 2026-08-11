from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def threat_caught(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Terminate if an active threat closes to within contact_threshold."""
  term = env.command_manager.get_term(command_name)
  return term.threat_active & (term.threat_range < term.cfg.contact_threshold)
