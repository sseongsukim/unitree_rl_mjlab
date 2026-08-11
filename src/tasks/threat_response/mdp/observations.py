from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def threat_field(env: ManagerBasedRlEnv, command_name: str, field: str) -> torch.Tensor:
  """Read one scalar field (bearing/range/level/active) off the threat command
  term. Split into separate observation terms (rather than one packed vector)
  because each field has a different physical scale and needs its own noise."""
  term = env.command_manager.get_term(command_name)
  return getattr(term, field).float().unsqueeze(-1)
