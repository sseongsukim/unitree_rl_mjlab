from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


class threat_distance_progress:
  """Backstop reward for actually gaining distance from an active threat.

  The command override in ``ThreatAwareVelocityCommand`` already makes the
  existing track_linear_velocity/track_angular_velocity rewards teach evasion
  (tracking the retreat command well = retreating well). This term only
  catches the residual case where the robot tracks the command perfectly but
  something (terrain, a shove) still keeps it from actually escaping.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev_range = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    range_now = term.threat_range
    active = term.threat_active.float()

    just_reset = env.episode_length_buf <= 1
    self.prev_range = torch.where(just_reset, range_now, self.prev_range)

    delta = torch.clamp(range_now - self.prev_range, min=0.0)
    reward = delta * active

    self.prev_range = range_now
    return reward
