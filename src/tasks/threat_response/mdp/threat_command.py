from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ThreatAwareVelocityCommand(UniformVelocityCommand):
  """Extends the twist velocity command with a tracked virtual threat.

  The threat is not a MuJoCo body -- it is a per-env kinematic point (world-frame
  xy position) that walks toward the robot's real simulated position. When it
  gets within ``safety_dist`` the twist command is overridden with a retreat
  velocity (backward) plus a heading target pointed at the threat, so the
  existing ``track_linear_velocity`` / ``track_angular_velocity`` rewards teach
  evasion for free.
  """

  cfg: ThreatAwareVelocityCommandCfg

  def __init__(self, cfg: ThreatAwareVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.threat_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.threat_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.threat_approach_speed = torch.zeros(self.num_envs, device=self.device)
    self.threat_bearing = torch.zeros(self.num_envs, device=self.device)
    self.threat_range = torch.full(
      (self.num_envs,), cfg.max_range, device=self.device
    )
    self.threat_level = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)

    roll = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
    start_mask = roll < self.cfg.threat_prob
    start_ids = env_ids[start_mask]
    stop_ids = env_ids[~start_mask]

    self.threat_active[stop_ids] = False
    self.threat_range[stop_ids] = self.cfg.max_range
    self.threat_bearing[stop_ids] = 0.0
    self.threat_level[stop_ids] = 0.0

    if len(start_ids) == 0:
      return

    n = len(start_ids)
    init_range = torch.empty(n, device=self.device).uniform_(*self.cfg.initial_range_range)
    bearing0 = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
    speed = torch.empty(n, device=self.device).uniform_(*self.cfg.approach_speed_range)

    robot_pos_w = self.robot.data.root_link_pos_w[start_ids, :2]
    robot_heading = self.robot.data.heading_w[start_ids]
    world_bearing = robot_heading + bearing0
    offset = torch.stack([torch.cos(world_bearing), torch.sin(world_bearing)], dim=-1)
    offset = offset * init_range.unsqueeze(-1)

    self.threat_pos_w[start_ids] = robot_pos_w + offset
    self.threat_approach_speed[start_ids] = speed
    self.threat_active[start_ids] = True

  def _update_command(self) -> None:
    super()._update_command()

    active_ids = self.threat_active.nonzero(as_tuple=False).flatten()
    if len(active_ids) == 0:
      return

    dt = self._env.step_dt
    robot_pos_w = self.robot.data.root_link_pos_w[active_ids, :2]

    to_robot = robot_pos_w - self.threat_pos_w[active_ids]
    dist = torch.norm(to_robot, dim=-1).clamp_min(1e-6)
    direction = to_robot / dist.unsqueeze(-1)
    step = torch.minimum(self.threat_approach_speed[active_ids] * dt, dist)
    self.threat_pos_w[active_ids] = self.threat_pos_w[active_ids] + direction * step.unsqueeze(-1)

    rel_w = self.threat_pos_w[active_ids] - robot_pos_w
    rel_w3 = torch.cat([rel_w, torch.zeros_like(rel_w[:, :1])], dim=-1)
    robot_quat = self.robot.data.root_link_quat_w[active_ids]
    rel_b = quat_apply_inverse(robot_quat, rel_w3)

    bearing = torch.atan2(rel_b[:, 1], rel_b[:, 0])
    rng = torch.norm(rel_b[:, :2], dim=-1).clamp_max(self.cfg.max_range)

    self.threat_bearing[active_ids] = bearing
    self.threat_range[active_ids] = rng
    self.threat_level[active_ids] = torch.clamp(
      1.0 - rng / self.cfg.safety_dist, min=0.0, max=1.0
    )

    engage_mask = rng <= self.cfg.safety_dist
    engage_ids = active_ids[engage_mask]
    if len(engage_ids) == 0:
      return

    engage_bearing = self.threat_bearing[engage_ids]
    engage_range = self.threat_range[engage_ids]
    retreat_speed = torch.clamp(
      self.cfg.k_retreat_gain * (self.cfg.safety_dist - engage_range),
      min=0.0,
      max=self.cfg.v_retreat_max,
    )
    self.vel_command_b[engage_ids, 0] = -retreat_speed
    self.vel_command_b[engage_ids, 1] = 0.0
    # engage_bearing is already the heading error (target = current heading +
    # bearing), so it can be fed straight into the same control law the parent
    # class uses for heading tracking.
    self.heading_target[engage_ids] = self.robot.data.heading_w[engage_ids] + engage_bearing
    self.vel_command_b[engage_ids, 2] = torch.clip(
      self.cfg.heading_control_stiffness * engage_bearing,
      min=self.cfg.ranges.ang_vel_z[0],
      max=self.cfg.ranges.ang_vel_z[1],
    )


@dataclass(kw_only=True)
class ThreatAwareVelocityCommandCfg(UniformVelocityCommandCfg):
  threat_prob: float = 0.0
  """Probability that a resample cycle starts a new threat approach. Set by the
  ``threat_exposure`` curriculum term."""

  safety_dist: float = 3.0
  """Range (m) below which the twist command is overridden with a retreat."""

  contact_threshold: float = 0.6
  """Range (m) below which the threat is considered to have caught the robot."""

  v_retreat_max: float = 1.0
  """Maximum backward retreat speed (m/s). Should stay within the trained
  ``ranges.lin_vel_x`` negative bound."""

  k_retreat_gain: float = 0.5
  """Gain mapping (safety_dist - range) to retreat speed."""

  approach_speed_range: tuple[float, float] = (0.3, 1.5)
  """Random approach speed (m/s) sampled for each new threat."""

  initial_range_range: tuple[float, float] = (2.5, 4.5)
  """Random spawn distance (m) sampled for each new threat."""

  max_range: float = 5.0
  """Clamp for the range observation (also the reset value when inactive)."""

  def build(self, env: ManagerBasedRlEnv) -> ThreatAwareVelocityCommand:
    return ThreatAwareVelocityCommand(self, env)


def make_threat_aware_command_cfg(
  base: UniformVelocityCommandCfg, **threat_kwargs
) -> ThreatAwareVelocityCommandCfg:
  """Copy an existing ``UniformVelocityCommandCfg`` (e.g. the task's "twist"
  command) into a ``ThreatAwareVelocityCommandCfg`` with the same base fields."""
  base_fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
  base_fields.update(threat_kwargs)
  return ThreatAwareVelocityCommandCfg(**base_fields)
