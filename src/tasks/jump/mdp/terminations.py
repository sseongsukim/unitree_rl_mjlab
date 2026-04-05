from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_ALL_FEET_CFG = SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))
_DEFAULT_OBSTACLE_CFG = SceneEntityCfg("cube")


def _cube_top_height(
    env: ManagerBasedRlEnv,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    terrain_height: float = 0.0,
) -> torch.Tensor:
    obstacle: Entity = env.scene[obstacle_cfg.name]
    obstacle_geom_ids = obstacle.indexing.geom_ids
    obstacle_half_height = env.sim.model.geom_size[:, obstacle_geom_ids[0], 2]
    return terrain_height + 2.0 * obstacle_half_height


def off_track(
    env: ManagerBasedRlEnv,
    target_y: float,
    max_offset: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the robot drifts too far sideways from the straight path."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_link_pos_w[:, 1] - target_y) > max_offset


def moved_backward(
    env: ManagerBasedRlEnv,
    min_x: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the robot retreats behind the spawn line."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 0] < min_x


def reached_goal_x(
    env: ManagerBasedRlEnv,
    goal_x: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate once the robot reaches the target x-position on top of the cube."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 0] >= goal_x


class low_body_on_cube:
    """Terminate when the robot slumps too low while standing on the cube top."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.low_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        min_base_height_offset: float,
        top_height: float | None,
        height_tolerance: float,
        min_top_contacts: int = 2,
        hold_steps: int = 4,
        terrain_height: float = 0.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
        obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
        top_height_offset: float = 0.0,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        sensor = env.scene[sensor_name]
        assert sensor.data.found is not None

        just_reset = env.episode_length_buf <= 1
        self.low_steps[just_reset] = 0

        root_x = asset.data.root_link_pos_w[:, 0]
        foot_contacts = (sensor.data.found > 0).float()
        foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
        if top_height is None:
            top_height_tensor = (
                _cube_top_height(env, obstacle_cfg, terrain_height) + top_height_offset
            )
        else:
            top_height_tensor = torch.full_like(root_x, top_height + top_height_offset)
        on_top_surface = (
            torch.abs(foot_heights - top_height_tensor.unsqueeze(1)) <= height_tolerance
        )
        top_contacts = foot_contacts * on_top_surface.float()
        stable_on_top = top_contacts.sum(dim=1) >= float(min_top_contacts)

        min_height = top_height_tensor + min_base_height_offset
        too_low = stable_on_top & (asset.data.root_link_pos_w[:, 2] < min_height)
        self.low_steps = torch.where(
            too_low,
            self.low_steps + 1,
            torch.zeros_like(self.low_steps),
        )
        return self.low_steps >= hold_steps


class reached_goal_on_cube:
    """Terminate after stably reaching the target x-position on the cube top."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.goal_hold_steps = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        goal_x: float,
        sensor_name: str,
        top_height: float | None,
        height_tolerance: float,
        min_top_contacts: int = 2,
        hold_steps: int = 3,
        min_forward_alignment: float = 0.8,
        terrain_height: float = 0.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
        obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
        top_height_offset: float = 0.0,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        sensor = env.scene[sensor_name]
        assert sensor.data.found is not None

        just_reset = env.episode_length_buf <= 1
        self.goal_hold_steps[just_reset] = 0

        root_x = asset.data.root_link_pos_w[:, 0]
        foot_contacts = (sensor.data.found > 0).float()
        foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
        if top_height is None:
            top_height_tensor = (
                _cube_top_height(env, obstacle_cfg, terrain_height) + top_height_offset
            )
        else:
            top_height_tensor = torch.full_like(root_x, top_height + top_height_offset)
        on_top_surface = (
            torch.abs(foot_heights - top_height_tensor.unsqueeze(1)) <= height_tolerance
        )
        top_contacts = foot_contacts * on_top_surface.float()
        stable_on_top = top_contacts.sum(dim=1) >= float(min_top_contacts)
        heading_ok = torch.cos(asset.data.heading_w) >= min_forward_alignment
        reached = (root_x >= goal_x) & stable_on_top & heading_ok

        self.goal_hold_steps = torch.where(
            reached,
            self.goal_hold_steps + 1,
            torch.zeros_like(self.goal_hold_steps),
        )
        return self.goal_hold_steps >= hold_steps


class stuck_in_x:
    """Terminate if the robot makes too little forward progress for too long."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.best_x = torch.zeros(env.num_envs, device=env.device)
        self.stuck_steps = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        min_progress: float,
        grace_steps: int,
        max_stuck_steps: int,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        root_x = asset.data.root_link_pos_w[:, 0]

        just_reset = env.episode_length_buf <= 1
        self.best_x[just_reset] = root_x[just_reset]
        self.stuck_steps[just_reset] = 0

        improved = root_x > (self.best_x + min_progress)
        self.best_x = torch.maximum(self.best_x, root_x)
        self.stuck_steps = torch.where(
            improved,
            torch.zeros_like(self.stuck_steps),
            self.stuck_steps + 1,
        )

        can_check = env.episode_length_buf >= grace_steps
        return can_check & (self.stuck_steps >= max_stuck_steps)
