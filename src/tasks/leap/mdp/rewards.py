from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def world_x_velocity_reward(
    env: ManagerBasedRlEnv,
    clamp_min: float = 0.0,
    clamp_max: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward velocity along the world x-axis."""
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.clamp(asset.data.root_link_lin_vel_w[:, 0], min=clamp_min)
    if clamp_max is not None:
        reward = torch.clamp(reward, max=clamp_max)
    return reward


def com_yaw_reward(
    env: ManagerBasedRlEnv,
    desired_yaw: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return squared yaw error as a cost term."""
    asset: Entity = env.scene[asset_cfg.name]
    yaw_diff = asset.data.heading_w - desired_yaw
    yaw_diff = torch.atan2(
        torch.sin(yaw_diff),
        torch.cos(yaw_diff),
    )
    return torch.square(yaw_diff)


def cube_approach_velocity_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    clamp_min: float = 0.0,
    clamp_max: float | None = None,
    gate_x_min_b: float = 0.0,
    gate_x_max_b: float = 2.0,
    gate_y_abs_b: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward velocity that closes distance to the object while it is in front."""
    asset: Entity = env.scene[asset_cfg.name]
    cube: Entity = env.scene[object_name]

    rel_pos_w = cube.data.root_link_pos_w - asset.data.root_link_pos_w
    rel_xy_w = rel_pos_w[:, :2]
    dist_xy = torch.norm(rel_xy_w, dim=1, keepdim=True)
    dir_to_cube_xy = rel_xy_w / torch.clamp(dist_xy, min=1e-6)

    lin_vel_w_xy = asset.data.root_link_lin_vel_w[:, :2]
    closing_speed = torch.sum(lin_vel_w_xy * dir_to_cube_xy, dim=1)

    cube_pos_b = quat_apply_inverse(asset.data.root_link_quat_w, rel_pos_w)
    in_window = (
        (cube_pos_b[:, 0] >= gate_x_min_b)
        & (cube_pos_b[:, 0] <= gate_x_max_b)
        & (torch.abs(cube_pos_b[:, 1]) <= gate_y_abs_b)
    )

    reward = torch.where(
        in_window,
        torch.clamp(closing_speed, min=clamp_min),
        torch.zeros_like(closing_speed),
    )
    if clamp_max is not None:
        reward = torch.clamp(reward, max=clamp_max)
    return reward


class cube_clear_bonus:
    """Give a one-time bonus when the robot base clears the cube front face."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.rewarded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        object_name: str,
        margin_x: float = 0.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        cube: Entity = env.scene[object_name]

        is_reset = env.episode_length_buf == 0
        self.rewarded[is_reset] = False

        cube_front_face_x = cube.data.root_link_pos_w[:, 0]
        crossed = asset.data.root_link_pos_w[:, 0] > (cube_front_face_x + margin_x)
        bonus = crossed & (~self.rewarded)
        self.rewarded |= crossed
        return bonus.float()


def foot_lift_near_obstacle(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_height: float,
    cube_half_length: float,
    cube_half_width: float,
    pre_margin_x: float = 0.18,
    post_margin_x: float = 0.08,
    y_margin: float = 0.08,
    height_tolerance: float = 0.05,
    require_front_feet_active: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feet for lifting near an obstacle crossing window.

    Each foot becomes active when it approaches the obstacle's x-span and overlaps
    laterally with the obstacle. This naturally makes front feet activate earlier
    than rear feet, and rear feet activate as the robot passes over the obstacle.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cube: Entity = env.scene[object_name]

    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    cube_center = cube.data.root_link_pos_w

    cube_near_x = cube_center[:, 0].unsqueeze(1) - cube_half_length
    cube_far_x = cube_center[:, 0].unsqueeze(1) + cube_half_length
    cube_center_y = cube_center[:, 1].unsqueeze(1)

    foot_x = foot_pos_w[:, :, 0]
    foot_y = foot_pos_w[:, :, 1]
    foot_z = foot_pos_w[:, :, 2]

    in_x_window = (foot_x >= (cube_near_x - pre_margin_x)) & (
        foot_x <= (cube_far_x + post_margin_x)
    )
    in_y_window = torch.abs(foot_y - cube_center_y) <= (cube_half_width + y_margin)
    active_feet = in_x_window & in_y_window
    if require_front_feet_active and active_feet.shape[1] >= 2:
        front_feet_active = active_feet[:, 0] & active_feet[:, 1]
        active_feet = active_feet & front_feet_active.unsqueeze(1)

    height_error = (foot_z - target_height) / max(height_tolerance, 1e-6)
    foot_reward = torch.exp(-torch.square(height_error)) * active_feet.float()
    num_active = active_feet.float().sum(dim=1)
    return torch.sum(foot_reward, dim=1) / torch.clamp(num_active, min=1.0)


def total_reward_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the per-step total reward rate before dt scaling."""
    return env.reward_manager._step_reward.sum(dim=1)


###
### New rewards for leap env
###


def _terrain_patch_bounds(
    env: ManagerBasedRlEnv,
    patch_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain = env.scene.terrain
    assert terrain is not None, "Terrain is required for terrain-patch rewards."
    assert (
        patch_name in terrain.flat_patches
    ), f"Terrain flat patch '{patch_name}' not found."
    patches = terrain.flat_patches[patch_name]  # [rows, cols, num_patches, 3]
    levels = terrain.terrain_levels
    types = terrain.terrain_types
    env_patches = patches[levels, types]
    x = env_patches[..., 0]
    y = env_patches[..., 1]
    return x.min(dim=1).values, x.max(dim=1).values, y.mean(dim=1)


def terrain_gap_foot_lift(
    env: ManagerBasedRlEnv,
    target_height: float,
    spawn_patch_name: str = "spawn",
    landing_patch_name: str = "landing",
    pre_margin_x: float = 0.15,
    post_margin_x: float = 0.12,
    y_margin: float = 0.20,
    height_tolerance: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward lifting feet while they pass through the terrain gap window."""

    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    foot_x = foot_pos_w[:, :, 0]
    foot_y = foot_pos_w[:, :, 1]
    foot_z = foot_pos_w[:, :, 2]

    spawn_x_min, spawn_x_max, spawn_center_y = _terrain_patch_bounds(
        env, spawn_patch_name
    )
    landing_x_min, _, landing_center_y = _terrain_patch_bounds(env, landing_patch_name)

    gap_start_x = spawn_x_max.unsqueeze(1)
    gap_end_x = landing_x_min.unsqueeze(1)
    center_y = ((spawn_center_y + landing_center_y) * 0.5).unsqueeze(1)

    in_x_window = (foot_x >= (gap_start_x - pre_margin_x)) & (
        foot_x <= (gap_end_x + post_margin_x)
    )
    in_y_window = torch.abs(foot_y - center_y) <= y_margin
    active_feet = in_x_window & in_y_window

    height_error = (foot_z - target_height) / max(height_tolerance, 1e-6)
    foot_reward = torch.exp(-torch.square(height_error)) * active_feet.float()
    num_active = active_feet.float().sum(dim=1)
    return torch.sum(foot_reward, dim=1) / torch.clamp(num_active, min=1.0)


def landing_patch_approach_velocity(
    env: ManagerBasedRlEnv,
    patch_name: str = "landing",
    clamp_min: float = 0.0,
    clamp_max: float | None = None,
    gate_x_min_w: float | None = None,
    gate_x_max_w: float | None = None,
    y_margin: float = 0.45,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward velocity that closes distance toward the landing patch."""

    asset: Entity = env.scene[asset_cfg.name]
    patch_x_min, patch_x_max, patch_center_y = _terrain_patch_bounds(env, patch_name)

    root_pos = asset.data.root_link_pos_w
    root_vel = asset.data.root_link_lin_vel_w[:, :2]

    target_x = 0.5 * (patch_x_min + patch_x_max)
    target_y = patch_center_y
    target_xy = torch.stack([target_x, target_y], dim=1)
    rel_xy = target_xy - root_pos[:, :2]
    dist_xy = torch.norm(rel_xy, dim=1, keepdim=True)
    dir_xy = rel_xy / torch.clamp(dist_xy, min=1e-6)
    closing_speed = torch.sum(root_vel * dir_xy, dim=1)

    in_lane = torch.abs(root_pos[:, 1] - patch_center_y) <= y_margin
    in_front_gate = torch.ones_like(in_lane, dtype=torch.bool)
    if gate_x_min_w is not None:
        in_front_gate &= root_pos[:, 0] >= gate_x_min_w
    if gate_x_max_w is not None:
        in_front_gate &= root_pos[:, 0] <= gate_x_max_w

    reward = torch.where(
        in_lane & in_front_gate,
        torch.clamp(closing_speed, min=clamp_min),
        torch.zeros_like(closing_speed),
    )
    if clamp_max is not None:
        reward = torch.clamp(reward, max=clamp_max)
    return reward


class terrain_landing_bonus:
    """Give a one-time bonus when the robot reaches the landing patch and lands."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.rewarded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        patch_name: str = "landing",
        y_margin: float = 0.25,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        asset: Entity = env.scene[asset_cfg.name]
        assert sensor.data.found is not None

        is_reset = env.episode_length_buf == 0
        self.rewarded[is_reset] = False

        patch_x_min, _, patch_center_y = _terrain_patch_bounds(env, patch_name)
        root_pos = asset.data.root_link_pos_w
        in_landing_zone = (root_pos[:, 0] >= patch_x_min) & (
            torch.abs(root_pos[:, 1] - patch_center_y) <= y_margin
        )
        in_contact = (sensor.data.found > 0).any(dim=1)
        bonus = in_landing_zone & in_contact & (~self.rewarded)
        self.rewarded |= bonus
        return bonus.float()
