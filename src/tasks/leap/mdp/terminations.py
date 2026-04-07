from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def backward_x_velocity(
    env: ManagerBasedRlEnv,
    threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot moves backward too quickly in world x."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w[:, 0] < -threshold


class insufficient_x_progress:
    """Terminate when recent forward progress stays too small for too long."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.anchor_x = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
        self.anchor_step = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        min_progress: float,
        grace_period_s: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Check progress against a recent x anchor instead of the spawn point."""
        asset: Entity = env.scene[asset_cfg.name]
        current_x = asset.data.root_link_pos_w[:, 0]

        is_reset = env.episode_length_buf == 0
        self.anchor_x = torch.where(is_reset, current_x, self.anchor_x)
        self.anchor_step = torch.where(
            is_reset, env.episode_length_buf, self.anchor_step
        )

        min_steps = max(1, int(grace_period_s / env.step_dt))
        elapsed_steps = env.episode_length_buf - self.anchor_step
        recent_progress = current_x - self.anchor_x
        timed_out = elapsed_steps >= min_steps
        should_terminate = timed_out & (recent_progress < min_progress)

        made_progress = recent_progress >= min_progress
        self.anchor_x = torch.where(made_progress, current_x, self.anchor_x)
        self.anchor_step = torch.where(
            made_progress, env.episode_length_buf, self.anchor_step
        )

        return should_terminate


class excessive_y_drift:
    """Terminate when the robot moves too far laterally from its reset lane."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.anchor_y = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_abs_y: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        current_y = asset.data.root_link_pos_w[:, 1]

        is_reset = env.episode_length_buf == 0
        self.anchor_y = torch.where(is_reset, current_y, self.anchor_y)
        return torch.abs(current_y - self.anchor_y) > max_abs_y


# LEAP


def _terrain_patch_bounds(
    env: ManagerBasedRlEnv,
    patch_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain = env.scene.terrain
    assert terrain is not None, "Terrain is required for terrain-patch termination."
    assert (
        patch_name in terrain.flat_patches
    ), f"Terrain flat patch '{patch_name}' not found."
    patches = terrain.flat_patches[patch_name]
    levels = terrain.terrain_levels
    types = terrain.terrain_types
    env_patches = patches[levels, types]
    x = env_patches[..., 0]
    y = env_patches[..., 1]
    return x.min(dim=1).values, x.max(dim=1).values, y.mean(dim=1)


def landing_progress(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    patch_name: str = "landing",
    forward_distance: float = 0.45,
    y_margin: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully after the robot lands and walks into the landing patch."""

    sensor: ContactSensor = env.scene[sensor_name]
    asset: Entity = env.scene[asset_cfg.name]
    assert sensor.data.found is not None

    patch_x_min, _, patch_center_y = _terrain_patch_bounds(env, patch_name)
    root_pos = asset.data.root_link_pos_w
    in_landing_lane = torch.abs(root_pos[:, 1] - patch_center_y) <= y_margin
    reached_progress = root_pos[:, 0] >= (patch_x_min + forward_distance)
    in_contact = (sensor.data.found > 0).any(dim=1)
    return in_landing_lane & reached_progress & in_contact
