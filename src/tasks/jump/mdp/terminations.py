from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
