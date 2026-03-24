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
