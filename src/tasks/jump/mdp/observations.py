from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def obstacle_height_map(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    grid_size: tuple[float, float] | None = None,
    resolution: float | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    clamp_min: float = 0.0,
    clamp_max: float | None = None,
    miss_value: float = 0.0,
) -> torch.Tensor:
    """Return a local height map with the lowest visible surface as zero.

    On the flat jump task this turns the ground into zeros and the cube top into
    positive heights, without exposing an explicit cube distance signal.
    """
    sensor: RayCastSensor = env.scene[sensor_name]
    hit_pos_w = sensor.data.hit_pos_w
    distances = sensor.data.distances

    hit_heights = hit_pos_w[..., 2]
    valid_hits = distances >= 0.0

    inf = torch.full_like(hit_heights, torch.inf)
    min_visible_height = torch.min(
        torch.where(valid_hits, hit_heights, inf),
        dim=1,
    ).values
    min_visible_height = torch.where(
        torch.isfinite(min_visible_height),
        min_visible_height,
        torch.zeros_like(min_visible_height),
    )

    height_map = torch.where(
        valid_hits,
        hit_heights - min_visible_height.unsqueeze(1),
        torch.full_like(hit_heights, miss_value),
    )

    if (
        grid_size is not None
        and resolution is not None
        and (x_range is not None or y_range is not None)
    ):
        size_x, size_y = grid_size
        grid_x = torch.arange(
            -size_x / 2,
            size_x / 2 + resolution * 0.5,
            resolution,
            device=height_map.device,
            dtype=height_map.dtype,
        )
        grid_y = torch.arange(
            -size_y / 2,
            size_y / 2 + resolution * 0.5,
            resolution,
            device=height_map.device,
            dtype=height_map.dtype,
        )
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
        keep_mask = torch.ones_like(grid_x, dtype=torch.bool)
        if x_range is not None:
            keep_mask &= (grid_x >= x_range[0]) & (grid_x <= x_range[1])
        if y_range is not None:
            keep_mask &= (grid_y >= y_range[0]) & (grid_y <= y_range[1])
        height_map = height_map[:, keep_mask.reshape(-1)]

    height_map = torch.clamp_min(height_map, clamp_min)
    if clamp_max is not None:
        height_map = torch.clamp_max(height_map, clamp_max)
    return height_map


def privileged_obstacle_state(
    env: ManagerBasedRlEnv,
    front_x: float,
    center_x: float,
    goal_x: float,
    target_y: float,
    clamp_distance: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return critic-only obstacle progress cues.

    This gives the value function explicit access to obstacle-relative progress
    without exposing cube distance to the actor policy.
    """
    asset: Entity = env.scene[asset_cfg.name]
    root_pos = asset.data.root_link_pos_w

    distances = torch.stack(
        (
            front_x - root_pos[:, 0],
            center_x - root_pos[:, 0],
            goal_x - root_pos[:, 0],
            root_pos[:, 1] - target_y,
        ),
        dim=-1,
    )
    return torch.clamp(distances, min=-clamp_distance, max=clamp_distance)
