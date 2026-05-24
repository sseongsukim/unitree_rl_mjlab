"""Custom terrain generators for Unitree Go2 jump tasks."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import FlatPatchSamplingCfg, SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


@dataclass(kw_only=True)
class BoxForwardStairsTerrainCfg(SubTerrainCfg):
    """Terrain-generator stair obstacle built from primitive terrain boxes.

    This follows mjlab's velocity rough-terrain style: the obstacle is part of
    the generated terrain body, not a separate scene cube entity.
    """

    obstacle_start_x: float = 0.55
    obstacle_center_y: float = 0.0
    obstacle_width_y: float = 1.5
    total_depth: float = 3.2
    landing_length: float = 1.2
    step_height: float = 0.25
    num_steps: int = 4
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    plane_thickness: float = 1.0

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")

        size_x, size_y = self.size
        origin_x = 0.5 * size_x
        center_y = 0.5 * size_y + self.spawn_y
        start_x = origin_x + self.obstacle_start_x
        width_center_y = center_y + self.obstacle_center_y

        num_steps = max(1, int(self.num_steps))
        step_depth = self.total_depth / num_steps
        final_height = self.step_height * num_steps

        geometries: list[TerrainGeometry] = []
        ground_geom = make_plane(
            body,
            self.size,
            0.0,
            center_zero=False,
            plane_thickness=self.plane_thickness,
        )[0]
        geometries.append(
            TerrainGeometry(geom=ground_geom, color=(0.46, 0.52, 0.46, 1.0))
        )

        for step_idx in range(num_steps):
            top_height = self.step_height * (step_idx + 1)
            x_center = start_x + step_depth * (step_idx + 0.5)
            z_center = 0.5 * top_height
            t = step_idx / max(num_steps - 1, 1)
            color = (
                0.40 + 0.18 * t,
                0.47 + 0.14 * t,
                0.42 + 0.10 * t,
                1.0,
            )
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.5 * step_depth, 0.5 * self.obstacle_width_y, z_center),
                pos=(x_center, width_center_y, z_center),
            )
            geometries.append(TerrainGeometry(geom=geom, color=color))

        landing_center_x = start_x + self.total_depth + 0.5 * self.landing_length
        landing_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                0.5 * self.landing_length,
                0.5 * self.obstacle_width_y,
                0.5 * final_height,
            ),
            pos=(landing_center_x, width_center_y, 0.5 * final_height),
        )
        geometries.append(
            TerrainGeometry(geom=landing_geom, color=(0.58, 0.61, 0.52, 1.0))
        )

        spawn_origin = np.array(
            [origin_x + self.spawn_x, center_y, 0.0],
            dtype=float,
        )
        flat_patches = {
            "spawn": np.array([spawn_origin], dtype=float),
            "landing": np.array(
                [
                    [
                        start_x + self.total_depth + 0.5 * self.landing_length,
                        center_y,
                        final_height,
                    ]
                ],
                dtype=float,
            ),
        }
        return TerrainOutput(
            origin=spawn_origin,
            geometries=geometries,
            flat_patches=flat_patches,
        )


def make_jump_stair_terrain_cfg(
    *,
    size: tuple[float, float] = (8.0, 3.0),
    num_rows: int = 1,
    num_cols: int = 1,
    border_width: float = 1.0,
    obstacle_center_x: float = 2.15,
    obstacle_width_y: float = 1.5,
    total_depth: float = 3.2,
    landing_length: float = 1.2,
    step_height: float = 0.25,
    num_steps: int = 4,
    resolution: float = 0.1,
) -> TerrainGeneratorCfg:
    """Build a terrain generator with one raised stair obstacle per tile."""

    del resolution
    obstacle_start_x = obstacle_center_x - 0.5 * total_depth
    return TerrainGeneratorCfg(
        curriculum=False,
        size=size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        add_lights=True,
        sub_terrains={
            "stairs": BoxForwardStairsTerrainCfg(
                proportion=1.0,
                obstacle_start_x=obstacle_start_x,
                obstacle_width_y=obstacle_width_y,
                total_depth=total_depth,
                landing_length=landing_length,
                step_height=step_height,
                num_steps=num_steps,
                flat_patch_sampling={
                    "spawn": FlatPatchSamplingCfg(num_patches=1),
                    "landing": FlatPatchSamplingCfg(num_patches=1),
                },
            )
        },
    )
