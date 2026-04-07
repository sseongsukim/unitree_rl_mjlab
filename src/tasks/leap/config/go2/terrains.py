"""Custom terrain generators for Unitree Go2 leap tasks."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import FlatPatchSamplingCfg, SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border


@dataclass(kw_only=True)
class BoxGapTerrainCfg(SubTerrainCfg):
    """Flat terrain with a single pit-like gap spanning the travel direction."""

    gap_width_range: tuple[float, float] = (0.20, 0.45)
    floor_depth: float = 2.0
    approach_length: float = 5.0
    landing_length: float = 3.0
    border_width: float = 0.25
    spawn_offset_from_gap: float = 0.8

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        size_x, size_y = self.size
        inner_x = max(1e-6, size_x - 2.0 * self.border_width)
        inner_y = max(1e-6, size_y - 2.0 * self.border_width)

        gap_min, gap_max = self.gap_width_range
        gap_width = gap_min + difficulty * (gap_max - gap_min)

        total_required = self.approach_length + gap_width + self.landing_length
        if total_required > inner_x:
            scale = inner_x / total_required
            approach_length = self.approach_length * scale
            landing_length = self.landing_length * scale
            gap_width *= scale
        else:
            approach_length = self.approach_length
            landing_length = self.landing_length

        left_start_x = self.border_width
        left_end_x = left_start_x + approach_length
        right_start_x = left_end_x + gap_width
        right_end_x = right_start_x + landing_length
        center_y = 0.5 * size_y

        half_height = self.floor_depth / 2.0
        z_center = -half_height

        platform_rgba = (0.50, 0.70, 0.55, 1.0)
        border_rgba = (0.18, 0.24, 0.20, 1.0)
        floor_rgba = (0.10, 0.10, 0.10, 1.0)

        if self.border_width > 0.0:
            border_center = (0.5 * size_x, center_y, z_center)
            border_boxes = make_border(
                body,
                (size_x, size_y),
                (inner_x, inner_y),
                self.floor_depth,
                border_center,
            )
            for geom in border_boxes:
                geometries.append(TerrainGeometry(geom=geom, color=border_rgba))

        floor_h = 0.1
        floor_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(size_x / 2.0, size_y / 2.0, floor_h / 2.0),
            pos=(size_x / 2.0, center_y, -self.floor_depth - floor_h / 2.0),
        )
        geometries.append(TerrainGeometry(geom=floor_geom, color=floor_rgba))

        left_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(approach_length / 2.0, inner_y / 2.0, half_height),
            pos=(left_start_x + approach_length / 2.0, center_y, z_center),
        )
        geometries.append(TerrainGeometry(geom=left_geom, color=platform_rgba))

        right_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(landing_length / 2.0, inner_y / 2.0, half_height),
            pos=(right_start_x + landing_length / 2.0, center_y, z_center),
        )
        geometries.append(TerrainGeometry(geom=right_geom, color=platform_rgba))

        if right_end_x < (size_x - self.border_width):
            tail_length = size_x - self.border_width - right_end_x
            tail_geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(tail_length / 2.0, inner_y / 2.0, half_height),
                pos=(right_end_x + tail_length / 2.0, center_y, z_center),
            )
            geometries.append(TerrainGeometry(geom=tail_geom, color=platform_rgba))

        landing_margin = min(0.55, max(0.20, 0.20 * landing_length))

        spawn_x0 = max(left_start_x, left_end_x - self.spawn_offset_from_gap)
        spawn_x1 = spawn_x0
        landing_x0 = right_start_x + landing_margin
        landing_x1 = max(landing_x0, right_end_x - landing_margin)

        def _line_patches(x0: float, x1: float, n: int) -> np.ndarray:
            xs = np.linspace(x0, x1, n, dtype=float)
            ys = np.full_like(xs, center_y)
            zs = np.zeros_like(xs)
            return np.stack([xs, ys, zs], axis=-1)

        num_spawn_patches = 8
        num_landing_patches = 8
        flat_patches = {
            "spawn": _line_patches(spawn_x0, spawn_x1, num_spawn_patches),
            "landing": _line_patches(landing_x0, landing_x1, num_landing_patches),
        }

        spawn_origin = np.array(
            [0.5 * (spawn_x0 + spawn_x1), center_y, 0.0],
            dtype=float,
        )
        return TerrainOutput(
            origin=spawn_origin,
            geometries=geometries,
            flat_patches=flat_patches,
        )


def make_leap_gap_terrain_cfg(
    *,
    size: tuple[float, float] = (12.0, 3.2),
    num_rows: int = 8,
    num_cols: int = 8,
    border_width: float = 6.0,
    gap_width_range: tuple[float, float] = (0.20, 0.40),
    approach_length: float = 5.0,
    landing_length: float = 3.0,
    spawn_offset_from_gap: float = 0.8,
    floor_depth: float = 2.0,
) -> TerrainGeneratorCfg:
    """Build a terrain generator for leap tasks with a single ground gap."""

    return TerrainGeneratorCfg(
        curriculum=True,
        size=size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        add_lights=True,
        sub_terrains={
            "gap": BoxGapTerrainCfg(
                proportion=1.0,
                gap_width_range=gap_width_range,
                floor_depth=floor_depth,
                approach_length=approach_length,
                landing_length=landing_length,
                border_width=0.25,
                spawn_offset_from_gap=spawn_offset_from_gap,
                flat_patch_sampling={
                    "spawn": FlatPatchSamplingCfg(num_patches=8),
                    "landing": FlatPatchSamplingCfg(num_patches=8),
                },
            )
        },
    )
