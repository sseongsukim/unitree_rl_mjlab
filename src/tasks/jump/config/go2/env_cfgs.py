"""Unitree Go2 jump environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_go2_robot_cfg
from src.tasks.jump.jump_env_cfg import (
  CubeObstacleOffsetCfg,
  CubeHeightResetCfg,
  DEFAULT_CUBE_OFFSET,
  make_jump_env_cfg,
)


def unitree_go2_jump_env_cfg(
  play: bool = False,
  robot_spawn_xy: tuple[float, float] = (0.0, 0.0),
  cube_offset: CubeObstacleOffsetCfg = DEFAULT_CUBE_OFFSET,
  cube_height_reset: CubeHeightResetCfg | None = CubeHeightResetCfg(),
  line_spacing: float = 1.5,
  shared_layout: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 jump configuration with one cube per environment."""
  return make_jump_env_cfg(
    robot_cfg=get_go2_robot_cfg(),
    foot_names=("FR", "FL", "RR", "RL"),
    site_names=("FR", "FL", "RR", "RL"),
    play=play,
    robot_spawn_xy=robot_spawn_xy,
    cube_offset=cube_offset,
    cube_height_reset=cube_height_reset,
    line_spacing=line_spacing,
    shared_layout=shared_layout,
    base_body_name="base_link",
  )
