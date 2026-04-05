"""Preview the Go2 jump environment in a viewer.

Example:
  python scripts/view_jump_env.py --viewer native --agent zero
  python scripts/view_jump_env.py --num-envs 4 --cube0-dx 0.8 --cube0-sx 0.15
"""

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.tasks.jump.config.go2.env_cfgs import (
    CubeObstacleOffsetCfg,
    unitree_go2_jump_env_cfg,
)


@dataclass(frozen=True)
class ViewJumpConfig:
    agent: Literal["zero", "random"] = "zero"
    viewer: Literal["auto", "native", "viser"] = "auto"
    device: str | None = None
    num_envs: int = 10
    shared_layout: bool = True
    line_spacing: float = 1.5
    robot_x: float = 0.0
    robot_y: float = 0.0
    cube0_dx: float = 1.5
    cube0_dy: float = 0.0
    cube0_z: float = 0.06
    cube0_sx: float = 0.8
    cube0_sy: float = 0.30
    cube0_sz: float = 0.06


def _make_policy(agent: Literal["zero", "random"], env: RslRlVecEnvWrapper):
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

    if agent == "zero":

        class PolicyZero:
            def __call__(self, obs) -> torch.Tensor:
                del obs
                return torch.zeros(action_shape, device=env.unwrapped.device)

        return PolicyZero()

    class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
            del obs
            return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

    return PolicyRandom()


def main(cfg: ViewJumpConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cube_offset = CubeObstacleOffsetCfg(
        offset=(cfg.cube0_dx, cfg.cube0_dy, cfg.cube0_z),
        size=(cfg.cube0_sx, cfg.cube0_sy, cfg.cube0_sz),
        rgba=(0.75, 0.45, 0.20, 1.0),
    )

    env_cfg = unitree_go2_jump_env_cfg(
        play=True,
        robot_spawn_xy=(cfg.robot_x, cfg.robot_y),
        cube_offset=cube_offset,
        line_spacing=cfg.line_spacing,
        shared_layout=cfg.shared_layout,
    )
    env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    policy = _make_policy(cfg.agent, env)

    if cfg.viewer == "auto":
        import os

        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        viewer = "native" if has_display else "viser"
    else:
        viewer = cfg.viewer

    if viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
