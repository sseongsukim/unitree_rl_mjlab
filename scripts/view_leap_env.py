"""Preview registered environments in a viewer.

Examples:
  python scripts/view_env.py Unitree-Go2-Flat
  python scripts/view_env.py Unitree-Go2-Jump --num-envs 4 --agent random
  python scripts/view_env.py Unitree-Go2-Leap --viewer native --no-terminations
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import src.tasks.velocity.config.go2  # noqa: F401
import src.tasks.leap.config.go2  # noqa: F401


@dataclass(frozen=True)
class ViewEnvConfig:
    agent: Literal["zero", "random"] = "zero"
    viewer: Literal["auto", "native", "viser"] = "auto"
    env_mode: Literal["auto", "play", "train"] = "auto"
    device: str | None = None
    num_envs: int = 4
    no_terminations: bool = False


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


def run_view(task_id: str, cfg: ViewEnvConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.env_mode == "train":
        use_play_env = False
    elif cfg.env_mode == "play":
        use_play_env = True
    else:
        use_play_env = task_id != "Unitree-Go2-Leap"

    env_cfg = load_env_cfg(task_id, play=use_play_env)
    env_cfg.scene.num_envs = cfg.num_envs

    if cfg.no_terminations:
        env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    policy = _make_policy(cfg.agent, env)

    if cfg.viewer == "auto":
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


def main() -> None:
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
        default="Unitree-Go2-Leap",
    )

    args = tyro.cli(
        ViewEnvConfig,
        args=remaining_args,
        default=ViewEnvConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_view(chosen_task, args)


if __name__ == "__main__":
    main()
