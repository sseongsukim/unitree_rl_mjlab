"""Minimal smoke test for the local empty environment."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from local_tasks import register_local_tasks

register_local_tasks()


@dataclass(frozen=True)
class EmptyEnvTestConfig:
    env: ManagerBasedRlEnvCfg
    video: bool = False
    device: str | None = None

    @staticmethod
    def from_task(task_id: str) -> "EmptyEnvTestConfig":
        env_cfg = load_env_cfg(task_id)
        return EmptyEnvTestConfig(env=env_cfg)


def run_empty_env_test(task_id: str, cfg: EmptyEnvTestConfig) -> None:
    configure_torch_backends()
    if cfg.device is not None:
        device = cfg.device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Creating env: task={task_id}, device={device}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )
    print(f"[INFO] Env created: num_envs={env.num_envs}, step_dt={env.step_dt}")
    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
        default="Unitree-Go2-Empty",
    )

    args = tyro.cli(
        EmptyEnvTestConfig,
        args=remaining_args,
        default=EmptyEnvTestConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_empty_env_test(chosen_task, args)


if __name__ == "__main__":
    main()
