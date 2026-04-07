from dataclasses import asdict, dataclass, field

import random
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg

from src.rl_core.flash_rl.agents import create_agent
from src.rl_core.flash_rl.envs.mjlab import FlashVecEnvWrapper
from src.rl_core.flash_rl.common.config import FlashRlCfg


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: FlashRlCfg

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = FlashRlCfg()
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def launch_off_policy(task_id: str, cfg: TrainConfig):

    print("debug")
