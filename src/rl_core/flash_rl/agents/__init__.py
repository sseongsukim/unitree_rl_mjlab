from typing import Any

import gymnasium as gym

from src.rl_core.flash_rl.agents.base_agent import BaseAgent
from src.rl_core.flash_rl.agents.flashSAC.agent import FlashSACAgent
from src.rl_core.flash_rl.types import NDArray


def create_agent(
    observation_space: gym.spaces.Space[NDArray],
    action_space: gym.spaces.Space[NDArray],
    env_info: dict[str, Any],
    cfg: Any,
) -> BaseAgent[Any]:

    if cfg.agent_type == "flashSAC":
        agent = FlashSACAgent(
            observation_space=observation_space,
            action_space=action_space,
            env_info=env_info,
            cfg=cfg,
        )

    return agent
