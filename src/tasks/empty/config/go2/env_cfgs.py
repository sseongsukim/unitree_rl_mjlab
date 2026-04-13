"""Unitree Go2 empty environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.empty.empty_env_cfg import make_empty_env_cfg


def unitree_go2_empty_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 flat-terrain empty configuration."""
    return make_empty_env_cfg(play=play)
