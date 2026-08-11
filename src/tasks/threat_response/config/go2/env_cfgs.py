"""Unitree Go2 threat-response environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.threat_response.threat_env_cfg import make_threat_response_env_cfg


def unitree_go2_flat_threat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 flat terrain threat-response configuration."""
    return make_threat_response_env_cfg(play=play)
