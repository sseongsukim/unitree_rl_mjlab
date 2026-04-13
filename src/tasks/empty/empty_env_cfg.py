"""Empty task configuration for unsupervised skill discovery.

This task reuses the Unitree Go2 velocity-flat observation and action spaces,
while removing locomotion rewards and all early termination logic. Episodes end
only because of the configured time limit.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg


def make_empty_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create a minimal skill-discovery environment."""
    cfg = unitree_go2_flat_env_cfg(play=play)

    # Keep the velocity-flat observations and actions exactly as-is, but remove
    # task shaping so the downstream skill-discovery method can define its own
    # objective.
    cfg.rewards = {}
    cfg.curriculum = {}

    # Episodes should terminate only because of the fixed horizon.
    cfg.terminations = {
        "time_out": cfg.terminations["time_out"],
    }

    return cfg
