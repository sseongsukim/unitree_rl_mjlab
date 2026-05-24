from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_HEIGHT_REWARD_PARAMS: dict[str, dict[str, float]] = {
    "pre_obstacle_front_foot_lift": {"target_height": 0.04},
    "front_swing_clearance": {"target_height": 0.03},
    "front_contact_before_jump": {"desired_contact_height": 0.01},
    "rear_stance_push": {"front_target_height": 0.03},
    "rear_air_before_takeoff": {"front_target_height": 0.03},
    "feet_on_cube_top": {"top_height": 0.0},
    "rear_feet_on_cube_top": {"top_height": 0.0},
    "cube_top_step": {"top_height": 0.0},
    "rear_swing_clearance": {"target_height": 0.02},
}


def _set_height_map_clamp(env: ManagerBasedRlEnv, height: float) -> None:
    obs_manager = env.observation_manager
    for group_name in ("actor", "critic"):
        term_names = obs_manager.active_terms.get(group_name, [])
        if "height_map" not in term_names:
            continue
        term_index = term_names.index("height_map")
        term_cfg = obs_manager._group_obs_term_cfgs[group_name][term_index]
        term_cfg.params["clamp_max"] = height


def _set_height_reward_params(env: ManagerBasedRlEnv, height: float) -> None:
    for reward_name, param_offsets in _HEIGHT_REWARD_PARAMS.items():
        if reward_name not in env.reward_manager.active_terms:
            continue
        term_cfg = env.reward_manager.get_term_cfg(reward_name)
        for param_name, offset in param_offsets.items():
            term_cfg.params[param_name] = height + offset


def obstacle_height_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    min_height: float,
    max_height: float,
    goal_x: float,
    success_threshold: float = 0.70,
    failure_threshold: float = 0.20,
    height_step: float = 0.02,
    decrease_step: float = 0.01,
    update_interval_episodes: int = 512,
) -> dict[str, torch.Tensor]:
    """Adapt obstacle height from recent crossing success rate."""
    if not hasattr(env, "_jump_obstacle_height"):
        env._jump_obstacle_height = min_height
        env._jump_curriculum_success_count = 0
        env._jump_curriculum_episode_count = 0
        env._jump_curriculum_success_rate = 0.0

    height = float(env._jump_obstacle_height)

    if env.common_step_counter > 0:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        robot = env.scene["robot"]
        root_x = robot.data.root_link_pos_w[env_ids, 0]
        successes = int(torch.count_nonzero(root_x >= goal_x).item())
        episodes = int(root_x.numel())

        env._jump_curriculum_success_count += successes
        env._jump_curriculum_episode_count += episodes

        if env._jump_curriculum_episode_count >= update_interval_episodes:
            success_rate = (
                env._jump_curriculum_success_count
                / max(env._jump_curriculum_episode_count, 1)
            )
            env._jump_curriculum_success_rate = success_rate
            if success_rate >= success_threshold:
                height = min(max_height, height + height_step)
            elif success_rate <= failure_threshold:
                height = max(min_height, height - decrease_step)
            env._jump_curriculum_success_count = 0
            env._jump_curriculum_episode_count = 0

    height = min(max(height, min_height), max_height)

    env._jump_obstacle_height = height
    _set_height_map_clamp(env, height)
    _set_height_reward_params(env, height)

    return {
        "height": torch.tensor(height, device=env.device),
        "success_rate": torch.tensor(
            float(getattr(env, "_jump_curriculum_success_rate", 0.0)),
            device=env.device,
        ),
    }
