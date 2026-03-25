from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FRONT_FEET_CFG = SceneEntityCfg("robot", site_names=("FR", "FL"))
_DEFAULT_ALL_FEET_CFG = SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))


def centerline_penalty(
    env: ManagerBasedRlEnv,
    target_y: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Keep the robot moving straight along the obstacle centerline."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_pos_w[:, 1] - target_y)


def feet_on_obstacle(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    approach_x: float,
    leave_x: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward placing feet on the cube while traversing the obstacle."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    in_window = (root_x >= approach_x) & (root_x <= leave_x)
    foot_contacts = (sensor.data.found > 0).float()
    return foot_contacts.mean(dim=1) * in_window.float()


def feet_on_top_surface(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    top_height: float,
    height_tolerance: float,
    approach_x: float,
    leave_x: float,
    min_forward_alignment: float = 0.5,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
) -> torch.Tensor:
    """Reward top-surface contacts that happen while facing forward.

    This helps distinguish "stepping onto the top while going straight" from
    sideways pole-vault-like contacts.
    """
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    in_window = (root_x >= approach_x) & (root_x <= leave_x)

    foot_contacts = (sensor.data.found > 0).float()
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    on_top_surface = torch.abs(foot_heights - top_height) <= height_tolerance
    top_contacts = foot_contacts * on_top_surface.float()
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return top_contacts.mean(dim=1) * in_window.float() * heading_gate * heading_scale


def obstacle_approach_speed(
    env: ManagerBasedRlEnv,
    goal_x: float,
    target_speed: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward moving forward while the obstacle has not been cleared yet."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    forward_speed = asset.data.root_link_lin_vel_w[:, 0]
    active = (root_x < goal_x).float()
    speed = torch.clamp(forward_speed, min=0.0, max=target_speed) / max(
        target_speed, 1.0e-6
    )
    return active * speed


def obstacle_stall_penalty(
    env: ManagerBasedRlEnv,
    goal_x: float,
    speed_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize getting stuck before the robot clears the obstacle."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    forward_speed = asset.data.root_link_lin_vel_w[:, 0]
    active = root_x < goal_x
    return active.float() * (forward_speed < speed_threshold).float()


def knee_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize knee or shank collisions.

    This is most useful with a dedicated contact sensor that only tracks thigh/calf
    contacts. If used with a broader non-foot contact sensor, it becomes a more
    general body-collision penalty.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return (sensor.data.found > 0).float().sum(dim=1)


def conditional_com_height_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height: float,
    std: float = 0.08,
    terrain_height: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward COM height only around the obstacle zone."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    root_z = asset.data.root_link_pos_w[:, 2]
    active = (root_x >= start_x) & (root_x <= end_x)
    height_error = root_z - terrain_height - target_height
    reward = torch.exp(-torch.square(height_error) / max(std**2, 1.0e-6))
    return reward * active.float()


def front_foot_clearance_over_obstacle(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height: float,
    std: float = 0.04,
    asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
) -> torch.Tensor:
    """Reward front-foot lift while crossing the obstacle window."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    height_error = foot_height - target_height
    reward = torch.exp(-torch.square(height_error) / max(std**2, 1.0e-6))
    return reward.mean(dim=1) * active.float()


def distance_to_clear_goal(
    env: ManagerBasedRlEnv,
    goal_x: float,
    clamp_distance: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Shape progress using the remaining forward distance to the clear-goal."""
    asset: Entity = env.scene[asset_cfg.name]
    distance = torch.clamp(
        goal_x - asset.data.root_link_pos_w[:, 0], min=0.0, max=clamp_distance
    )
    return 1.0 - (distance / max(clamp_distance, 1.0e-6))


def persistent_air_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    air_time_threshold: float = 0.35,
    max_excess_air_time: float = 0.75,
    require_single_swing: bool = True,
    goal_x: float | None = None,
    only_before_crossing: bool = False,
) -> torch.Tensor:
    """Penalize keeping one foot in the air for too long.

    This targets the common exploit where the policy learns to carry one leg
    lifted throughout most of the episode instead of using a proper four-leg
    gait. By default the penalty is strongest when exactly one foot is in swing,
    which avoids punishing brief multi-foot flight during the obstacle crossing.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    air_time = sensor_data.current_air_time
    contact_time = sensor_data.current_contact_time
    assert air_time is not None
    assert contact_time is not None

    in_air = contact_time <= 0.0
    excess_air = torch.clamp(
        air_time - air_time_threshold,
        min=0.0,
        max=max_excess_air_time,
    )
    penalty = excess_air.mean(dim=1)

    if require_single_swing:
        single_swing = (in_air.sum(dim=1) == 1).float()
        penalty = penalty * single_swing

    if only_before_crossing:
        assert goal_x is not None
        robot: Entity = env.scene["robot"]
        active = robot.data.root_link_pos_w[:, 0] < goal_x
        penalty = penalty * active.float()

    return penalty


def forward_alignment_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the robot for keeping its heading aligned with world +x."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.cos(asset.data.heading_w)


def post_crossing_heading_reward(
    env: ManagerBasedRlEnv,
    goal_x: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward-facing heading after the obstacle is cleared."""
    asset: Entity = env.scene[asset_cfg.name]
    crossed = asset.data.root_link_pos_w[:, 0] >= goal_x
    return crossed.float() * torch.cos(asset.data.heading_w)


def yaw_rate_penalty(
    env: ManagerBasedRlEnv,
    goal_x: float | None = None,
    only_after_crossing: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive yaw rotation, optionally only after obstacle crossing."""
    asset: Entity = env.scene[asset_cfg.name]
    penalty = torch.square(asset.data.root_link_ang_vel_b[:, 2])
    if only_after_crossing:
        assert goal_x is not None
        crossed = asset.data.root_link_pos_w[:, 0] >= goal_x
        penalty = penalty * crossed.float()
    return penalty


class obstacle_progress:
    """Reward forward progress toward clearing the cube exactly once."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.prev_progress = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        start_x: float,
        goal_x: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        just_reset = env.episode_length_buf <= 1
        self.prev_progress[just_reset] = 0.0

        root_x = asset.data.root_link_pos_w[:, 0]
        total_distance = max(goal_x - start_x, 1.0e-6)
        progress = torch.clamp((root_x - start_x) / total_distance, min=0.0, max=1.0)
        reward = torch.clamp(progress - self.prev_progress, min=0.0)
        self.prev_progress[:] = progress

        env.extras["log"]["Metrics/obstacle_progress_mean"] = progress.mean()
        return reward


class obstacle_crossing_bonus:
    """Give a one-time bonus when the robot fully clears the cube."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.already_crossed = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        goal_x: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        just_reset = env.episode_length_buf <= 1
        self.already_crossed[just_reset] = False

        root_x = asset.data.root_link_pos_w[:, 0]
        crossed = root_x >= goal_x
        reward = (~self.already_crossed & crossed).float()
        self.already_crossed |= crossed

        env.extras["log"]["Metrics/obstacle_crossed_ratio"] = crossed.float().mean()
        return reward
