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
_DEFAULT_OBSTACLE_CFG = SceneEntityCfg("cube")


def _cube_top_height(
    env: ManagerBasedRlEnv,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    terrain_height: float = 0.0,
) -> torch.Tensor:
    obstacle: Entity = env.scene[obstacle_cfg.name]
    obstacle_geom_ids = obstacle.indexing.geom_ids
    obstacle_half_height = env.sim.model.geom_size[:, obstacle_geom_ids[0], 2]
    return terrain_height + 2.0 * obstacle_half_height


def _stable_top_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    top_height: float | None,
    height_tolerance: float,
    min_top_contacts: int,
    asset_cfg: SceneEntityCfg,
    obstacle_cfg: SceneEntityCfg,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    foot_contacts = (sensor.data.found > 0).float()
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    if top_height is None:
        top_height_tensor = _cube_top_height(env, obstacle_cfg) + top_height_offset
    else:
        top_height_tensor = torch.full_like(
            asset.data.root_link_pos_w[:, 0], top_height + top_height_offset
        )
    on_top_surface = (
        torch.abs(foot_heights - top_height_tensor.unsqueeze(1)) <= height_tolerance
    )
    top_contacts = foot_contacts * on_top_surface.float()
    return top_contacts.sum(dim=1) >= float(min_top_contacts)


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
    top_height: float | None,
    height_tolerance: float,
    approach_x: float,
    leave_x: float,
    min_forward_alignment: float = 0.5,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
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
    if top_height is None:
        top_height_tensor = _cube_top_height(env, obstacle_cfg) + top_height_offset
    else:
        top_height_tensor = torch.full_like(root_x, top_height + top_height_offset)
    on_top_surface = (
        torch.abs(foot_heights - top_height_tensor.unsqueeze(1)) <= height_tolerance
    )
    top_contacts = foot_contacts * on_top_surface.float()
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return top_contacts.mean(dim=1) * in_window.float() * heading_gate * heading_scale


def cube_top_heading_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    top_height: float | None,
    height_tolerance: float,
    min_top_contacts: int = 2,
    min_forward_alignment: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    """Reward keeping the original heading while stably standing on the cube top."""
    asset: Entity = env.scene[asset_cfg.name]
    stable_on_top = _stable_top_contacts(
        env,
        sensor_name,
        top_height,
        height_tolerance,
        min_top_contacts,
        asset_cfg,
        obstacle_cfg,
        top_height_offset,
    ).float()
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    return stable_on_top * heading_gate * forward_alignment


def cube_top_min_height_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    min_base_height_offset: float,
    top_height: float | None,
    height_tolerance: float,
    min_top_contacts: int = 2,
    terrain_height: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    """Penalize the base dropping below a minimum height while on the cube top."""
    asset: Entity = env.scene[asset_cfg.name]
    stable_on_top = _stable_top_contacts(
        env,
        sensor_name,
        top_height,
        height_tolerance,
        min_top_contacts,
        feet_asset_cfg,
        obstacle_cfg,
        top_height_offset,
    ).float()
    min_height = (
        _cube_top_height(env, obstacle_cfg, terrain_height) + min_base_height_offset
    )
    height_deficit = torch.clamp(min_height - asset.data.root_link_pos_w[:, 2], min=0.0)
    return stable_on_top * height_deficit


def cube_top_upright_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    top_height: float | None,
    height_tolerance: float,
    min_top_contacts: int = 2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    """Reward upright posture specifically while the robot is on top of the cube."""
    asset: Entity = env.scene[asset_cfg.name]
    stable_on_top = _stable_top_contacts(
        env,
        sensor_name,
        top_height,
        height_tolerance,
        min_top_contacts,
        feet_asset_cfg,
        obstacle_cfg,
        top_height_offset,
    ).float()
    upright = 1.0 - torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    upright = torch.clamp(upright, min=0.0, max=1.0)
    return stable_on_top * upright


def cube_top_step_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    top_height: float | None,
    height_tolerance: float,
    swing_clearance: float = 0.03,
    min_top_contacts: int = 2,
    min_forward_speed: float = 0.15,
    min_forward_alignment: float = 0.7,
    asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    """Reward lifting swing feet above the cube top while moving forward on it."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    stable_on_top = _stable_top_contacts(
        env,
        sensor_name,
        top_height,
        height_tolerance,
        min_top_contacts,
        asset_cfg,
        obstacle_cfg,
        top_height_offset,
    ).float()
    if top_height is None:
        top_height_tensor = _cube_top_height(env, obstacle_cfg) + top_height_offset
    else:
        top_height_tensor = torch.full_like(
            asset.data.root_link_pos_w[:, 0], top_height + top_height_offset
        )

    foot_contacts = (sensor.data.found > 0).float()
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    swing_feet = foot_contacts <= 0.0
    lifted_swing = swing_feet & (
        foot_heights >= (top_height_tensor.unsqueeze(1) + swing_clearance)
    )

    forward_speed = torch.clamp(asset.data.root_link_lin_vel_w[:, 0], min=0.0)
    moving_gate = (forward_speed >= min_forward_speed).float()
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    return stable_on_top * moving_gate * heading_gate * lifted_swing.float().mean(dim=1)


def cube_top_clearance_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_clearance: float,
    top_height: float | None,
    height_tolerance: float,
    min_top_contacts: int = 2,
    min_forward_speed: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    """Reward swing-foot clearance relative to the current cube top."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    stable_on_top = _stable_top_contacts(
        env,
        sensor_name,
        top_height,
        height_tolerance,
        min_top_contacts,
        asset_cfg,
        obstacle_cfg,
        top_height_offset,
    ).float()
    if top_height is None:
        top_height_tensor = _cube_top_height(env, obstacle_cfg) + top_height_offset
    else:
        top_height_tensor = torch.full_like(
            asset.data.root_link_pos_w[:, 0], top_height + top_height_offset
        )

    foot_contacts = (sensor.data.found > 0).float()
    swing_feet = foot_contacts <= 0.0
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    rel_height = foot_heights - top_height_tensor.unsqueeze(1)
    clearance_error = torch.abs(rel_height - target_clearance)
    swing_reward = torch.exp(-torch.square(clearance_error) / max(height_tolerance**2, 1.0e-6))
    swing_reward = (swing_reward * swing_feet.float()).mean(dim=1)

    forward_speed = torch.clamp(asset.data.root_link_lin_vel_w[:, 0], min=0.0)
    moving_gate = (forward_speed >= min_forward_speed).float()
    return stable_on_top * moving_gate * swing_reward


def cube_top_gait_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    period: float,
    offset: list[float],
    threshold: float,
    top_height: float | None,
    height_tolerance: float,
    min_top_contacts: int = 2,
    min_forward_speed: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    """Reward a desired contact timing pattern while walking on the cube top."""
    stable_on_top = _stable_top_contacts(
        env,
        sensor_name,
        top_height,
        height_tolerance,
        min_top_contacts,
        asset_cfg,
        obstacle_cfg,
        top_height_offset,
    ).float()
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = leg_phase < threshold
    gait_reward = (is_stance == is_contact).float().mean(dim=1)

    robot: Entity = env.scene[asset_cfg.name]
    forward_speed = torch.clamp(robot.data.root_link_lin_vel_w[:, 0], min=0.0)
    moving_gate = (forward_speed >= min_forward_speed).float()
    return stable_on_top * moving_gate * gait_reward


def cube_top_slip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    top_height: float | None,
    height_tolerance: float,
    min_top_contacts: int = 2,
    min_forward_speed: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
    top_height_offset: float = 0.0,
) -> torch.Tensor:
    """Penalize foot slip while the feet are in contact with the cube top."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    stable_on_top = _stable_top_contacts(
        env,
        sensor_name,
        top_height,
        height_tolerance,
        min_top_contacts,
        asset_cfg,
        obstacle_cfg,
        top_height_offset,
    ).float()
    foot_contacts = (sensor.data.found > 0).float()
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    if top_height is None:
        top_height_tensor = _cube_top_height(env, obstacle_cfg) + top_height_offset
    else:
        top_height_tensor = torch.full_like(
            asset.data.root_link_pos_w[:, 0], top_height + top_height_offset
        )
    on_top_surface = (
        torch.abs(foot_heights - top_height_tensor.unsqueeze(1)) <= height_tolerance
    )
    top_contacts = foot_contacts * on_top_surface.float()
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    slip_cost = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * top_contacts, dim=1)

    forward_speed = torch.clamp(asset.data.root_link_lin_vel_w[:, 0], min=0.0)
    moving_gate = (forward_speed >= min_forward_speed).float()
    return stable_on_top * moving_gate * slip_cost


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


def contact_count_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize arbitrary contact counts reported by a contact sensor."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return (sensor.data.found > 0).float().sum(dim=1)


def conditional_com_height_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height_offset: float,
    std: float = 0.08,
    terrain_height: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
) -> torch.Tensor:
    """Reward COM height only around the obstacle zone.

    The target COM height tracks the current cube top plus a fixed offset.
    """
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    root_z = asset.data.root_link_pos_w[:, 2]
    active = (root_x >= start_x) & (root_x <= end_x)
    target_height = _cube_top_height(env, obstacle_cfg, terrain_height) + target_height_offset

    height_error = root_z - target_height
    reward = torch.exp(-torch.square(height_error) / max(std**2, 1.0e-6))
    return reward * active.float()


class cube_top_goal_progress:
    """Reward forward progress toward the goal while maintaining top contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.prev_progress = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        start_x: float,
        goal_x: float,
        top_height: float | None,
        height_tolerance: float,
        min_top_contacts: int = 2,
        asset_cfg: SceneEntityCfg = _DEFAULT_ALL_FEET_CFG,
        obstacle_cfg: SceneEntityCfg = _DEFAULT_OBSTACLE_CFG,
        top_height_offset: float = 0.0,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        just_reset = env.episode_length_buf <= 1
        self.prev_progress[just_reset] = 0.0

        root_x = asset.data.root_link_pos_w[:, 0]
        stable_on_top = _stable_top_contacts(
            env,
            sensor_name,
            top_height,
            height_tolerance,
            min_top_contacts,
            asset_cfg,
            obstacle_cfg,
            top_height_offset,
        )

        total_distance = max(goal_x - start_x, 1.0e-6)
        progress = torch.clamp((root_x - start_x) / total_distance, min=0.0, max=1.0)
        reward = torch.where(
            stable_on_top,
            torch.clamp(progress - self.prev_progress, min=0.0),
            torch.zeros_like(progress),
        )
        self.prev_progress = torch.where(
            stable_on_top,
            torch.maximum(self.prev_progress, progress),
            self.prev_progress,
        )
        return reward


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
