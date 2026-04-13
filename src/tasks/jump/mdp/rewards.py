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


def _asset_forward_position(
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if asset_cfg.body_ids:
        return asset.data.body_link_pos_w[:, asset_cfg.body_ids, 0].mean(dim=1)
    return asset.data.root_link_pos_w[:, 0]


def _asset_height(
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if asset_cfg.body_ids:
        return asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].mean(dim=1)
    return asset.data.root_link_pos_w[:, 2]


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
    foot_indices: tuple[int, ...] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward placing feet on the cube while traversing the obstacle."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    in_window = (root_x >= approach_x) & (root_x <= leave_x)
    foot_contacts = (sensor.data.found > 0).float()
    if foot_indices is not None:
        foot_contacts = foot_contacts[:, foot_indices]
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


def pre_obstacle_front_foot_lift_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height: float,
    sensor_name: str | None = None,
    std: float = 0.05,
    min_forward_alignment: float = 0.5,
    alignment_power: float = 2.0,
    require_no_contact: bool = True,
    asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
) -> torch.Tensor:
    """Reward lifting the front feet before the obstacle while still approaching cleanly."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    if require_no_contact and sensor_name is not None:
        sensor: ContactSensor = env.scene[sensor_name]
        assert sensor.data.found is not None
        no_contact_yet = ~(sensor.data.found > 0).any(dim=1)
        active = active & no_contact_yet

    foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    height_error = foot_height - target_height
    lift_reward = torch.exp(-torch.square(height_error) / max(std**2, 1.0e-6)).mean(
        dim=1
    )
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return lift_reward * active.float() * heading_gate * heading_scale


def front_swing_clearance_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height: float,
    ground_sensor_name: str,
    cube_sensor_name: str | None = None,
    std: float = 0.04,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    foot_indices: tuple[int, ...] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
) -> torch.Tensor:
    """Reward front-foot swing clearance above a minimum height in the approach window."""
    asset: Entity = env.scene[asset_cfg.name]
    ground_sensor: ContactSensor = env.scene[ground_sensor_name]
    assert ground_sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    swing_mask = ground_sensor.data.found == 0
    if foot_indices is not None:
        swing_mask = swing_mask[:, foot_indices]

    if cube_sensor_name is not None:
        cube_sensor: ContactSensor = env.scene[cube_sensor_name]
        assert cube_sensor.data.found is not None
        cube_contact = cube_sensor.data.found > 0
        if foot_indices is not None:
            cube_contact = cube_contact[:, foot_indices]
        swing_mask = swing_mask & (~cube_contact)

    foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    height_reward = torch.clamp(
        (foot_height - target_height) / max(std, 1.0e-6),
        min=0.0,
    )
    swing_reward = (height_reward * swing_mask.float()).sum(dim=1)
    swing_count = swing_mask.float().sum(dim=1)
    swing_reward = swing_reward / torch.clamp(swing_count, min=1.0)
    swing_reward = swing_reward * (swing_count > 0).float()

    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return swing_reward * active.float() * heading_gate * heading_scale


def swing_contact_penalty_before_obstacle(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    ground_sensor_name: str,
    cube_sensor_name: str | None = None,
    foot_indices: tuple[int, ...] | None = None,
    min_forward_alignment: float = 0.5,
    alignment_power: float = 1.0,
    desired_contact_height: float | None = None,
    contact_margin: float = 0.015,
    asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
) -> torch.Tensor:
    """Penalize front-foot contact in the obstacle approach window when lift is desired."""
    asset: Entity = env.scene[asset_cfg.name]
    ground_sensor: ContactSensor = env.scene[ground_sensor_name]
    assert ground_sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    ground_contact = ground_sensor.data.found > 0
    if foot_indices is not None:
        ground_contact = ground_contact[:, foot_indices]

    contact_mask = ground_contact
    if cube_sensor_name is not None:
        cube_sensor: ContactSensor = env.scene[cube_sensor_name]
        assert cube_sensor.data.found is not None
        cube_contact = cube_sensor.data.found > 0
        if foot_indices is not None:
            cube_contact = cube_contact[:, foot_indices]
        contact_mask = contact_mask | cube_contact

    if desired_contact_height is not None:
        foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
        should_be_lifted = foot_height < (desired_contact_height - contact_margin)
        contact_mask = contact_mask & should_be_lifted

    penalty = contact_mask.float().mean(dim=1)
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return penalty * active.float() * heading_gate * heading_scale


def front_support_rear_lift_penalty(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    ground_sensor_name: str,
    cube_sensor_name: str | None = None,
    front_foot_indices: tuple[int, ...] | None = None,
    rear_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    rear_height_threshold: float = 0.10,
    rear_height_scale: float = 0.08,
    body_height_threshold: float = 0.34,
    body_height_scale: float = 0.08,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize using front-foot support to lever the rear/body upward before a clean takeoff."""
    asset: Entity = env.scene[asset_cfg.name]
    ground_sensor: ContactSensor = env.scene[ground_sensor_name]
    assert ground_sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    front_contact = ground_sensor.data.found > 0
    if front_foot_indices is not None:
        front_contact = front_contact[:, front_foot_indices]

    if cube_sensor_name is not None:
        cube_sensor: ContactSensor = env.scene[cube_sensor_name]
        assert cube_sensor.data.found is not None
        cube_front_contact = cube_sensor.data.found > 0
        if front_foot_indices is not None:
            cube_front_contact = cube_front_contact[:, front_foot_indices]
        front_contact = front_contact | cube_front_contact

    front_support = front_contact.float().mean(dim=1)
    rear_mean_height = asset.data.site_pos_w[:, rear_asset_cfg.site_ids, 2].mean(dim=1)
    rear_lift = torch.clamp(
        (rear_mean_height - rear_height_threshold) / max(rear_height_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    body_height = asset.data.root_link_pos_w[:, 2]
    body_raise = torch.clamp(
        (body_height - body_height_threshold) / max(body_height_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return (
        front_support
        * rear_lift
        * body_raise
        * active.float()
        * heading_gate
        * heading_scale
    )


def pre_obstacle_front_up_rear_down_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    front_target_height: float,
    rear_target_max_height: float = 0.10,
    front_height_std: float = 0.05,
    rear_height_scale: float = 0.06,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    front_asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
    rear_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward a pre-obstacle posture with the front lifted and the rear kept low."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    front_heights = asset.data.site_pos_w[:, front_asset_cfg.site_ids, 2]
    rear_heights = asset.data.site_pos_w[:, rear_asset_cfg.site_ids, 2]

    front_up = torch.clamp(
        (front_heights - front_target_height) / max(front_height_std, 1.0e-6),
        min=0.0,
    ).mean(dim=1)

    rear_mean_height = rear_heights.mean(dim=1)
    rear_down = 1.0 - torch.clamp(
        (rear_mean_height - rear_target_max_height) / max(rear_height_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return front_up * rear_down * active.float() * heading_gate * heading_scale


def pre_obstacle_body_height_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height: float,
    std: float = 0.05,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward raising the body in a specific obstacle-relative window."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    root_z = asset.data.root_link_pos_w[:, 2]
    active = (root_x >= start_x) & (root_x <= end_x)

    height_error = root_z - target_height
    height_reward = torch.exp(-torch.square(height_error) / max(std**2, 1.0e-6))
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return height_reward * active.float() * heading_gate * heading_scale


class body_region_clearance_reward:
    """Reward lifting a body region relative to its reset-time standing height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self.baseline_height = _asset_height(asset, cfg.params["asset_cfg"]).clone()

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        start_x: float,
        end_x: float,
        target_height: float,
        height_scale: float = 0.05,
        min_forward_alignment: float = 0.6,
        alignment_power: float = 2.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Reward a body region being lifted above its baseline by target_height."""
        asset: Entity = env.scene[asset_cfg.name]
        just_reset = env.episode_length_buf <= 1
        current_height = _asset_height(asset, asset_cfg)
        self.baseline_height[just_reset] = current_height[just_reset]

        root_x = asset.data.root_link_pos_w[:, 0]
        active = (root_x >= start_x) & (root_x <= end_x)
        target = self.baseline_height + target_height
        clearance_reward = torch.clamp(
            (current_height - target) / max(height_scale, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
        heading_gate = (forward_alignment >= min_forward_alignment).float()
        heading_scale = torch.pow(forward_alignment, alignment_power)
        return clearance_reward * active.float() * heading_gate * heading_scale


class body_region_relative_height_reward:
    """Reward the front body region rising above the rear relative to reset posture."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        front_asset_cfg = cfg.params["front_asset_cfg"]
        rear_asset_cfg = cfg.params["rear_asset_cfg"]
        self.baseline_gap = (
            _asset_height(asset, front_asset_cfg) - _asset_height(asset, rear_asset_cfg)
        ).clone()

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        start_x: float,
        end_x: float,
        target_gap: float,
        gap_scale: float = 0.05,
        min_forward_alignment: float = 0.6,
        alignment_power: float = 2.0,
        front_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        rear_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Reward the front body region staying higher than the rear by extra target_gap."""
        asset: Entity = env.scene[asset_cfg.name]
        just_reset = env.episode_length_buf <= 1
        front_height = _asset_height(asset, front_asset_cfg)
        rear_height = _asset_height(asset, rear_asset_cfg)
        current_gap = front_height - rear_height
        self.baseline_gap[just_reset] = current_gap[just_reset]

        root_x = asset.data.root_link_pos_w[:, 0]
        active = (root_x >= start_x) & (root_x <= end_x)
        target = self.baseline_gap + target_gap
        relative_reward = torch.clamp(
            (current_gap - target) / max(gap_scale, 1.0e-6),
            min=0.0,
            max=1.0,
        )

        forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
        heading_gate = (forward_alignment >= min_forward_alignment).float()
        heading_scale = torch.pow(forward_alignment, alignment_power)
        return relative_reward * active.float() * heading_gate * heading_scale


def pre_obstacle_twist_penalty(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    heading_weight: float = 1.0,
    yaw_rate_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize yaw twisting in an obstacle-relative approach window."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    heading_error = 1.0 - torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    yaw_rate = torch.square(asset.data.root_link_ang_vel_b[:, 2])
    penalty = (heading_weight * heading_error) + (yaw_rate_weight * yaw_rate)
    return penalty * active.float()


def rear_stance_push_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    ground_sensor_name: str,
    front_foot_indices: tuple[int, ...],
    rear_foot_indices: tuple[int, ...],
    front_target_height: float,
    front_min_height: float = 0.05,
    target_speed: float = 1.2,
    target_rear_force: float = 120.0,
    min_front_over_rear_height: float = 0.03,
    front_over_rear_height_scale: float = 0.08,
    front_over_rear_bonus_weight: float = 1.0,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    front_asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
    rear_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward a takeoff pattern with front-feet lift and rear-feet push."""
    robot: Entity = env.scene[asset_cfg.name]
    ground_sensor: ContactSensor = env.scene[ground_sensor_name]
    sensor_data = ground_sensor.data
    assert sensor_data.found is not None
    assert sensor_data.force is not None

    root_x = robot.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    front_heights = robot.data.site_pos_w[:, front_asset_cfg.site_ids, 2]
    front_lift = torch.clamp(
        (front_heights - front_min_height)
        / max(front_target_height - front_min_height, 1.0e-6),
        min=0.0,
        max=1.0,
    ).mean(dim=1)
    rear_heights = robot.data.site_pos_w[:, rear_asset_cfg.site_ids, 2]
    front_mean_height = front_heights.mean(dim=1)
    rear_mean_height = rear_heights.mean(dim=1)
    front_over_rear = torch.clamp(
        (front_mean_height - rear_mean_height - min_front_over_rear_height)
        / max(front_over_rear_height_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    rear_contact = (sensor_data.found[:, rear_foot_indices] > 0).float()
    rear_stance = rear_contact.mean(dim=1)
    rear_vertical_force = torch.clamp(sensor_data.force[:, rear_foot_indices, 2], min=0.0)
    rear_push = torch.clamp(
        rear_vertical_force / max(target_rear_force, 1.0e-6),
        min=0.0,
        max=1.0,
    ).mean(dim=1)

    front_contact = (sensor_data.found[:, front_foot_indices] > 0).float()
    front_clear = 1.0 - front_contact.mean(dim=1)
    forward_speed = torch.clamp(
        robot.data.root_link_lin_vel_w[:, 0], min=0.0, max=target_speed
    ) / max(target_speed, 1.0e-6)
    forward_alignment = torch.clamp(torch.cos(robot.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)

    reward = (
        front_lift
        * front_clear
        * rear_stance
        * rear_push
        * forward_speed
        * active.float()
        * heading_gate
        * heading_scale
    )
    return reward * (1.0 + front_over_rear_bonus_weight * front_over_rear)


def rear_air_penalty_before_takeoff(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    ground_sensor_name: str,
    front_foot_indices: tuple[int, ...],
    rear_foot_indices: tuple[int, ...],
    front_target_height: float,
    front_min_height: float = 0.05,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    front_asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize rear feet lifting too early before the push-off completes."""
    robot: Entity = env.scene[asset_cfg.name]
    ground_sensor: ContactSensor = env.scene[ground_sensor_name]
    assert ground_sensor.data.found is not None

    root_x = robot.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    front_heights = robot.data.site_pos_w[:, front_asset_cfg.site_ids, 2]
    front_lift = torch.clamp(
        (front_heights - front_min_height)
        / max(front_target_height - front_min_height, 1.0e-6),
        min=0.0,
        max=1.0,
    ).mean(dim=1)

    front_contact = (ground_sensor.data.found[:, front_foot_indices] > 0).float()
    front_clear = 1.0 - front_contact.mean(dim=1)
    rear_in_air = (ground_sensor.data.found[:, rear_foot_indices] == 0).float().mean(dim=1)

    forward_alignment = torch.clamp(torch.cos(robot.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)

    return (
        rear_in_air
        * front_lift
        * front_clear
        * active.float()
        * heading_gate
        * heading_scale
    )


def pre_obstacle_full_air_penalty(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    ground_sensor_name: str,
    cube_sensor_name: str | None = None,
    min_air_feet: int = 4,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize jumping with too many feet airborne before front-foot placement is established."""
    asset: Entity = env.scene[asset_cfg.name]
    ground_sensor: ContactSensor = env.scene[ground_sensor_name]
    assert ground_sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    ground_in_air = ground_sensor.data.found <= 0
    air_count = ground_in_air.sum(dim=1)
    too_many_airborne = air_count >= min_air_feet

    if cube_sensor_name is not None:
        cube_sensor: ContactSensor = env.scene[cube_sensor_name]
        assert cube_sensor.data.found is not None
        cube_touch = (cube_sensor.data.found > 0).any(dim=1)
        too_many_airborne = too_many_airborne & (~cube_touch)

    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return too_many_airborne.float() * active.float() * heading_gate * heading_scale


def rear_above_front_penalty(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    min_height_gap: float = 0.01,
    height_scale: float = 0.06,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    front_asset_cfg: SceneEntityCfg = _DEFAULT_FRONT_FEET_CFG,
    rear_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize pre-obstacle poses where the rear feet rise above the front feet."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    front_mean_height = asset.data.site_pos_w[:, front_asset_cfg.site_ids, 2].mean(dim=1)
    rear_mean_height = asset.data.site_pos_w[:, rear_asset_cfg.site_ids, 2].mean(dim=1)
    rear_over_front = torch.clamp(
        (rear_mean_height - front_mean_height - min_height_gap)
        / max(height_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return rear_over_front * active.float() * heading_gate * heading_scale


def rear_swing_clearance_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height: float,
    ground_sensor_name: str,
    cube_sensor_name: str | None = None,
    std: float = 0.05,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    foot_indices: tuple[int, ...] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward rear-foot swing clearance while crossing the obstacle."""
    asset: Entity = env.scene[asset_cfg.name]
    ground_sensor: ContactSensor = env.scene[ground_sensor_name]
    assert ground_sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    swing_mask = ground_sensor.data.found == 0
    if foot_indices is not None:
        swing_mask = swing_mask[:, foot_indices]

    if cube_sensor_name is not None:
        cube_sensor: ContactSensor = env.scene[cube_sensor_name]
        assert cube_sensor.data.found is not None
        cube_contact = cube_sensor.data.found > 0
        if foot_indices is not None:
            cube_contact = cube_contact[:, foot_indices]
        swing_mask = swing_mask & (~cube_contact)

    foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    height_error = foot_height - target_height
    height_reward = torch.exp(-torch.square(height_error) / max(std**2, 1.0e-6))
    swing_reward = (height_reward * swing_mask.float()).sum(dim=1)
    swing_count = swing_mask.float().sum(dim=1)
    swing_reward = swing_reward / torch.clamp(swing_count, min=1.0)
    swing_reward = swing_reward * (swing_count > 0).float()

    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return swing_reward * active.float() * heading_gate * heading_scale


def obstacle_top_stall_penalty(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    cube_sensor_name: str,
    speed_threshold: float = 0.2,
    min_contacts: int = 2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize getting stuck on top of the obstacle after mounting it."""
    asset: Entity = env.scene[asset_cfg.name]
    cube_sensor: ContactSensor = env.scene[cube_sensor_name]
    assert cube_sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)
    num_contacts = (cube_sensor.data.found > 0).sum(dim=1)
    low_speed = asset.data.root_link_lin_vel_w[:, 0] < speed_threshold
    stuck = (num_contacts >= min_contacts) & low_speed
    return stuck.float() * active.float()


def crossing_body_height_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_height: float,
    std: float = 0.05,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward extra body height while the robot is traversing the obstacle."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    root_z = asset.data.root_link_pos_w[:, 2]
    active = (root_x >= start_x) & (root_x <= end_x)

    height_error = root_z - target_height
    height_reward = torch.exp(-torch.square(height_error) / max(std**2, 1.0e-6))
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return height_reward * active.float() * heading_gate * heading_scale


def post_mount_forward_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    cube_sensor_name: str,
    target_speed: float = 0.8,
    min_contacts: int = 2,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward drive after mounting the obstacle instead of stalling on top."""
    asset: Entity = env.scene[asset_cfg.name]
    cube_sensor: ContactSensor = env.scene[cube_sensor_name]
    assert cube_sensor.data.found is not None

    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)
    num_contacts = (cube_sensor.data.found > 0).sum(dim=1)
    mounted = num_contacts >= min_contacts
    forward_speed = torch.clamp(
        asset.data.root_link_lin_vel_w[:, 0],
        min=0.0,
        max=target_speed,
    ) / max(target_speed, 1.0e-6)
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return forward_speed * active.float() * mounted.float() * heading_gate * heading_scale


def post_landing_forward_velocity_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_speed: float = 0.9,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward recovering forward walking speed after descending from the obstacle."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    forward_speed = torch.clamp(
        asset.data.root_link_lin_vel_w[:, 0],
        min=0.0,
        max=target_speed,
    ) / max(target_speed, 1.0e-6)
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)
    return forward_speed * active.float() * heading_gate * heading_scale


def post_landing_body_pitch_penalty(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    pitch_threshold: float = 0.15,
    pitch_scale: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize pitching the body forward too much after obstacle descent."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    pitch_like = -asset.data.projected_gravity_b[:, 0]
    forward_pitch = torch.clamp(
        (pitch_like - pitch_threshold) / max(pitch_scale, 1.0e-6),
        min=0.0,
    )
    return forward_pitch * active.float()


def obstacle_jump_x_velocity_reward(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    target_speed: float = 1.2,
    pitch_threshold: float | None = None,
    min_forward_alignment: float = 0.6,
    alignment_power: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward x-velocity in the obstacle traversal window."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    forward_speed = torch.clamp(
        asset.data.root_link_lin_vel_w[:, 0],
        min=0.0,
        max=target_speed,
    ) / max(target_speed, 1.0e-6)
    forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
    heading_gate = (forward_alignment >= min_forward_alignment).float()
    heading_scale = torch.pow(forward_alignment, alignment_power)

    if pitch_threshold is not None:
        pitch_like = -asset.data.projected_gravity_b[:, 0]
        pitch_gate = (pitch_like >= pitch_threshold).float()
    else:
        pitch_gate = 1.0

    return forward_speed * active.float() * heading_gate * heading_scale * pitch_gate


def obstacle_front_hip_pos_penalty(
    env: ManagerBasedRlEnv,
    start_x: float,
    end_x: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize excessive front-hip deviation in an obstacle-relative window."""
    asset: Entity = env.scene[asset_cfg.name]
    root_x = asset.data.root_link_pos_w[:, 0]
    active = (root_x >= start_x) & (root_x <= end_x)

    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.mean(torch.square(diff), dim=1)
    return penalty * active.float()


class obstacle_leg_symmetry_penalty:
    """Penalize left-right asymmetry of leg joint motion in an obstacle window."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self.pair_ids: list[tuple[int, int]] = []
        for right_name, left_name in cfg.params["joint_pairs"]:
            right_ids, _ = asset.find_joints(right_name)
            left_ids, _ = asset.find_joints(left_name)
            assert len(right_ids) == 1, f"Expected one joint for pattern {right_name}"
            assert len(left_ids) == 1, f"Expected one joint for pattern {left_name}"
            self.pair_ids.append((right_ids[0], left_ids[0]))
        self.flip_right_sign = tuple(cfg.params.get("flip_right_sign", (False,) * len(self.pair_ids)))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        start_x: float,
        end_x: float,
        joint_pairs,
        flip_right_sign,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del joint_pairs, flip_right_sign
        asset: Entity = env.scene[asset_cfg.name]
        root_x = asset.data.root_link_pos_w[:, 0]
        active = (root_x >= start_x) & (root_x <= end_x)

        penalty = torch.zeros(env.num_envs, device=env.device)
        joint_delta = asset.data.joint_pos - asset.data.default_joint_pos
        for idx, (right_id, left_id) in enumerate(self.pair_ids):
            right_val = joint_delta[:, right_id]
            left_val = joint_delta[:, left_id]
            if self.flip_right_sign[idx]:
                right_val = -right_val
            penalty = penalty + torch.abs(right_val - left_val)

        penalty = penalty / max(len(self.pair_ids), 1)
        return penalty * active.float()


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


def post_crossing_upright_reward(
    env: ManagerBasedRlEnv,
    goal_x: float,
    min_upright: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward keeping the trunk upright after clearing the obstacle."""
    asset: Entity = env.scene[asset_cfg.name]
    crossed = asset.data.root_link_pos_w[:, 0] >= goal_x
    upright = 1.0 - torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    upright = torch.clamp(upright, min=0.0, max=1.0)
    upright_gate = (upright >= min_upright).float()
    return crossed.float() * upright * upright_gate


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
        self.obstacle_touched = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        start_x: float,
        goal_x: float,
        contact_sensor_name: str | None = None,
        contact_required_x: float | None = None,
        min_forward_alignment: float = 0.0,
        alignment_power: float = 1.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        just_reset = env.episode_length_buf <= 1
        self.prev_progress[just_reset] = 0.0
        self.obstacle_touched[just_reset] = False

        if contact_sensor_name is not None:
            sensor: ContactSensor = env.scene[contact_sensor_name]
            assert sensor.data.found is not None
            touched_now = (sensor.data.found > 0).any(dim=1)
            self.obstacle_touched |= touched_now

        forward_x = _asset_forward_position(asset, asset_cfg)
        effective_forward_x = forward_x
        if contact_required_x is not None:
            effective_forward_x = torch.where(
                self.obstacle_touched,
                effective_forward_x,
                torch.clamp(effective_forward_x, max=contact_required_x),
            )
        total_distance = max(goal_x - start_x, 1.0e-6)
        progress = torch.clamp(
            (effective_forward_x - start_x) / total_distance,
            min=0.0,
            max=1.0,
        )
        forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
        heading_gate = (forward_alignment >= min_forward_alignment).float()
        heading_scale = torch.pow(forward_alignment, alignment_power)
        reward = torch.clamp(progress - self.prev_progress, min=0.0)
        reward = reward * heading_gate * heading_scale
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
        self.obstacle_touched = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        goal_x: float,
        contact_sensor_name: str | None = None,
        min_forward_alignment: float = 0.0,
        alignment_power: float = 1.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        just_reset = env.episode_length_buf <= 1
        self.already_crossed[just_reset] = False
        self.obstacle_touched[just_reset] = False

        if contact_sensor_name is not None:
            sensor: ContactSensor = env.scene[contact_sensor_name]
            assert sensor.data.found is not None
            touched_now = (sensor.data.found > 0).any(dim=1)
            self.obstacle_touched |= touched_now

        forward_x = _asset_forward_position(asset, asset_cfg)
        crossed = forward_x >= goal_x
        forward_alignment = torch.clamp(torch.cos(asset.data.heading_w), min=0.0, max=1.0)
        heading_gate = forward_alignment >= min_forward_alignment
        heading_scale = torch.pow(forward_alignment, alignment_power)
        contact_gate = self.obstacle_touched if contact_sensor_name is not None else True
        reward = (
            (~self.already_crossed & crossed & heading_gate & contact_gate).float()
            * heading_scale
        )
        self.already_crossed |= crossed

        env.extras["log"]["Metrics/obstacle_crossed_ratio"] = crossed.float().mean()
        return reward
