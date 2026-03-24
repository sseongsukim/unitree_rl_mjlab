"""Jump task configuration factory.

This module mirrors the structure of ``velocity_env_cfg.py`` by providing a
base factory that robot-specific jump tasks can customize.
"""

from copy import deepcopy

from dataclasses import dataclass

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    RayCastSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.jump.mdp as jump_mdp
import src.tasks.velocity.mdp as velocity_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


@dataclass(frozen=True)
class CubeObstacleOffsetCfg:
    offset: tuple[float, float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.75, 0.45, 0.20, 1.0)


DEFAULT_CUBE_OFFSET = CubeObstacleOffsetCfg(
    offset=(0.8, 0.0, 0.06),
    size=(0.12, 0.30, 0.06),
    rgba=(0.75, 0.45, 0.20, 1.0),
)


def _line_y_positions(
    env_ids: torch.Tensor,
    num_envs: int,
    spacing: float,
    center_y: float,
) -> torch.Tensor:
    env_ids_f = env_ids.to(dtype=torch.float32)
    center_idx = 0.5 * float(num_envs - 1)
    return (env_ids_f - center_idx) * spacing + center_y


def reset_robot_line(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    spacing: float,
    spawn_xy: tuple[float, float],
    shared_layout: bool,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place all robots on a centered line along y."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, 0] = spawn_xy[0]
    if shared_layout:
        root_states[:, 1] = spawn_xy[1]
    else:
        root_states[:, 1] = _line_y_positions(
            env_ids, env.num_envs, spacing, spawn_xy[1]
        )

    asset.write_root_link_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)


def reset_cube_line(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    spacing: float,
    robot_spawn_xy: tuple[float, float],
    cube_offset: CubeObstacleOffsetCfg,
    shared_layout: bool,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Place one cube in front of each robot using the same line layout."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, :3].clone()
    positions[:, 0] = robot_spawn_xy[0] + cube_offset.offset[0]
    if shared_layout:
        positions[:, 1] = robot_spawn_xy[1] + cube_offset.offset[1]
    else:
        positions[:, 1] = (
            _line_y_positions(env_ids, env.num_envs, spacing, robot_spawn_xy[1])
            + cube_offset.offset[1]
        )
    positions[:, 2] = cube_offset.offset[2]
    orientations = root_states[:, 3:7]

    asset.write_mocap_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )


def _get_cube_spec(
    size: tuple[float, float, float], rgba: tuple[float, float, float, float]
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        rgba=rgba,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(1.0, 0.005, 0.0001),
    )
    return spec


def get_jump_cube_cfg(
    cube_offset: CubeObstacleOffsetCfg = DEFAULT_CUBE_OFFSET,
) -> EntityCfg:
    """Create one fixed cube entity to be repositioned per-environment at reset."""
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spec_fn=lambda: _get_cube_spec(cube_offset.size, cube_offset.rgba),
    )


def make_jump_env_cfg(
    robot_cfg: EntityCfg,
    foot_names: tuple[str, ...],
    site_names: tuple[str, ...] | None = None,
    *,
    play: bool = False,
    robot_spawn_xy: tuple[float, float] = (0.0, 0.0),
    cube_offset: CubeObstacleOffsetCfg = DEFAULT_CUBE_OFFSET,
    line_spacing: float = 1.5,
    shared_layout: bool = True,
    base_body_name: str = "base_link",
) -> ManagerBasedRlEnvCfg:
    """Create a base jump environment config for a legged robot."""
    cfg = make_velocity_env_cfg()
    height_map_grid = (2.0, 1.0)
    height_map_resolution = 0.1
    obstacle_height = 2.0 * cube_offset.size[2]
    site_names = site_names or foot_names

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    cfg.scene.entities = {
        "robot": robot_cfg,
        "cube": get_jump_cube_cfg(cube_offset=cube_offset),
    }
    cfg.scene.env_spacing = line_spacing

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = base_body_name
            sensor.ray_alignment = "yaw"
            sensor.pattern = GridPatternCfg(
                size=height_map_grid, resolution=height_map_resolution
            )
            sensor.max_distance = 3.0
            sensor.debug_vis = play

    actor_height_map = cfg.observations["actor"].terms.pop("height_scan")
    actor_height_map.func = jump_mdp.obstacle_height_map
    actor_height_map.params = {
        "sensor_name": "terrain_scan",
        "grid_size": height_map_grid,
        "resolution": height_map_resolution,
        "x_range": (0.2, 1.0),
        "y_range": (-0.5, 0.5),
        "clamp_max": obstacle_height,
    }
    actor_height_map.noise = None
    actor_height_map.scale = 1.0
    cfg.observations["actor"].terms["height_map"] = actor_height_map

    critic_height_map = cfg.observations["critic"].terms.pop("height_scan")
    critic_height_map.func = jump_mdp.obstacle_height_map
    critic_height_map.params = {
        "sensor_name": "terrain_scan",
        "grid_size": height_map_grid,
        "resolution": height_map_resolution,
        "x_range": (0.2, 1.0),
        "y_range": (-0.5, 0.5),
        "clamp_max": obstacle_height,
    }
    critic_height_map.scale = 1.0
    cfg.observations["critic"].terms["height_map"] = critic_height_map

    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)
    obstacle_start_x = robot_spawn_xy[0] + cube_offset.offset[0] - cube_offset.size[0]
    obstacle_center_x = robot_spawn_xy[0] + cube_offset.offset[0]
    obstacle_end_x = robot_spawn_xy[0] + cube_offset.offset[0] + cube_offset.size[0]
    obstacle_goal_x = obstacle_end_x + 0.2

    critic_obstacle_state = deepcopy(critic_height_map)
    critic_obstacle_state.func = jump_mdp.privileged_obstacle_state
    critic_obstacle_state.params = {
        "front_x": obstacle_start_x,
        "center_x": obstacle_center_x,
        "goal_x": obstacle_goal_x,
        "target_y": robot_spawn_xy[1],
        "clamp_distance": 2.0,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    critic_obstacle_state.noise = None
    critic_obstacle_state.scale = 1.0
    cfg.observations["critic"].terms["obstacle_state"] = critic_obstacle_state

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=None,
        secondary_policy="any",
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision\d*$",
            exclude=tuple(geom_names),
        ),
        secondary=None,
        secondary_policy="any",
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    feet_cube_cfg = ContactSensorCfg(
        name="feet_cube_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
        secondary_policy="first",
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # Optional jump-specific sensor for knee / shank impacts.
    # Uncomment this together with the reward block below if you want a more
    # specific obstacle-contact penalty than `nonfoot_ground_touch`.
    #
    # knee_cube_cfg = ContactSensorCfg(
    #     name="knee_cube_contact",
    #     primary=ContactMatch(
    #         mode="geom",
    #         entity="robot",
    #         pattern=r".*(thigh|calf).*collision.*",
    #     ),
    #     secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
    #     secondary_policy="first",
    #     fields=("found",),
    #     reduce="none",
    #     num_slots=1,
    # )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
        feet_cube_cfg,
        # knee_cube_cfg,
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)

    cfg.viewer.body_name = base_body_name
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = (base_body_name,)

    cfg.rewards["pose"].params["std_standing"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
    }
    cfg.rewards["body_orientation_l2"].weight = -1.0
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].weight = 0.25
    cfg.rewards["body_orientation_l2"].weight = -0.5
    cfg.rewards["body_ang_vel"].weight = -0.025
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["stand_still"].weight = 0.0
    cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (
        base_body_name,
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (base_body_name,)
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["foot_clearance"].params["target_height"] = obstacle_height + 0.02
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
    cfg.rewards["track_linear_velocity"].params["std"] = 0.20
    cfg.rewards["track_angular_velocity"].params["std"] = 0.25

    cfg.rewards["centerline"] = RewardTermCfg(
        func=jump_mdp.centerline_penalty,
        weight=-1.0,
        params={
            "target_y": robot_spawn_xy[1],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["obstacle_progress"] = RewardTermCfg(
        func=jump_mdp.obstacle_progress,
        weight=20.0,
        params={
            "start_x": robot_spawn_xy[0],
            "goal_x": obstacle_goal_x,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["feet_on_cube"] = RewardTermCfg(
        func=jump_mdp.feet_on_obstacle,
        weight=10.0,
        params={
            "sensor_name": feet_cube_cfg.name,
            "approach_x": obstacle_start_x - 0.15,
            "leave_x": obstacle_goal_x,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["obstacle_crossing_bonus"] = RewardTermCfg(
        func=jump_mdp.obstacle_crossing_bonus,
        weight=100.0,
        params={
            "goal_x": obstacle_goal_x,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["obstacle_approach_speed"] = RewardTermCfg(
        func=jump_mdp.obstacle_approach_speed,
        weight=5.0,
        params={
            "goal_x": obstacle_goal_x,
            "target_speed": 1.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["obstacle_stall_penalty"] = RewardTermCfg(
        func=jump_mdp.obstacle_stall_penalty,
        weight=-3.0,
        params={
            "goal_x": obstacle_goal_x,
            "speed_threshold": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Optional jump-shaping rewards. These are intentionally left disabled so
    # they are easy to experiment with later without changing the current task.

    # cfg.rewards["knee_contact"] = RewardTermCfg(
    #     func=jump_mdp.knee_contact_penalty,
    #     weight=-4.0,
    #     params={"sensor_name": knee_cube_cfg.name},
    # )
    # cfg.rewards["conditional_com_height"] = RewardTermCfg(
    #     func=jump_mdp.conditional_com_height_reward,
    #     weight=3.0,
    #     params={
    #         "start_x": obstacle_start_x - 0.10,
    #         "end_x": obstacle_end_x + 0.10,
    #         "target_height": 0.34,
    #         "std": 0.06,
    #         "terrain_height": 0.0,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    # cfg.rewards["front_foot_clearance_over_obstacle"] = RewardTermCfg(
    #     func=jump_mdp.front_foot_clearance_over_obstacle,
    #     weight=2.0,
    #     params={
    #         "start_x": obstacle_start_x - 0.05,
    #         "end_x": obstacle_end_x + 0.10,
    #         "target_height": obstacle_height + 0.05,
    #         "std": 0.04,
    #         "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL")),
    #     },
    # )
    # cfg.rewards["distance_to_clear_goal"] = RewardTermCfg(
    #     func=jump_mdp.distance_to_clear_goal,
    #     weight=3.0,
    #     params={
    #         "goal_x": obstacle_goal_x,
    #         "clamp_distance": 2.0,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.resampling_time_range = (1.0e9, 1.0e9)
    twist_cmd.heading_command = False
    twist_cmd.ranges.heading = None
    twist_cmd.ranges.lin_vel_x = (1.2, 1.2)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    cfg.events["reset_base"] = EventTermCfg(
        func=reset_robot_line,
        mode="reset",
        params={
            "spacing": line_spacing,
            "spawn_xy": robot_spawn_xy,
            "shared_layout": shared_layout,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events["reset_cube"] = EventTermCfg(
        func=reset_cube_line,
        mode="reset",
        params={
            "spacing": line_spacing,
            "robot_spawn_xy": robot_spawn_xy,
            "cube_offset": cube_offset,
            "shared_layout": shared_layout,
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=velocity_mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name},
    )
    cfg.terminations["off_track"] = TerminationTermCfg(
        func=jump_mdp.off_track,
        params={
            "target_y": robot_spawn_xy[1],
            "max_offset": 0.45,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations["moved_backward"] = TerminationTermCfg(
        func=jump_mdp.moved_backward,
        params={
            "min_x": robot_spawn_xy[0] - 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg
