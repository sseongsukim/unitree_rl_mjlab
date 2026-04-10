"""Unitree Go2 leap environment configurations."""

from src.assets.robots import get_go2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import src.tasks.leap.mdp as mdp
from src.tasks.leap.config.go2.terrains import make_leap_gap_terrain_cfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg


def unitree_go2_leap_env_cfg_old(
    play: bool = False,
    terrain_size: tuple[float, float] = (16.0, 4.0),
    num_rows: int = 8,
    num_cols: int = 8,
    gap_width_range: tuple[float, float] = (0.20, 0.50),
    spawn_offset_from_gap: float = 1.5,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 leap environment with a terrain gap."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.sim.mujoco.ccd_iterations = 300
    cfg.sim.contact_sensor_maxmatch = 300

    # ==================== Scene ==================== #
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.scene.terrain.terrain_generator = make_leap_gap_terrain_cfg(
        size=terrain_size,
        num_rows=num_rows,
        num_cols=num_cols,
        gap_width_range=gap_width_range,
        approach_length=7.0,
        landing_length=5.0,
        spawn_offset_from_gap=spawn_offset_from_gap,
    )
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.curriculum.pop("terrain_levels", None)

    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        debug_vis=True,
        viz=RayCastSensorCfg.VizCfg(show_normals=True),
    )
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    ) + (terrain_scan,)

    cfg.scene.entities = {
        "robot": get_go2_robot_cfg(),
    }

    # ==================== Observations ==================== #
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        scale=1 / terrain_scan.max_distance,
    )
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        scale=1 / terrain_scan.max_distance,
    )

    # ==================== Rewards ==================== #
    cfg.rewards["pose"].weight = 0.0
    cfg.rewards["foot_gait"].weight = 0.3
    cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.0, 0.5, 0.5]
    cfg.rewards["foot_gait"].params["period"] = 0.6
    cfg.rewards["foot_gait"].params["threshold"] = 0.56

    cfg.rewards.pop("foot_clearance", None)

    cfg.rewards["world_x_velocity"] = RewardTermCfg(
        func=mdp.world_x_velocity_reward,
        weight=1.5,
        params={
            "clamp_min": 0.0,
            "clamp_max": 1.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["landing_patch_approach"] = RewardTermCfg(
        func=mdp.landing_patch_approach_velocity,
        weight=3.0,
        params={
            "patch_name": "landing",
            "clamp_min": 0.0,
            "clamp_max": 1.2,
            "y_margin": 0.45,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.rewards["landing_bonus"] = RewardTermCfg(
        func=mdp.terrain_landing_bonus,
        weight=20.0,
        params={
            "sensor_name": "feet_ground_contact",
            "patch_name": "landing",
            "y_margin": 0.25,
            "contact_indices": (2, 3),
            "min_contacts": 2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["foot_lift_near_gap"] = RewardTermCfg(
        func=mdp.terrain_gap_foot_lift,
        weight=3.0,
        params={
            "spawn_patch_name": "spawn",
            "landing_patch_name": "landing",
            "target_height": 0.14,
            "pre_margin_x": 0.50,
            "post_margin_x": 0.20,
            "y_margin": 0.20,
            "height_tolerance": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot",
                site_names=("FR", "FL", "RR", "RL"),
            ),
        },
    )

    cfg.rewards["com_yaw"] = RewardTermCfg(
        func=mdp.com_yaw_reward,
        weight=-1.0,
        params={
            "desired_yaw": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ==================== Commands ==================== #
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.heading_command = False
    twist_cmd.ranges.lin_vel_x = (0.5, 1.6)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)
    twist_cmd.ranges.heading = None

    if "command_vel" in cfg.curriculum:
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
            {
                "step": 0,
                "lin_vel_x": (0.5, 1.6),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-0.0, 0.0),
            },
            {
                "step": 5000 * 24,
                "lin_vel_x": (1.0, 2.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-0.0, 0.0),
            },
        ]

    # ==================== Events ==================== #
    cfg.events["reset_base"] = EventTermCfg(
        func=envs_mdp.reset_root_state_from_flat_patches,
        mode="reset",
        params={
            "patch_name": "spawn",
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events.pop("push_robot", None)

    # ==================== Terminations ==================== #
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_touch"},
    )

    cfg.terminations["backward_x_velocity"] = TerminationTermCfg(
        func=mdp.backward_x_velocity,
        params={
            "threshold": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations["insufficient_x_progress"] = TerminationTermCfg(
        func=mdp.insufficient_x_progress,
        params={
            "min_progress": 0.08,
            "grace_period_s": 1.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations["landing_progress"] = TerminationTermCfg(
        func=mdp.landing_progress,
        params={
            "sensor_name": "feet_ground_contact",
            "patch_name": "landing",
            "forward_distance": 0.4,
            "y_margin": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    if play and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.num_envs = 1
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_rows = 4
        cfg.scene.terrain.terrain_generator.num_cols = 4

    return cfg


def unitree_go2_leap_env_cfg(
    play: bool = False,
    terrain_size: tuple[float, float] = (16.0, 4.0),
    num_rows: int = 10,
    num_cols: int = 10,
    gap_width_range: tuple[float, float] = (0.20, 0.80),
    spawn_offset_from_gap: float = 2.0,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 leap environment with a terrain gap."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.sim.mujoco.ccd_iterations = 300
    cfg.sim.contact_sensor_maxmatch = 300

    # ==================== Scene ==================== #
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.scene.terrain.terrain_generator = make_leap_gap_terrain_cfg(
        size=terrain_size,
        num_rows=num_rows,
        num_cols=num_cols,
        gap_width_range=gap_width_range,
        approach_length=7.0,
        landing_length=5.0,
        spawn_offset_from_gap=spawn_offset_from_gap,
    )
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.curriculum.pop("terrain_levels", None)

    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        debug_vis=True,
        viz=RayCastSensorCfg.VizCfg(show_normals=True),
    )
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    ) + (terrain_scan,)

    cfg.scene.entities = {
        "robot": get_go2_robot_cfg(),
    }

    # ==================== Observations ==================== #
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        scale=1 / terrain_scan.max_distance,
    )
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        scale=1 / terrain_scan.max_distance,
    )

    # ==================== Rewards ==================== #
    # cfg.rewards["track_linear_velocity"].weight = 0.25
    # cfg.rewards["track_angular_velocity"].weight = 0.05
    cfg.rewards["foot_gait"].weight = 0.25

    cfg.rewards.pop("foot_clearance", None)

    cfg.rewards["world_x_velocity"] = RewardTermCfg(
        func=mdp.world_x_velocity_reward,
        weight=0.5,
        params={
            "clamp_min": 0.0,
            "clamp_max": 1.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["landing_patch_approach"] = RewardTermCfg(
        func=mdp.landing_patch_approach_velocity,
        weight=5.0,
        params={
            "patch_name": "landing",
            "clamp_min": 0.0,
            "clamp_max": 1.2,
            "y_margin": 0.45,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.rewards["landing_bonus"] = RewardTermCfg(
        func=mdp.terrain_landing_bonus,
        weight=20.0,
        params={
            "sensor_name": "feet_ground_contact",
            "patch_name": "landing",
            "y_margin": 0.25,
            "contact_indices": (2, 3),
            "min_contacts": 2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["com_distance_to_goal_squared"] = RewardTermCfg(
        func=mdp.com_distance_to_goal_squared_reward,
        weight=2.0,
        params={
            "goal_patch_name": "landing",
            "start_patch_name": "spawn",
            "min_normalization_distance": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["com_height"] = RewardTermCfg(
        func=mdp.com_height_reward,
        weight=1.0,
        params={
            "terrain_height": 0.0,
            "max_height": 1.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["vertical_distance"] = RewardTermCfg(
        func=mdp.vertical_distance_reward,
        weight=1.0,
        params={
            "patch_name": "landing",
            "desired_base_height": 1.2,  # 0.8
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.rewards["com_yaw"] = RewardTermCfg(
        func=mdp.com_yaw_reward,
        weight=-1.0,
        params={
            "desired_yaw": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ==================== Commands ==================== #
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.heading_command = False
    twist_cmd.ranges.lin_vel_x = (0.5, 1.6)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)
    twist_cmd.ranges.heading = None

    if "command_vel" in cfg.curriculum:
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
            {
                "step": 0,
                "lin_vel_x": (0.5, 1.6),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-0.0, 0.0),
            },
            {
                "step": 5000 * 24,
                "lin_vel_x": (1.0, 2.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-0.0, 0.0),
            },
        ]

    # ==================== Events ==================== #
    cfg.events["reset_base"] = EventTermCfg(
        func=envs_mdp.reset_root_state_from_flat_patches,
        mode="reset",
        params={
            "patch_name": "spawn",
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events.pop("push_robot", None)

    # ==================== Terminations ==================== #
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_touch"},
    )

    cfg.terminations["backward_x_velocity"] = TerminationTermCfg(
        func=mdp.backward_x_velocity,
        params={
            "threshold": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations["insufficient_x_progress"] = TerminationTermCfg(
        func=mdp.insufficient_x_progress,
        params={
            "min_progress": 0.06,
            "grace_period_s": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.terminations["landing_progress"] = TerminationTermCfg(
        func=mdp.landing_progress,
        params={
            "sensor_name": "feet_ground_contact",
            "patch_name": "landing",
            "forward_distance": 0.20,
            "y_margin": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    if play and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.num_envs = 1
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_rows = 4
        cfg.scene.terrain.terrain_generator.num_cols = 4
        cfg.scene.terrain.terrain_generator.sub_terrains["gap"].gap_width_range = (
            0.70,
            0.70,
        )
        cfg.scene.terrain.terrain_generator.sub_terrains[
            "gap"
        ].spawn_offset_from_gap = 2.5
        cfg.terminations.pop("insufficient_x_progress", None)
        cfg.terminations.pop("backward_x_velocity", None)
        cfg.terminations.pop("illegal_contact", None)

    return cfg
