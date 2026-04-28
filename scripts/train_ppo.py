"""Script to train RL agent with RSL-RL."""

import copy
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from local_tasks import register_local_tasks

from src.rl_core.rsl_rl.rl.vecenv_wrapper import RslRlVecEnvWrapper
from src.rl_core.rsl_rl.rl.config import (
    RslRlBaseRunnerCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlRndCfg,
)
from src.rl_core.rsl_rl.rl.runner import MjlabOnPolicyRunner
import src.tasks.jump.mdp as jump_mdp

register_local_tasks()


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    use_rnd: bool = False
    use_height_map: bool = True
    symmetric_obs: bool = False
    motion_file: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def _apply_common_rnd_cfg(cfg: TrainConfig) -> None:
    if not cfg.use_rnd:
        return

    if not isinstance(cfg.agent, RslRlOnPolicyRunnerCfg):
        raise TypeError("RND is only supported for on-policy PPO configs.")

    final_step = cfg.agent.max_iterations * cfg.agent.num_steps_per_env
    cfg.agent.obs_groups["rnd_state"] = ("actor",)
    cfg.agent.algorithm.rnd_cfg = RslRlRndCfg(
        weight=1.0,
        weight_schedule={
            "mode": "linear",
            "initial_step": 0,
            "final_step": final_step,
            "final_value": 0.0,
        },
    )


def _apply_no_height_map_observations(env_cfg: ManagerBasedRlEnvCfg) -> None:
    """Replace jump height-map observations with compact base pose observations."""
    removed_any = False
    for group_name in ("actor", "critic"):
        obs_group = env_cfg.observations.get(group_name)
        if obs_group is None:
            continue
        had_height_map = obs_group.terms.pop("height_map", None) is not None
        removed_any = had_height_map or removed_any
        if not had_height_map:
            continue
        obs_group.terms.setdefault(
            "base_position",
            ObservationTermCfg(func=jump_mdp.base_position),
        )
        obs_group.terms.setdefault(
            "base_rotation",
            ObservationTermCfg(func=jump_mdp.base_yaw_pitch_roll),
        )

    if removed_any:
        env_cfg.scene.sensors = tuple(
            sensor
            for sensor in (env_cfg.scene.sensors or ())
            if sensor.name != "terrain_scan"
        )


def _apply_symmetric_observations(env_cfg: ManagerBasedRlEnvCfg) -> None:
    """Make actor observations use the critic observation layout."""
    critic_obs = env_cfg.observations.get("critic")
    if critic_obs is None:
        raise ValueError(
            "Cannot enable symmetric observations: critic observation group is missing."
        )
    env_cfg.observations["actor"] = copy.deepcopy(critic_obs)


def _metra_wrapper_kwargs(agent_cfg: RslRlBaseRunnerCfg) -> dict:
    algorithm = agent_cfg.algorithm
    class_name = getattr(algorithm, "class_name", "")
    if "metra" not in class_name.lower():
        return {}
    return {
        "use_skill": True,
        "discrete_skill": getattr(algorithm, "discrete_option"),
        "unit_length_skill": getattr(algorithm, "unit_length_option"),
        "num_dims": getattr(algorithm, "dim_option"),
    }


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        # Set EGL device to match the CUDA device.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        # Set seed to have diversity in different processes.
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = "motion" in cfg.env.commands and isinstance(
        cfg.env.commands["motion"], MotionCommandCfg
    )

    if is_tracking_task:
        if not cfg.motion_file:
            raise ValueError("For tracking tasks, --motion-file must be set ...")
        motion_path = Path(cfg.motion_file).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        motion_cmd = cfg.env.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.motion_file = str(motion_path)
        print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

        # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
        if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
            print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

    # Enable NaN guard if requested.
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(
            f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}"
        )

    if not cfg.use_height_map:
        _apply_no_height_map_observations(cfg.env)
        if rank == 0:
            print(
                "[INFO] Height-map observations disabled; base pose observations enabled for actor and critic."
            )

    if cfg.symmetric_obs:
        _apply_symmetric_observations(cfg.env)
        if rank == 0:
            print("[INFO] Symmetric observations enabled: actor uses critic observations.")

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    eval_env = None
    if rank == 0:
        eval_env_cfg = copy.deepcopy(cfg.env)
        eval_env_cfg.scene.num_envs = 1
        eval_env = ManagerBasedRlEnv(
            cfg=eval_env_cfg,
            device=device,
            render_mode="rgb_array" if cfg.video else None,
        )

    log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

    resume_path: Path | None = None
    if cfg.agent.resume:
        # Load checkpoint from local filesystem.
        resume_path = get_checkpoint_path(
            log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
        )

    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    wrapper_kwargs = _metra_wrapper_kwargs(cfg.agent)
    env = RslRlVecEnvWrapper(
        env, clip_actions=cfg.agent.clip_actions, **wrapper_kwargs
    )

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    runner_kwargs = {
        "eval_env": (
            RslRlVecEnvWrapper(
                eval_env, clip_actions=cfg.agent.clip_actions, **wrapper_kwargs
            )
            if eval_env is not None
            else None
        ),
        "eval_interval": cfg.agent.eval_interval,
        "eval_video": cfg.video and rank == 0,
        "eval_video_length": cfg.video_length,
    }
    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    # Only write config files from rank 0 to avoid race conditions.
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )

    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
    args = args or TrainConfig.from_task(task_id)
    _apply_common_rnd_cfg(args)

    # Create log directory once before launching workers.
    task_name = task_id.split("-")[-1].lower()
    log_root_path = Path("logs") / "rsl_rl" / task_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    # Set environment variables for all modes.
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        # CPU or single GPU: run directly without torchrunx.
        run_train(task_id, args, log_dir)
    else:
        # Multi-GPU: use torchrunx.
        import torchrunx

        # torchrunx redirects stdout to logging.
        logging.basicConfig(level=logging.INFO)

        # Configure torchrunx logging directory.
        # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                # User specified a value via flag (could be "" to disable).
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                # Default: put logs in training directory.
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let rsl_rl handle process group initialization.
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401
    import src.tasks

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
        default="Unitree-Go2-Jump",
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args

    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
