import copy
from dataclasses import asdict, dataclass, field
from typing import Optional
from pathlib import Path
import os
import time

import random
import numpy as np
import torch
import tqdm


from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg
from local_tasks import register_local_tasks

from src.rl_core.flash_rl.agents import create_agent
from src.rl_core.flash_rl.envs.mjlab import FlashVecEnvWrapper
from src.rl_core.flash_rl.common.config import FlashRlCfg
from src.rl_core.flash_rl.types import Tensor
from src.rl_core.flash_rl.common.logger import WandbTrainerLogger

register_local_tasks()


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: FlashRlCfg

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = FlashRlCfg()
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def evaluation(
    agent,
    eval_env: FlashVecEnvWrapper,
    interaction_step: int,
    logger: WandbTrainerLogger,
) -> None:
    def _extract_rgb_frame(frame) -> np.ndarray | None:
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3:
            return None
        if frame.shape[-1] == 3:
            return frame
        if frame.shape[0] == 3:
            return np.transpose(frame, (1, 2, 0))
        return None

    observations, _ = eval_env.reset()
    prev_transition: dict[str, Tensor] = {"next_observation": observations}

    episode_return = 0.0
    episode_length = 0
    done = False
    video_frames: list[np.ndarray] = []

    initial_frame = _extract_rgb_frame(eval_env.render())
    if initial_frame is not None:
        video_frames.append(initial_frame)

    while not done and episode_length < eval_env.max_episode_length:
        with torch.no_grad():
            actions = agent.sample_actions(
                interaction_step=interaction_step,
                prev_transition=prev_transition,
                training=False,
            )
        next_observations, rewards, terminateds, truncateds, _ = eval_env.step(actions)

        reward_value = float(
            rewards[0].item() if isinstance(rewards, torch.Tensor) else rewards[0]
        )
        terminated = bool(
            terminateds[0].item()
            if isinstance(terminateds, torch.Tensor)
            else terminateds[0]
        )
        truncated = bool(
            truncateds[0].item() if isinstance(truncateds, torch.Tensor) else truncateds[0]
        )

        episode_return += reward_value
        episode_length += 1
        done = terminated or truncated
        prev_transition = {"next_observation": next_observations}

        frame = _extract_rgb_frame(eval_env.render())
        if frame is not None:
            video_frames.append(frame)

    eval_log = {
        "Eval/episode_return": episode_return,
        "Eval/episode_length": float(episode_length),
    }
    if video_frames:
        video = np.stack(video_frames, axis=0)
        video = np.expand_dims(video, axis=0).transpose(0, 1, 4, 2, 3)
        eval_log["Eval/video"] = video

    logger.update_metric(**eval_log)


def launch_off_policy(task_id: str, cfg: TrainConfig):

    cuda_visible = torch.cuda.is_available()
    if not cuda_visible:
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Log
    task_name = task_id.split("-")[-1].lower()
    save_path_base = str(Path("logs") / cfg.agent.algo.agent_type / task_name)
    os.makedirs(save_path_base, exist_ok=True)

    # Environment.
    env = ManagerBasedRlEnv(
        cfg=cfg.env,
        device=device,
        render_mode=None,
    )
    env = FlashVecEnvWrapper(
        env=env,
        clip_actions=cfg.agent.algo.env_clip_actions,
        normalize_joint_actions=cfg.agent.algo.env_normalize_joint_actions,
        joint_action_gain=cfg.agent.algo.env_joint_action_gain,
    )

    eval_env_cfg = copy.deepcopy(cfg.env)
    eval_env_cfg.scene.num_envs = 1
    eval_env = ManagerBasedRlEnv(
        cfg=eval_env_cfg,
        device=device,
        render_mode="rgb_array",
    )
    eval_env = FlashVecEnvWrapper(
        env=eval_env,
        clip_actions=cfg.agent.algo.env_clip_actions,
        normalize_joint_actions=cfg.agent.algo.env_normalize_joint_actions,
        joint_action_gain=cfg.agent.algo.env_joint_action_gain,
    )
    logger = WandbTrainerLogger(cfg=cfg, num_envs=env.num_envs, device=env.device)

    observation_space = env.observation_space
    action_space = env.action_space

    _, env_info = env.reset()

    # Agent.
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent.algo,
    )

    observations, env_infos = env.reset()

    actions: Optional[Tensor] = None
    transition: Optional[dict[str, Tensor]] = None
    update_counter = 0
    update_info = {}
    evaluation(
        agent=agent,
        eval_env=eval_env,
        interaction_step=0,
        logger=logger,
    )
    logger.log_metric(step=0)
    logger.reset()
    cfg = cfg.agent

    for interaction_step in tqdm.tqdm(
        range(1, int(cfg.num_interaction_steps + 1)), smoothing=0.1, mininterval=0.5
    ):
        collect_start_time = time.perf_counter()
        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(
                interaction_step, prev_transition=transition, training=True
            )
        else:
            actions = env.action_space.sample()

        next_observations, rewards, terminateds, truncateds, env_infos = env.step(
            actions
        )
        logger.process_env_step(
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            extras=env_infos,
        )
        next_buffer_observations = next_observations.clone()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        transition = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }

        agent.process_transition(transition)
        transition["next_observation"] = next_observations
        observations = next_observations
        collect_time = time.perf_counter() - collect_start_time

        if agent.can_start_training():
            learn_start_time = time.perf_counter()
            # update network
            # updates_per_interaction_step can be below 1.0
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                update_info = agent.update()
                logger.update_metric(**update_info)
                update_counter -= 1

            # metrics
            if (
                cfg.metrics_per_interaction_step
                and interaction_step % cfg.metrics_per_interaction_step == 0
            ):
                metrics_info = agent.get_metrics()
                logger.update_metric(**metrics_info)

            if (
                cfg.eval_per_interaction_step
                and interaction_step % cfg.eval_per_interaction_step == 0
            ):
                evaluation(
                    agent=agent,
                    eval_env=eval_env,
                    interaction_step=interaction_step,
                    logger=logger,
                )

            # logging
            if (
                cfg.logging_per_interaction_step
                and interaction_step % cfg.logging_per_interaction_step == 0
            ):
                learn_time = time.perf_counter() - learn_start_time
                logger.log(
                    step=interaction_step,
                    collect_time=collect_time,
                    learn_time=learn_time,
                )
                logger.reset()

            # checkpointing
            if (
                cfg.save_checkpoint_per_interaction_step
                and interaction_step % cfg.save_checkpoint_per_interaction_step == 0
            ):
                save_path = os.path.join(save_path_base, f"step{interaction_step}")
                agent.save(save_path)
