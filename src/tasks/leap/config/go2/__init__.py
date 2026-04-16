from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go2_flat_pre_env_cfg, unitree_go2_leap_env_cfg
from .rl_cfg import (
    unitree_go2_flat_pre_gru_ppo_runner_cfg,
    unitree_go2_flat_pre_lstm_ppo_runner_cfg,
    unitree_go2_leap_gru_ppo_runner_cfg,
    unitree_go2_leap_lstm_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Unitree-Go2-Flat-Pre",
    env_cfg=unitree_go2_flat_pre_env_cfg(),
    play_env_cfg=unitree_go2_flat_pre_env_cfg(play=True),
    rl_cfg=unitree_go2_flat_pre_lstm_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Go2-Flat-Pre-Lstm",
    env_cfg=unitree_go2_flat_pre_env_cfg(),
    play_env_cfg=unitree_go2_flat_pre_env_cfg(play=True),
    rl_cfg=unitree_go2_flat_pre_lstm_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Go2-Flat-Pre-Gru",
    env_cfg=unitree_go2_flat_pre_env_cfg(),
    play_env_cfg=unitree_go2_flat_pre_env_cfg(play=True),
    rl_cfg=unitree_go2_flat_pre_gru_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Go2-Leap",
    env_cfg=unitree_go2_leap_env_cfg(),
    play_env_cfg=unitree_go2_leap_env_cfg(play=True),
    rl_cfg=unitree_go2_leap_lstm_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Go2-Leap-Lstm",
    env_cfg=unitree_go2_leap_env_cfg(),
    play_env_cfg=unitree_go2_leap_env_cfg(play=True),
    rl_cfg=unitree_go2_leap_lstm_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-Go2-Leap-Gru",
    env_cfg=unitree_go2_leap_env_cfg(),
    play_env_cfg=unitree_go2_leap_env_cfg(play=True),
    rl_cfg=unitree_go2_leap_gru_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
