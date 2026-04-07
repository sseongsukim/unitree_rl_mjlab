from mjlab.tasks.registry import register_mjlab_task
from src.rl_core.rsl_rl.rl.runner import ProjectOnPolicyRunner
from src.tasks.leap.config.go2.rl_cfg import unitree_go2_leap_ppo_runner_cfg

from .env_cfgs import unitree_go2_leap_env_cfg

register_mjlab_task(
    task_id="Unitree-Go2-Leap",
    env_cfg=unitree_go2_leap_env_cfg(),
    play_env_cfg=unitree_go2_leap_env_cfg(play=True),
    rl_cfg=unitree_go2_leap_ppo_runner_cfg(),
    runner_cls=ProjectOnPolicyRunner,
)
