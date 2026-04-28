from mjlab.tasks.registry import register_mjlab_task
from src.rl_core.rsl_rl.rl.runner import ProjectOnPolicyRunner
from src.rl_core.rsl_rl.runners.metra_runner import METRARunner

from .env_cfgs import unitree_go2_jump_env_cfg
from .rl_cfg import unitree_go2_jump_metra_runner_cfg, unitree_go2_jump_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Jump",
  env_cfg=unitree_go2_jump_env_cfg(),
  play_env_cfg=unitree_go2_jump_env_cfg(play=True),
  rl_cfg=unitree_go2_jump_ppo_runner_cfg(),
  runner_cls=ProjectOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Jump-METRA",
  env_cfg=unitree_go2_jump_env_cfg(),
  play_env_cfg=unitree_go2_jump_env_cfg(play=True),
  rl_cfg=unitree_go2_jump_metra_runner_cfg(),
  runner_cls=METRARunner,
)
