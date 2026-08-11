"""Threat-response task configuration.

Wraps the (already robot-customized) Go2 flat velocity config without
modifying it in place: the base walking task stays untouched and trainable
on its own, and this module only *adds* a threat-aware twist command, one
backstop reward, one termination, and one curriculum term on top of it.

See the design doc worked out in conversation for the reasoning behind each
of these choices (command override instead of new reward terms, virtual
tracked threat instead of a MuJoCo body, RLBase-internal trigger instead of a
new FSM state, etc).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.threat_response import mdp as threat_mdp
from src.tasks.threat_response.mdp.threat_command import make_threat_aware_command_cfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg

COMMAND_NAME = "twist"

# Threat exposure curriculum: ramp up gradually so the policy first solidifies
# ordinary walking/heading behavior before threats start appearing often.
DEFAULT_THREAT_STAGES: list[threat_mdp.ThreatStage] = [
  {"step": 0, "threat_prob": 0.0},
  {"step": 2000 * 24, "threat_prob": 0.15},
  {"step": 5000 * 24, "threat_prob": 0.30},
]


def make_threat_response_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Go2 flat threat-response configuration."""
  cfg = unitree_go2_flat_env_cfg(play=play)

  base_twist = cfg.commands[COMMAND_NAME]
  assert isinstance(base_twist, UniformVelocityCommandCfg)
  threat_twist = make_threat_aware_command_cfg(base_twist)
  if play:
    # No ramp-up in play mode -- just show the behavior.
    threat_twist.threat_prob = 1.0
  cfg.commands[COMMAND_NAME] = threat_twist

  threat_obs_terms = {
    "threat_bearing": ObservationTermCfg(
      func=threat_mdp.threat_field,
      params={"command_name": COMMAND_NAME, "field": "threat_bearing"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=1.0 / math.pi,
    ),
    "threat_range": ObservationTermCfg(
      func=threat_mdp.threat_field,
      params={"command_name": COMMAND_NAME, "field": "threat_range"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1.0 / threat_twist.max_range,
    ),
    "threat_level": ObservationTermCfg(
      func=threat_mdp.threat_field,
      params={"command_name": COMMAND_NAME, "field": "threat_level"},
    ),
    "threat_active": ObservationTermCfg(
      func=threat_mdp.threat_field,
      params={"command_name": COMMAND_NAME, "field": "threat_active"},
    ),
  }
  cfg.observations["actor"].terms.update(threat_obs_terms)

  critic_threat_obs_terms = {
    name: ObservationTermCfg(func=term.func, params=term.params, scale=term.scale)
    for name, term in threat_obs_terms.items()
  }
  cfg.observations["critic"].terms.update(critic_threat_obs_terms)

  cfg.rewards["threat_distance_progress"] = RewardTermCfg(
    func=threat_mdp.threat_distance_progress,
    weight=0.5,
    params={"command_name": COMMAND_NAME},
  )

  cfg.terminations["threat_caught"] = TerminationTermCfg(
    func=threat_mdp.threat_caught,
    params={"command_name": COMMAND_NAME},
  )

  if not play:
    cfg.curriculum["threat_exposure"] = CurriculumTermCfg(
      func=threat_mdp.threat_exposure,
      params={"command_name": COMMAND_NAME, "stages": DEFAULT_THREAT_STAGES},
    )

  return cfg
