import torch
import numpy as np


import gymnasium as gym
from gymnasium.vector.utils import batch_space

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class FlashVecEnvWrapper(object):

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        clip_actions: float | None = None,
        normalize_joint_actions: bool = False,
        joint_action_gain: float | None = None,
    ):
        self.env = env
        self.clip_actions = clip_actions
        self.normalize_joint_actions = normalize_joint_actions
        self.joint_action_gain = joint_action_gain
        self.num_envs = self.env.num_envs
        self.device = torch.device(self.env.device)
        self.max_episode_length = self.env.max_episode_length
        self.num_actions = self.env.action_manager.total_action_dim
        self._joint_action_min: torch.Tensor | None = None
        self._joint_action_max: torch.Tensor | None = None

        if self.normalize_joint_actions and self.joint_action_gain is not None:
            raise ValueError(
                "joint_action_gain must be None when normalize_joint_actions is True."
            )
        if not self.normalize_joint_actions and self.joint_action_gain is None:
            raise ValueError(
                "joint_action_gain must be set when normalize_joint_actions is False."
            )

        if self.normalize_joint_actions:
            self._setup_joint_action_normalization()

        self._modify_action_space()

        self.obs_size = self.env.single_observation_space.spaces["actor"].shape[0]
        self.asymmetric_obs = True
        self.critic_obs_size = self.env.single_observation_space.spaces["critic"].shape[
            0
        ]

        self.single_observation_space = gym.spaces.Box(
            low=0.0,
            high=0.0,
            shape=(self.obs_size + self.critic_obs_size,),
            dtype=np.float32,
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

        self.clip_actions = clip_actions
        self.action_size = self.env.single_action_space.shape
        if self.normalize_joint_actions:
            self.single_action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.action_size,
                dtype=np.float32,
            )
        else:
            self.single_action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.action_size,
                dtype=np.float32,
            )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(self):
        obs_dict, extras = self.env.reset()
        obs = obs_dict["actor"]

        critic_obs = obs_dict["critic"]
        obs = torch.cat((obs, critic_obs), dim=-1)

        extras.update(
            {
                "actor_observation_size": self.obs_size,
                "asymmetric_obs": self.asymmetric_obs,
            }
        )

        return obs, extras

    def step(self, actions: torch.Tensor):
        if isinstance(actions, torch.Tensor):
            actions = actions.to(self.device)
        else:
            actions = torch.from_numpy(actions).to(self.device)

        if self.normalize_joint_actions:
            actions = self.unnormalize_actions(actions)
        else:
            actions = torch.clamp(actions, -1.0, 1.0) * self.joint_action_gain
        obs_dict, rew, terminations, truncations, extras = self.env.step(actions)
        obs = obs_dict["actor"]
        critic_obs = obs_dict["critic"]
        obs = torch.cat((obs, critic_obs), dim=-1)

        infos = {"time_outs": truncations, "observations": {"critic": critic_obs}}
        infos["final_obs"] = obs

        extras.update(infos)
        return obs, rew, terminations, truncations, extras

    def render(self):
        return self.env.render()

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.normalize_joint_actions:
            return torch.clamp(
                actions.to(self.device) / self.joint_action_gain, -1.0, 1.0
            )
        assert self._joint_action_min is not None
        assert self._joint_action_max is not None
        actions = actions.to(self.device)
        pos_scale = torch.clamp(self._joint_action_max, min=1e-6)
        neg_scale = torch.clamp(-self._joint_action_min, min=1e-6)
        normalized = torch.where(
            actions >= 0.0, actions / pos_scale, actions / neg_scale
        )
        return torch.clamp(normalized, -1.0, 1.0)

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.normalize_joint_actions:
            return actions.to(self.device) * self.joint_action_gain
        assert self._joint_action_min is not None
        assert self._joint_action_max is not None
        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        pos_scale = self._joint_action_max
        neg_scale = -self._joint_action_min
        return torch.where(actions >= 0.0, actions * pos_scale, actions * neg_scale)

    def _modify_action_space(self) -> None:
        if self.normalize_joint_actions:
            from mjlab.utils.spaces import Box, batch_space

            self.env.single_action_space = Box(
                shape=(self.num_actions,),
                low=-1.0,
                high=1.0,
            )
            self.env.action_space = batch_space(
                self.env.single_action_space, self.num_envs
            )
            return

        from mjlab.utils.spaces import Box, batch_space

        self.env.single_action_space = Box(
            shape=(self.num_actions,), low=-1.0, high=1.0
        )
        self.env.action_space = batch_space(self.env.single_action_space, self.num_envs)

    def _setup_joint_action_normalization(self) -> None:
        robot = self.env.scene["robot"]
        joint_action = self.env.action_manager.get_term("joint_pos")

        action_scale = getattr(joint_action, "_scale", None)
        if action_scale is None:
            raise ValueError("Joint action normalization requires joint_pos action scale.")
        if not isinstance(action_scale, torch.Tensor):
            action_scale = torch.as_tensor(action_scale, device=self.device)
        else:
            action_scale = action_scale.to(self.device)
        if action_scale.ndim > 1:
            action_scale = action_scale[0]

        joint_ids = getattr(joint_action, "_joint_ids", None)
        default_joint_pos = robot.data.default_joint_pos[0].to(self.device)
        soft_joint_pos_limits = robot.data.soft_joint_pos_limits[0].to(self.device)
        if joint_ids is not None:
            joint_ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
            default_joint_pos = default_joint_pos[joint_ids]
            soft_joint_pos_limits = soft_joint_pos_limits[joint_ids]

        self._joint_action_min = (
            soft_joint_pos_limits[:, 0] - default_joint_pos
        ) / action_scale
        self._joint_action_max = (
            soft_joint_pos_limits[:, 1] - default_joint_pos
        ) / action_scale
