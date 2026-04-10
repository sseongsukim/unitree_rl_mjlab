import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class FlashVecEnvWrapper(object):

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        clip_actions: float | None = None,
    ):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim

        self._modify_action_space()

        self.obs_size = self.env.single_observation_space.shape
        self.asymmetric_obs = True
        self.critic_obs_size = self.env.single_observation_space["critic"].shape

    def reset(self):
        obs_dict, extras = self.env.reset()

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return

        from mjlab.utils.spaces import Box, batch_space

        self.unwrapped.single_action_space = Box(
            shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
        )
        self.unwrapped.action_space = batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )
