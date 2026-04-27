import torch
from src.rl_core.rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


class RslRlVecEnvWrapper(VecEnv):
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        clip_actions: float | None = None,
        use_skill: bool | None = None,
        discrete_skill: bool | None = None,
        num_dims: int | None = None,
    ):
        self.env = env
        self.clip_actions = clip_actions
        self.use_skill = bool(use_skill)
        self.discrete_skill = bool(discrete_skill)
        self.num_dims = num_dims

        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        if self.use_skill and (self.num_dims is None or self.num_dims <= 0):
            raise ValueError("num_dims must be a positive integer when use_skill=True.")
        self.current_skills = (
            self._sample_skills(self.num_envs) if self.use_skill else None
        )
        self._modify_action_space()
        self._modify_observation_space()

        # Reset at the start since rsl_rl does not call reset.
        self.env.reset()

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> Space:
        return self.env.observation_space

    @property
    def action_space(self) -> Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self.env.unwrapped

    # Properties.

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(
        self, value: torch.Tensor
    ) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        obs_dict = self._add_skills_to_obs(obs_dict)
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        if self.use_skill:
            self.current_skills = self._sample_skills(self.num_envs)
            extras["skills"] = self.current_skills.clone()
        obs_dict = self._add_skills_to_obs(obs_dict)
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        term_or_trunc = terminated | truncated
        assert isinstance(rew, torch.Tensor)
        assert isinstance(term_or_trunc, torch.Tensor)
        dones = term_or_trunc.to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        if self.use_skill:
            self._resample_done_skills(term_or_trunc)
            extras["skills"] = self.current_skills.clone()
        obs_dict = self._add_skills_to_obs(obs_dict)
        return (
            TensorDict(obs_dict, batch_size=[self.num_envs]),
            rew,
            dones,
            extras,
        )

    def close(self) -> None:
        return self.env.close()

    def render(self):
        return self.env.render()

    # Private methods.

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

    def _modify_observation_space(self) -> None:
        if not self.use_skill:
            return

        from mjlab.utils.spaces import Box, batch_space

        single_space = self.unwrapped.single_observation_space
        if not hasattr(single_space, "spaces"):
            return

        assert self.num_dims is not None
        for key in ("actor", "critic"):
            space = single_space.spaces.get(key)
            if not isinstance(space, Box) or len(space.shape) == 0:
                continue
            single_space.spaces[key] = Box(
                shape=(*space.shape[:-1], space.shape[-1] + self.num_dims),
                low=space.low,
                high=space.high,
                dtype=space.dtype,
            )

        single_space.spaces["skill"] = Box(
            shape=(self.num_dims,), low=-1.0, high=1.0
        )
        self.unwrapped.observation_space = batch_space(single_space, self.num_envs)

    def _sample_skills(self, num_skills: int) -> torch.Tensor:
        assert self.num_dims is not None
        if self.discrete_skill:
            skill_ids = torch.randint(
                low=0, high=self.num_dims, size=(num_skills,), device=self.device
            )
            skills = torch.zeros(num_skills, self.num_dims, device=self.device)
            skills[torch.arange(num_skills, device=self.device), skill_ids] = 1.0
            return skills

        skills = torch.randn(num_skills, self.num_dims, device=self.device)
        return skills / skills.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _resample_done_skills(self, dones: torch.Tensor) -> None:
        assert self.current_skills is not None
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self.current_skills[done_ids] = self._sample_skills(done_ids.numel())

    def _add_skills_to_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.use_skill:
            return obs_dict

        assert self.current_skills is not None
        obs_dict = dict(obs_dict)
        skills = self.current_skills.to(device=self.device)
        for key in ("actor", "critic"):
            if key in obs_dict:
                obs_dict[key] = torch.cat((obs_dict[key], skills), dim=-1)
        obs_dict["skill"] = skills
        return obs_dict
