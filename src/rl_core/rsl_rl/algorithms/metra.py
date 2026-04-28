from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from src.rl_core.rsl_rl.algorithms.ppo import PPO
from src.rl_core.rsl_rl.env import VecEnv
from src.rl_core.rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from src.rl_core.rsl_rl.models import MLPModel
from src.rl_core.rsl_rl.modules import EmpiricalNormalization, MLP
from src.rl_core.rsl_rl.storage import RolloutStorage
from src.rl_core.rsl_rl.utils import (
    resolve_callable,
    resolve_obs_groups,
)


class TrajectoryEncoder(nn.Module):
    """MLP trajectory encoder used by METRA."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(input_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))

    def update_normalization(self, obs: torch.Tensor) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(obs)  # type: ignore[attr-defined]


class METRAPPO(PPO):
    """PPO with a METRA trajectory encoder and fixed intrinsic reward scale."""

    traj_encoder: TrajectoryEncoder

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        traj_encoder: TrajectoryEncoder,
        storage: RolloutStorage,
        dim_option: int = 2,
        discrete_option: bool = False,
        unit_length_option: bool = True,
        metra_reward_coef: float = 0.01,
        metra_reward_coef_schedule: dict[str, Any] | None = None,
        traj_encoder_learning_rate: float = 1e-4,
        traj_encoder_num_epochs: int = 1,
        traj_encoder_num_mini_batches: int = 4,
        dual_reg: bool = True,
        dual_lam: float = 30.0,
        dual_slack: float = 1e-3,
        dual_learning_rate: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)

        self.traj_encoder = traj_encoder.to(self.device)
        self.dim_option = dim_option
        self.discrete_option = discrete_option
        self.unit_length_option = unit_length_option
        self.initial_metra_reward_coef = metra_reward_coef
        self.metra_reward_coef = metra_reward_coef
        self.metra_reward_coef_scheduler_params = metra_reward_coef_schedule
        self.metra_reward_coef_scheduler = (
            getattr(
                self,
                f"_{metra_reward_coef_schedule['mode']}_metra_reward_coef_schedule",
            )
            if metra_reward_coef_schedule is not None
            else None
        )
        self.traj_encoder_num_epochs = traj_encoder_num_epochs
        self.traj_encoder_num_mini_batches = traj_encoder_num_mini_batches
        self.dual_reg = dual_reg
        self.dual_slack = dual_slack

        self.log_dual_lam = nn.Parameter(
            torch.tensor([dual_lam], device=self.device).log()
        )
        self.traj_encoder_optimizer = optim.Adam(
            self.traj_encoder.parameters(), lr=traj_encoder_learning_rate
        )
        self.dual_optimizer = optim.Adam([self.log_dual_lam], lr=dual_learning_rate)

        self._metra_step = 0
        self._metra_obs = torch.zeros(
            storage.num_transitions_per_env,
            storage.num_envs,
            self._metra_state_dim,
            device=self.device,
        )
        self._metra_next_obs = torch.zeros_like(self._metra_obs)
        self._metra_skills = torch.zeros(
            storage.num_transitions_per_env,
            storage.num_envs,
            self.dim_option,
            device=self.device,
        )
        self._metra_task_rewards = torch.zeros_like(storage.rewards)
        self._metra_valid = torch.zeros_like(storage.rewards, dtype=torch.bool)
        self._last_metra_loss_dict: dict[str, float] = {}
        self.intrinsic_rewards: torch.Tensor | None = None
        self.metra_reward_coef_step = 0

    @property
    def _metra_state_dim(self) -> int:
        actor_obs_dim = self.actor.obs_dim
        if "skill" in self.actor.obs_groups:
            return actor_obs_dim - self.dim_option
        return actor_obs_dim - self.dim_option

    def _extract_metra_state(self, obs: TensorDict) -> torch.Tensor:
        state = torch.cat([obs[group] for group in self.actor.obs_groups], dim=-1)
        return state[..., : -self.dim_option]

    def _extract_skill(self, obs: TensorDict) -> torch.Tensor:
        state = torch.cat([obs[group] for group in self.actor.obs_groups], dim=-1)
        return state[..., -self.dim_option :]

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        step = self.storage.step
        if "skill" not in obs:
            raise KeyError(
                "METRAPPO requires RslRlVecEnvWrapper(use_skill=True), but obs has no 'skill' entry."
            )

        self._metra_obs[step].copy_(
            self._extract_metra_state(self.transition.observations).detach()
        )
        next_obs = extras.get("final_obs", obs)
        next_state = self._extract_metra_state(next_obs).detach()
        done_mask = dones.to(dtype=torch.bool)
        if done_mask.any():
            next_state = next_state.clone()
            next_state[done_mask] = self._metra_obs[step][done_mask]
        self._metra_next_obs[step].copy_(next_state)
        self._metra_valid[step].copy_((~done_mask).view(-1, 1))
        self._metra_skills[step].copy_(
            self._extract_skill(self.transition.observations).detach()
        )
        self._metra_task_rewards[step].copy_(rewards.view(-1, 1).detach())
        self._metra_step = step + 1
        self.metra_reward_coef_step += 1

        super().process_env_step(obs, rewards, dones, extras)

    def update_traj_encoder(self) -> dict[str, float]:
        if self._metra_step == 0:
            return {}

        obs = self._metra_obs[: self._metra_step].flatten(0, 1)
        next_obs = self._metra_next_obs[: self._metra_step].flatten(0, 1)
        skills = self._metra_skills[: self._metra_step].flatten(0, 1)
        valid = self._metra_valid[: self._metra_step].flatten(0, 1).squeeze(-1)
        obs = obs[valid]
        next_obs = next_obs[valid]
        skills = skills[valid]

        batch_size = obs.shape[0]
        if batch_size == 0:
            self._last_metra_loss_dict = {"metra/valid_transition_frac": 0.0}
            return self._last_metra_loss_dict
        num_mini_batches = min(self.traj_encoder_num_mini_batches, batch_size)
        mini_batch_size = max(1, batch_size // num_mini_batches)

        loss_te_sum = 0.0
        reward_sum = 0.0
        penalty_sum = 0.0
        dual_loss_sum = 0.0
        num_updates = 0

        self.traj_encoder.train()
        self.traj_encoder.update_normalization(torch.cat((obs, next_obs), dim=0))

        for _ in range(self.traj_encoder_num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                idx = indices[start : start + mini_batch_size]
                cur_obs = obs[idx]
                cur_next_obs = next_obs[idx]
                cur_skills = skills[idx]

                cur_z = self.traj_encoder(cur_obs)
                next_z = self.traj_encoder(cur_next_obs)
                delta_z = next_z - cur_z
                rewards = (delta_z * cur_skills).sum(dim=-1)

                if self.dual_reg:
                    cst_dist = torch.ones_like(rewards)
                    cst_penalty = cst_dist - torch.square(delta_z).mean(dim=-1)
                    cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
                    dual_lam = self.log_dual_lam.exp()
                    te_obj = rewards + dual_lam.detach() * cst_penalty
                else:
                    cst_penalty = torch.zeros_like(rewards)
                    te_obj = rewards

                loss_te = -te_obj.mean()
                self.traj_encoder_optimizer.zero_grad()
                loss_te.backward()
                self.traj_encoder_optimizer.step()

                if self.dual_reg:
                    loss_dual = self.log_dual_lam * cst_penalty.detach().mean()
                    self.dual_optimizer.zero_grad()
                    loss_dual.backward()
                    self.dual_optimizer.step()
                    dual_loss_sum += loss_dual.item()

                loss_te_sum += loss_te.item()
                reward_sum += rewards.mean().item()
                penalty_sum += cst_penalty.mean().item()
                num_updates += 1

        loss_dict = {
            "metra/loss_te": loss_te_sum / max(1, num_updates),
            "metra/reward": reward_sum / max(1, num_updates),
            "metra/dual_penalty": penalty_sum / max(1, num_updates),
            "metra/dual_lam": self.log_dual_lam.exp().item(),
            "metra/valid_transition_frac": valid.float().mean().item(),
        }
        if self.dual_reg:
            loss_dict["metra/loss_dual"] = dual_loss_sum / max(1, num_updates)
        self._last_metra_loss_dict = loss_dict
        return loss_dict

    def rebuild_rewards(self) -> dict[str, float]:
        if self._metra_step == 0:
            return {}

        with torch.no_grad():
            obs = self._metra_obs[: self._metra_step]
            next_obs = self._metra_next_obs[: self._metra_step]
            skills = self._metra_skills[: self._metra_step]
            delta_z = self.traj_encoder(next_obs.flatten(0, 1)) - self.traj_encoder(
                obs.flatten(0, 1)
            )
            metra_rewards = (delta_z * skills.flatten(0, 1)).sum(dim=-1).view(
                self._metra_step, self.storage.num_envs, 1
            )
            metra_rewards = metra_rewards * self._metra_valid[: self._metra_step]
            task_rewards = self._metra_task_rewards[: self._metra_step]
            if self.metra_reward_coef_scheduler is not None:
                self.metra_reward_coef = self.metra_reward_coef_scheduler(
                    step=self.metra_reward_coef_step,
                    **self.metra_reward_coef_scheduler_params,
                )
            else:
                self.metra_reward_coef = self.initial_metra_reward_coef
            total_rewards = task_rewards + self.metra_reward_coef * metra_rewards
            self.storage.rewards[: self._metra_step].copy_(total_rewards)
            self.intrinsic_rewards = self.metra_reward_coef * metra_rewards[-1]

        loss_dict = {
            "metra/task_reward_mean": task_rewards.mean().item(),
            "metra/intrinsic_reward_mean": metra_rewards.mean().item(),
            "metra/reward_coef": self.metra_reward_coef,
            "metra/total_reward_mean": total_rewards.mean().item(),
        }
        self._last_metra_loss_dict.update(loss_dict)
        return loss_dict

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict.update(self._last_metra_loss_dict)
        self._metra_step = 0
        return loss_dict

    def _constant_metra_reward_coef_schedule(
        self, step: int, **kwargs: dict[str, Any]
    ) -> float:
        """Keep the METRA reward coefficient constant."""
        return self.initial_metra_reward_coef

    def _step_metra_reward_coef_schedule(
        self, step: int, final_step: int, final_value: float, **kwargs: dict[str, Any]
    ) -> float:
        """Switch the METRA reward coefficient at a configured step."""
        return self.initial_metra_reward_coef if step < final_step else final_value

    def _linear_metra_reward_coef_schedule(
        self,
        step: int,
        initial_step: int,
        final_step: int,
        final_value: float,
        **kwargs: dict[str, Any],
    ) -> float:
        """Linearly interpolate the METRA reward coefficient over a step interval."""
        if final_step <= initial_step:
            return final_value
        if step < initial_step:
            return self.initial_metra_reward_coef
        if step > final_step:
            return final_value
        return self.initial_metra_reward_coef + (
            final_value - self.initial_metra_reward_coef
        ) * (step - initial_step) / (final_step - initial_step)

    def train_mode(self) -> None:
        super().train_mode()
        self.traj_encoder.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.traj_encoder.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["traj_encoder_state_dict"] = self.traj_encoder.state_dict()
        saved_dict["traj_encoder_optimizer_state_dict"] = (
            self.traj_encoder_optimizer.state_dict()
        )
        saved_dict["log_dual_lam"] = self.log_dual_lam.detach()
        saved_dict["dual_optimizer_state_dict"] = self.dual_optimizer.state_dict()
        saved_dict["metra_reward_coef_step"] = self.metra_reward_coef_step
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("traj_encoder", True):
            self.traj_encoder.load_state_dict(
                loaded_dict["traj_encoder_state_dict"], strict=strict
            )
        if load_cfg is None or load_cfg.get("optimizer", True):
            self.traj_encoder_optimizer.load_state_dict(
                loaded_dict["traj_encoder_optimizer_state_dict"]
            )
            self.log_dual_lam.data.copy_(loaded_dict["log_dual_lam"].to(self.device))
            self.dual_optimizer.load_state_dict(loaded_dict["dual_optimizer_state_dict"])
            self.metra_reward_coef_step = loaded_dict.get("metra_reward_coef_step", 0)
        return load_iteration

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        model_params = [self.traj_encoder.state_dict(), self.log_dual_lam.detach()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.traj_encoder.load_state_dict(model_params[0])
        self.log_dual_lam.data.copy_(model_params[1].to(self.device))

    def reduce_parameters(self) -> None:
        super().reduce_parameters()

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "METRAPPO":
        alg_class: type[METRAPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(
            cfg["algorithm"], obs, cfg["obs_groups"], env
        )
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")

        dim_option = cfg["algorithm"]["dim_option"]
        traj_encoder_input_dim = actor.obs_dim - dim_option
        traj_encoder_cfg = cfg["traj_encoder"]
        traj_encoder = TrajectoryEncoder(
            input_dim=traj_encoder_input_dim,
            output_dim=dim_option,
            hidden_dims=traj_encoder_cfg["hidden_dims"],
            activation=traj_encoder_cfg["activation"],
            obs_normalization=traj_encoder_cfg["obs_normalization"],
        ).to(device)
        print(f"Trajectory Encoder: {traj_encoder}")

        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg = alg_class(
            actor,
            critic,
            traj_encoder,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg
