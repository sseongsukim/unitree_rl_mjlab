"""Flash-rl configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlashRlRunnerCfg:
    seed: int = 42

    num_env_steps: int = 1_000_000

    num_train_envs: int = 4096

    updates_per_interaction_step: int = 1

    n_step: int = 1

    gamma: float = 0.99

    @property
    def num_interaction_steps(self) -> int:
        return self.num_env_steps // self.num_train_envs

    @property
    def num_update_steps(self) -> int:
        return self.num_interaction_steps * self.updates_per_interaction_step

    @property
    def evaluation_per_interaction_step(self) -> int:
        return max(1, self.num_interaction_steps // 10)

    @property
    def metrics_per_interaction_step(self) -> int:
        return max(1, self.num_interaction_steps // 10)

    @property
    def recording_per_interaction_step(self) -> int:
        return max(1, self.num_interaction_steps)

    @property
    def logging_per_interaction_step(self) -> int:
        return max(1, self.num_interaction_steps // 100)

    @property
    def save_checkpoint_per_interaction_step(self) -> int:
        return max(1, self.num_interaction_steps)


@dataclass
class FlashRlAlgorithmCfg:
    updates_per_interaction_step: int = 1
    num_interaction_steps: int = 1
    gamma: float = 0.99
    n_step: int = 1

    def sync_with_runner_cfg(self, runner_cfg: FlashRlRunnerCfg) -> None:
        self.updates_per_interaction_step = runner_cfg.updates_per_interaction_step
        self.num_interaction_steps = runner_cfg.num_interaction_steps
        self.gamma = runner_cfg.gamma
        self.n_step = runner_cfg.n_step

    @property
    def num_update_steps(self) -> int:
        return self.num_interaction_steps * self.updates_per_interaction_step


@dataclass
class FlashSACAlgorithmCfg(FlashRlAlgorithmCfg):
    agent_type: str = "flashSAC"
    device_type: str = "cuda"

    buffer_max_length: int = 1_000_000
    buffer_min_length: int = 10_000
    buffer_device_type: str = "cuda"
    sample_batch_size: int = 2048

    normalize_reward: bool = True
    normalized_G_max: float = 5.0

    asymmetric_observation: bool = False

    learning_rate_init: float = 3e-4
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_rate: float = 1e-6
    learning_rate_decay_rate: float = 1.0

    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    actor_update_period: int = 2

    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_bins: int = 101
    critic_target_update_tau: float = 0.01

    temp_initial_value: float = 0.01
    temp_target_sigma: float = 0.15
    temp_target_entropy: float | None = None

    use_compile: bool = True
    compile_mode: str = "auto"
    use_amp: bool = True

    load_optimizer: bool = True
    load_reward_normalizer: bool = True

    @property
    def learning_rate_warmup_step(self) -> int:
        return int(
            self.learning_rate_warmup_rate
            * self.num_interaction_steps
            * self.updates_per_interaction_step
        )

    @property
    def learning_rate_decay_step(self) -> int:
        return int(
            self.learning_rate_decay_rate
            * self.num_interaction_steps
            * self.updates_per_interaction_step
        )

    @property
    def critic_min_v(self) -> float:
        return -self.normalized_G_max

    @property
    def critic_max_v(self) -> float:
        return self.normalized_G_max


@dataclass
class FlashRlCfg(FlashRlRunnerCfg):
    project_name: str = "mjlab"
    entity_name: str | None = None
    group_name: str = "default"
    exp_name: str = "default"
    algo: FlashSACAlgorithmCfg = field(default_factory=FlashSACAlgorithmCfg)

    def __post_init__(self) -> None:
        self.sync_agent_cfg()

    def sync_agent_cfg(self) -> None:
        self.algo.sync_with_runner_cfg(self)
