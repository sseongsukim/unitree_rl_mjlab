"""RL configuration for Unitree Go2 jump task."""

from src.rl_core.rsl_rl.rl.config import (
    RslRlMetraAlgorithmCfg,
    RslRlMetraRunnerCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_go2_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree Go2 jump task."""

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="go2_jump",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10001,
    )


def unitree_go2_jump_metra_runner_cfg() -> RslRlMetraRunnerCfg:
    """Create METRA+PPO runner configuration for Unitree Go2 jump task."""

    cfg = RslRlMetraRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        traj_encoder=RslRlModelCfg(
            hidden_dims=(256, 256, 256),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlMetraAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            dim_option=2,
            discrete_option=False,
            unit_length_option=True,
            metra_reward_coef=0.01,
            traj_encoder_learning_rate=1.0e-4,
            traj_encoder_num_epochs=1,
            traj_encoder_num_mini_batches=4,
            dual_reg=True,
            dual_lam=30.0,
            dual_slack=1.0e-3,
            dual_learning_rate=1.0e-4,
        ),
        experiment_name="go2_jump_metra",
        run_name="metra",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10001,
    )
    return cfg
