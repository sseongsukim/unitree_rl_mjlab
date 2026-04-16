"""RL configuration for Unitree Go2 leap fine-tuning."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg

###
# RNN
###


def unitree_go2_recurrent_ppo_runner_cfg(
    *,
    rnn_type: str,
    experiment_name: str,
    hidden_dim: int = 256,
    num_layers: int = 1,
    learning_rate: float = 3.0e-4,
    entropy_coef: float = 0.015,
    desired_kl: float = 0.008,
) -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_go2_ppo_runner_cfg()

    cfg.actor.class_name = "RNNModel"
    cfg.actor.rnn_type = rnn_type
    cfg.actor.rnn_hidden_dim = hidden_dim
    cfg.actor.rnn_num_layers = num_layers

    cfg.critic.class_name = "RNNModel"
    cfg.critic.rnn_type = rnn_type
    cfg.critic.rnn_hidden_dim = hidden_dim
    cfg.critic.rnn_num_layers = num_layers

    cfg.algorithm.learning_rate = learning_rate
    cfg.algorithm.entropy_coef = entropy_coef
    cfg.algorithm.desired_kl = desired_kl
    cfg.experiment_name = experiment_name
    cfg.save_interval = 100
    return cfg


def unitree_go2_flat_pre_gru_ppo_runner_cfg():
    return unitree_go2_recurrent_ppo_runner_cfg(
        rnn_type="gru", experiment_name="go2_flat_pre_gru"
    )


def unitree_go2_flat_pre_lstm_ppo_runner_cfg():
    return unitree_go2_recurrent_ppo_runner_cfg(
        rnn_type="lstm",
        experiment_name="go2_flat_pre_lstm",
    )


def unitree_go2_leap_gru_ppo_runner_cfg():
    return unitree_go2_recurrent_ppo_runner_cfg(
        rnn_type="gru", experiment_name="go2_leap_gru"
    )


def unitree_go2_leap_lstm_ppo_runner_cfg():
    return unitree_go2_recurrent_ppo_runner_cfg(
        rnn_type="lstm",
        experiment_name="go2_leap_lstm",
    )


###
# MLP
###


def unitree_go2_leap_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for flat-to-leap fine-tuning."""

    cfg = unitree_go2_ppo_runner_cfg()
    cfg.algorithm.learning_rate = 3.0e-4
    cfg.algorithm.entropy_coef = 0.015
    cfg.algorithm.desired_kl = 0.008
    cfg.experiment_name = "go2_leap_finetune"
    cfg.save_interval = 100
    return cfg
