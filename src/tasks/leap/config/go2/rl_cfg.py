"""RL configuration for Unitree Go2 leap fine-tuning."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg


# def unitree_go2_leap_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
#     """Create RL runner configuration for flat-to-leap fine-tuning."""

#     cfg = unitree_go2_ppo_runner_cfg()
#     cfg.actor.class_name = "RNNModel"
#     cfg.actor.rnn_type = "lstm"
#     cfg.actor.rnn_hidden_dim = 256
#     cfg.actor.rnn_num_layers = 1
#     cfg.critic.class_name = "RNNModel"
#     cfg.critic.rnn_type = "lstm"
#     cfg.critic.rnn_hidden_dim = 256
#     cfg.critic.rnn_num_layers = 1
#     cfg.algorithm.learning_rate = 3.0e-4
#     cfg.algorithm.entropy_coef = 0.015
#     cfg.algorithm.desired_kl = 0.008
#     cfg.experiment_name = "go2_leap_finetune"
#     cfg.save_interval = 100
#     return cfg


def unitree_go2_leap_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for flat-to-leap fine-tuning."""

    cfg = unitree_go2_ppo_runner_cfg()
    cfg.algorithm.learning_rate = 3.0e-4
    cfg.algorithm.entropy_coef = 0.015
    cfg.algorithm.desired_kl = 0.008
    cfg.experiment_name = "go2_leap_finetune"
    cfg.save_interval = 100
    return cfg
