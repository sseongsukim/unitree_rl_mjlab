from dataclasses import replace

from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_envs", 4096, "Number of environments.")
flags.DEFINE_integer("seed", 125, "Seed number.")
flags.DEFINE_integer(
    "max_iterations", None, "Override the number of learning iterations."
)
flags.DEFINE_integer(
    "num_steps_per_env", None, "Override rollout steps per environment."
)
flags.DEFINE_string("agent_name", "ppo", "Agent name.")
flags.DEFINE_string("env_name", "jump_metra", "Task environment name.")
flags.DEFINE_string("logger", None, "Override logger: wandb or tensorboard.")
flags.DEFINE_boolean("use_rnd", False, "Enable PPO RND exploration.")
flags.DEFINE_boolean(
    "use_height_map", True, "Enable jump height-map observations for PPO."
)
flags.DEFINE_boolean(
    "symmetric_obs",
    True,
    "Use critic observations for the actor, making actor and critic observation layouts identical.",
)
flags.DEFINE_boolean("video", True, "Record videos during PPO training and evaluation.")
flags.DEFINE_integer("video_length", None, "Override PPO video recording length.")
flags.DEFINE_integer("video_interval", None, "Override PPO video recording interval.")
flags.DEFINE_float("metra_reward_coef", 2.0, "Override METRA reward coefficient.")
flags.DEFINE_float(
    "metra_reward_final_coef",
    0.2,
    "Enable linear METRA reward coefficient decay to this final value.",
)
flags.DEFINE_integer(
    "metra_reward_final_step",
    None,
    "Override final step for linear METRA reward coefficient decay.",
)
flags.DEFINE_boolean("upload_model", None, "Upload model checkpoints to W&B.")


register_env_dict = {
    "empty": "Unitree-Go2-Empty",
    "jump": "Unitree-Go2-Jump",
    "jump_metra": "Unitree-Go2-Jump-METRA",
    "leap": "Unitree-Go2-Leap",
    "flat": "Unitree-Go2-Flat",
}


def main(_):
    env_name = register_env_dict[FLAGS.env_name]
    project_name = f"{env_name}_{FLAGS.agent_name}"

    if FLAGS.agent_name == "ppo":
        from train_ppo import launch_training, TrainConfig

        args = TrainConfig.from_task(task_id=env_name)
        args.env.scene.num_envs = FLAGS.num_envs

        args.agent.wandb_project = project_name
        args.agent.seed = FLAGS.seed
        if FLAGS.max_iterations is not None:
            args.agent.max_iterations = FLAGS.max_iterations
        if FLAGS.num_steps_per_env is not None:
            args.agent.num_steps_per_env = FLAGS.num_steps_per_env
        if FLAGS.logger is not None:
            args.agent.logger = FLAGS.logger
        if FLAGS.upload_model is not None:
            args.agent.upload_model = FLAGS.upload_model
        algorithm = args.agent.algorithm
        if FLAGS.metra_reward_coef is not None:
            if not hasattr(algorithm, "metra_reward_coef"):
                raise ValueError("--metra_reward_coef is only supported for METRA PPO.")
            algorithm.metra_reward_coef = FLAGS.metra_reward_coef
        if FLAGS.metra_reward_final_coef is not None:
            if not hasattr(algorithm, "metra_reward_coef_schedule"):
                raise ValueError(
                    "--metra_reward_final_coef is only supported for METRA PPO."
                )
            if FLAGS.metra_reward_final_coef > algorithm.metra_reward_coef:
                raise ValueError(
                    "--metra_reward_final_coef must be less than or equal to "
                    "--metra_reward_coef for linear decay."
                )
            final_step = FLAGS.metra_reward_final_step
            if final_step is None:
                final_step = args.agent.max_iterations * args.agent.num_steps_per_env
            algorithm.metra_reward_coef_schedule = {
                "mode": "linear",
                "initial_step": 0,
                "final_step": final_step,
                "final_value": FLAGS.metra_reward_final_coef,
            }
        args = replace(
            args,
            use_rnd=FLAGS.use_rnd,
            use_height_map=FLAGS.use_height_map,
            symmetric_obs=FLAGS.symmetric_obs,
            video=FLAGS.video,
        )
        if FLAGS.video_length is not None:
            args = replace(args, video_length=FLAGS.video_length)
        if FLAGS.video_interval is not None:
            args = replace(args, video_interval=FLAGS.video_interval)

        launch_training(task_id=env_name, args=args)

    elif FLAGS.agent_name == "flash":
        from train_off_policy import launch_off_policy, TrainConfig

        args = TrainConfig.from_task(task_id=env_name)
        args.env.scene.num_envs = FLAGS.num_envs

        args.agent.project_name = project_name
        args.agent.num_train_envs = FLAGS.num_envs
        args.agent.seed = FLAGS.seed

        launch_off_policy(task_id=env_name, cfg=args)

    else:
        raise ValueError(f"Unsupported agent name: {FLAGS.agent_name}")


if __name__ == "__main__":
    app.run(main)
