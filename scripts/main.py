from absl import app, flags

from train_ppo import launch_training, TrainConfig
from train_off_policy import launch_off_policy, TrainConfig

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_envs", 4096, "Number of environments.")
flags.DEFINE_integer("seed", 42, "Seed number.")
flags.DEFINE_string("agent_name", "flash", "Agent name.")
flags.DEFINE_string("env_name", "jump", "Task environment name.")


register_env_dict = {
    "jump": "Unitree-Go2-Jump",
    "leap": "Unitree-Go2-Leap",
    "flat": "Unitree-Go2-Flat",
}


def main(_):
    env_name = register_env_dict[FLAGS.env_name]
    project_name = f"{env_name}_{FLAGS.agent_name}"

    if FLAGS.agent_name == "ppo":
        args = TrainConfig.from_task(task_id=env_name)
        args.env.scene.num_envs = FLAGS.num_envs

        args.agent.wandb_project = project_name
        args.agent.seed = FLAGS.seed
        launch_training(task_id=env_name, args=args)

    elif FLAGS.agent_name == "flash":
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
