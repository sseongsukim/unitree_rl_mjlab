import os

import numpy as np
import torch
import wandb

from src.rl_core.rsl_rl.env import VecEnv
from src.rl_core.rsl_rl.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from src.rl_core.rsl_rl.runners import OnPolicyRunner
from src.rl_core.rsl_rl.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
    """Base runner that persists environment state across checkpoints."""

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        eval_env: VecEnv | None = None,
        eval_interval: int | None = None,
        eval_video: bool = True,
        eval_video_length: int = 200,
    ) -> None:
        # Strip None-valued optional configs so MLPModel doesn't receive them.
        for key in ("actor", "critic"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.eval_video = eval_video
        self.eval_video_length = eval_video_length
        super().__init__(env, train_cfg, log_dir, device)

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export policy to ONNX format using legacy export path.

        Overrides the base implementation to set dynamo=False, avoiding warnings about
        dynamic_axes being deprecated with the new TorchDynamo export path
        (torch>=2.9 default).
        """
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore[operator]
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore[arg-type]
            output_names=onnx_model.output_names,  # type: ignore[arg-type]
            dynamic_axes={},
            dynamo=False,
        )

    def save(self, path: str, infos=None) -> None:
        """Save checkpoint.

        Extends the base implementation to persist the environment's
        common_step_counter and to respect the ``upload_model`` config flag.
        """
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load checkpoint.

        Extends the base implementation to:
        1. Restore common_step_counter to preserve curricula state.
        2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
          -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
        """
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

        if "model_state_dict" in loaded_dict:
            print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
            model_state_dict = loaded_dict.pop("model_state_dict")
            actor_state_dict = {}
            critic_state_dict = {}

            for key, value in model_state_dict.items():
                # Migrate actor keys.
                if key.startswith("actor."):
                    new_key = key.replace("actor.", "mlp.")
                    actor_state_dict[new_key] = value
                elif key.startswith("actor_obs_normalizer."):
                    new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
                    actor_state_dict[new_key] = value
                elif key in ["std", "log_std"]:
                    actor_state_dict[key] = value

                # Migrate critic keys.
                if key.startswith("critic."):
                    new_key = key.replace("critic.", "mlp.")
                    critic_state_dict[new_key] = value
                elif key.startswith("critic_obs_normalizer."):
                    new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
                    critic_state_dict[new_key] = value

            loaded_dict["actor_state_dict"] = actor_state_dict
            loaded_dict["critic_state_dict"] = critic_state_dict

        # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
        actor_sd = loaded_dict.get("actor_state_dict", {})
        if "std" in actor_sd:
            actor_sd["distribution.std_param"] = actor_sd.pop("std")
        if "log_std" in actor_sd:
            actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]

        infos = loaded_dict["infos"]
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"][
                "common_step_counter"
            ]
        return infos

    def after_iteration(self, it: int) -> None:
        if self.eval_env is None or self.logger.writer is None:
            return
        if self.eval_interval is None or it == 0 or it % self.eval_interval != 0:
            return

        eval_metrics = self._evaluate_policy(record_video=self.eval_video)
        for key, value in eval_metrics.items():
            if key == "Eval/video":
                continue
            self.logger.writer.add_scalar(key, value, it)  # type: ignore[arg-type]
        if "Eval/video" in eval_metrics and hasattr(self.logger.writer, "add_video"):
            self.logger.writer.add_video(  # type: ignore[call-arg]
                "Eval/video",
                eval_metrics["Eval/video"],
                global_step=it,
                fps=30,
            )

    def _evaluate_policy(
        self, record_video: bool = False
    ) -> dict[str, float | torch.Tensor]:
        assert self.eval_env is not None

        def _extract_rgb_frame(frame) -> np.ndarray | None:
            if frame is None:
                return None
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.ndim != 3:
                return None
            if frame.shape[-1] == 3:
                return frame
            if frame.shape[0] == 3:
                return np.transpose(frame, (1, 2, 0))
            return None

        self.alg.eval_mode()
        policy = self.get_inference_policy(device=self.device)
        obs, _ = self.eval_env.reset()
        obs = obs.to(self.device)

        episode_return = 0.0
        episode_length = 0
        done = False
        video_frames: list[np.ndarray] = []

        if record_video:
            frame = _extract_rgb_frame(self.eval_env.render())
            if frame is not None:
                video_frames.append(frame)

        while not done and episode_length < self.eval_env.max_episode_length:
            with torch.no_grad():
                actions = policy(obs)
            obs, rewards, dones, _ = self.eval_env.step(actions.to(self.eval_env.device))
            obs = obs.to(self.device)

            episode_return += float(rewards[0].item())
            episode_length += 1
            done = bool(dones[0].item())

            if record_video:
                frame = _extract_rgb_frame(self.eval_env.render())
                if frame is not None:
                    video_frames.append(frame)

        self.alg.train_mode()

        metrics: dict[str, float | torch.Tensor] = {
            "Eval/episode_return": episode_return,
            "Eval/episode_length": float(episode_length),
        }
        if video_frames:
            video_frames = video_frames[: self.eval_video_length]
            video = np.stack(video_frames, axis=0)
            video = np.expand_dims(video, axis=0).transpose(0, 1, 4, 2, 3)
            metrics["Eval/video"] = torch.from_numpy(video)
        return metrics


class ProjectOnPolicyRunner(MjlabOnPolicyRunner):
    """Project-wide on-policy runner with ONNX export and metadata attachment."""

    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        self.export_policy_to_onnx(policy_path, filename)
        run_name: str = (
            wandb.run.name
            if self.logger.logger_type == "wandb" and wandb.run
            else "local"
        )  # type: ignore[assignment]
        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        if self.logger.logger_type in ["wandb"]:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
