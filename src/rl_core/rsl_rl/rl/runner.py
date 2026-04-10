import os

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
    ) -> None:
        # Strip None-valued optional configs so model constructors only receive
        # parameters that are actually set for the selected model class.
        for key in ("actor", "critic"):
            if key in train_cfg:
                for opt, value in list(train_cfg[key].items()):
                    if value is None:
                        train_cfg[key].pop(opt, None)
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
