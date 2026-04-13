from collections import deque
from dataclasses import asdict
import statistics
import time
from typing import Any, Optional

import numpy as np
import torch


class WandbTrainerLogger:
    def __init__(self, cfg: Any, num_envs: int, device: torch.device | str):
        import wandb

        self._wandb = wandb
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.tot_timesteps = 0
        self.tot_time = 0.0
        dict_cfg = asdict(cfg)
        wandb.init(
            project=cfg.agent.project_name,
            config=dict_cfg,  # type: ignore,
            mode="online" if cfg.agent.use_wandb else "offline",
        )
        self.ep_extras: list[dict[str, Any]] = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.cur_episode_length = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.media_dict: dict[str, Any] = {}
        self.reset()

    def update_metric(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                self.average_meter_dict.update(k, v)
            elif isinstance(v, np.ndarray) and v.ndim == 5:
                self.media_dict[k] = self._wandb.Video(v, fps=30, format="gif")
            else:
                self.media_dict[k] = v

    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminateds: torch.Tensor,
        truncateds: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        if "episode" in extras:
            self.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self.ep_extras.append(extras["log"])

        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        dones = torch.logical_or(terminateds > 0, truncateds > 0)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return

        self.rewbuffer.extend(self.cur_reward_sum[done_ids].detach().cpu().tolist())
        self.lenbuffer.extend(self.cur_episode_length[done_ids].detach().cpu().tolist())
        self.cur_reward_sum[done_ids] = 0
        self.cur_episode_length[done_ids] = 0

    def log_metric(self, step: int) -> None:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        self._wandb.log(log_data, step=step)

    def log(self, step: int, collect_time: float, learn_time: float) -> None:
        log_data = {}
        averaged_metrics = self.average_meter_dict.averages()
        log_data.update(averaged_metrics)
        log_data.update(self.media_dict)

        extras_lines = []
        if self.ep_extras:
            keys = set().union(*(ep_info.keys() for ep_info in self.ep_extras))
            for key in sorted(keys):
                values = []
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    value = ep_info[key]
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 0:
                            continue
                        values.append(value.detach().reshape(-1).float().cpu())
                    else:
                        values.append(torch.tensor([value], dtype=torch.float32))

                if not values:
                    continue

                mean_value = torch.cat(values).mean().item()
                log_key = key if "/" in key else f"Episode/{key}"
                log_data[log_key] = mean_value
                extras_lines.append(f"{log_key}: {mean_value:.4f}")

        if len(self.rewbuffer) > 0:
            mean_reward = statistics.mean(self.rewbuffer)
            mean_episode_length = statistics.mean(self.lenbuffer)
            log_data["Train/mean_reward"] = mean_reward
            log_data["Train/mean_episode_length"] = mean_episode_length

        iteration_time = collect_time + learn_time
        self.tot_timesteps = step * self.num_envs
        self.tot_time += iteration_time
        fps = int(self.num_envs / iteration_time) if iteration_time > 0 else 0
        log_data["Perf/total_fps"] = fps
        log_data["Perf/collection_time"] = collect_time
        log_data["Perf/learning_time"] = learn_time

        self._wandb.log(log_data, step=step)

        log_lines = [
            "#" * 80,
            f"Learning iteration {step}",
            f"Total steps: {self.tot_timesteps}",
            f"Steps per second: {fps}",
            f"Collection time: {collect_time:.3f}s",
            f"Learning time: {learn_time:.3f}s",
        ]
        for key, value in averaged_metrics.items():
            if isinstance(value, (int, float)):
                log_lines.append(f"{key}: {value:.4f}")
        if len(self.rewbuffer) > 0:
            log_lines.append(f"Mean reward: {statistics.mean(self.rewbuffer):.2f}")
            log_lines.append(
                f"Mean episode length: {statistics.mean(self.lenbuffer):.2f}"
            )
        log_lines.extend(extras_lines)
        log_lines.extend(
            [
                "-" * 80,
                f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}",
            ]
        )
        print("\n".join(log_lines))

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()
        self.ep_extras.clear()


class AverageMeter:
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format: str) -> str:
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )


class AverageMeterDict:
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters: Optional[dict[str, AverageMeter]] = None):
        self.meters = meters if meters else {}

    def __getitem__(self, key: str) -> AverageMeter:
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string: str = "{}") -> dict[str, float]:
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string: str = "{}") -> dict[str, float]:
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }
