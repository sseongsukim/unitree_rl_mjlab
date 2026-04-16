from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from tensordict import TensorDict

from src.rl_core.rsl_rl.models import MLPModel


class METRAPPO:
    """Proximal Policy Optimization algorithm + METRA"""

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    traj_encoder: MLPModel
    """The trajectory encoder model."""
