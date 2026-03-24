"""Helpers to register local tasks needed by CLI scripts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_LOCAL_TASK_MODULES = (
  "src.tasks.jump.config.go2",
)


def register_local_tasks() -> None:
  repo_root = Path(__file__).resolve().parent.parent
  repo_root_str = str(repo_root)
  if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
  for module_name in _LOCAL_TASK_MODULES:
    importlib.import_module(module_name)
