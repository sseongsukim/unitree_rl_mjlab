"""Helpers to register local tasks without importing the whole src.tasks tree."""

from __future__ import annotations

import importlib


_LOCAL_TASK_MODULES = (
  "src.tasks.empty.config.go2",
  "src.tasks.jump.config.go2",
)


def register_local_tasks() -> None:
  """Import the local task modules needed in this workspace."""
  for module_name in _LOCAL_TASK_MODULES:
    importlib.import_module(module_name)
