from __future__ import annotations

"""Compatibility reward entrypoint.

This module is kept for backward compatibility. The implementation has been
split into:
- `base_rewards.py` for common task/regularization/gait rewards
- `jump_rewards.py` for jump-only rewards
- per-skill files (`walk_rewards.py`, `stairs_rewards.py`, `climb_rewards.py`, `crouch_rewards.py`)
"""

from .base_rewards import *
from .jump_rewards import *
from .walk_rewards import *
from .stairs_rewards import *
from .climb_rewards import *
from .crouch_rewards import *
