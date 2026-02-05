from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .base_rewards import skill_phase_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def crouch_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str = "path_tracking",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base-height penalty gated to crouch segments only."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    mask = skill_phase_mask(env, command_name, "crouch")
    return error * mask.float()
