from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .base_rewards import skill_phase_mask, track_velocity_along_path_exp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def climb_track_velocity_along_path_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "path_tracking",
    desired_speed: float = 0.65,
) -> torch.Tensor:
    """Path-direction speed reward gated to climb segments only."""
    reward = track_velocity_along_path_exp(
        env=env,
        std=std,
        asset_cfg=asset_cfg,
        command_name=command_name,
        desired_speed=desired_speed,
    )
    mask = skill_phase_mask(env, command_name, "climb")
    return reward * mask.float()
