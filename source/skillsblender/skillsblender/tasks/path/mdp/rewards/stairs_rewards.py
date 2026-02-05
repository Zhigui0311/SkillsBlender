from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .base_rewards import feet_air_time_1, skill_phase_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stairs_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    dis_threshold: float = 0.25,
    heading_threshold: float = 0.5,
) -> torch.Tensor:
    """Air-time reward gated to stairs segments only."""
    reward = feet_air_time_1(
        env=env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        threshold=threshold,
        dis_threshold=dis_threshold,
        heading_threshold=heading_threshold,
    )
    is_stairs = skill_phase_mask(env, command_name, "stairs_up") | skill_phase_mask(
        env, command_name, "stairs_down"
    )
    return reward * is_stairs.float()
