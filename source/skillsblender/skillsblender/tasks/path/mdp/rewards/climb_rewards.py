from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

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
    """Path-direction speed reward gated to platform segments only."""
    reward = track_velocity_along_path_exp(
        env=env,
        std=std,
        asset_cfg=asset_cfg,
        command_name=command_name,
        desired_speed=desired_speed,
    )
    mask = skill_phase_mask(env, command_name, "platform")
    return reward * mask.float()

#------------------------virtual climb specific rewards------------------------#

def track_base_pitch(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    command_name: str = "path_tracking",
) -> torch.Tensor:
    """Track target base pitch from virtual climb command."""
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "pitch_target"):
        return torch.zeros(env.num_envs, device=env.device)

    quat = env.scene["robot"].data.root_quat_w
    _, pitch, _ = euler_xyz_from_quat(quat)
    target_pitch = command.pitch_target
    err = wrap_to_pi(pitch - target_pitch)
    reward = torch.exp(-torch.square(err) / (std**2))
    if hasattr(command, "is_in_climb_phase"):
        reward = torch.where(command.is_in_climb_phase, reward, torch.zeros_like(reward))
    return reward
