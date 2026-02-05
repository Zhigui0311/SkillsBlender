from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Tuple

import torch
from isaaclab.utils import configclass

from .virtual_jump_path_command import VirtualJumpPathCommand, VirtualJumpPathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class VirtualClimbPathCommandCfg(VirtualJumpPathCommandCfg):
    """Virtual climb command config (flat terrain hallucination)."""

    climb_prob: float = 0.02
    climb_height_range: Tuple[float, float] = (0.22, 0.45)
    climb_length_range: Tuple[float, float] = (1.0, 1.8)

    def __post_init__(self):
        self.jump_prob = self.climb_prob
        self.jump_height_range = self.climb_height_range
        self.jump_length_range = self.climb_length_range
        self.skill_name = "climb"
        self.profile_type = "ramp_up"
        super().__post_init__()
        if getattr(self, "class_type", None) in (None, MISSING):
            self.class_type = VirtualClimbPathCommand


class VirtualClimbPathCommand(VirtualJumpPathCommand):
    """Virtual climb command with synthetic slope-state outputs."""

    cfg: VirtualClimbPathCommandCfg

    def __init__(self, cfg: VirtualClimbPathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Optional synthetic gravity projection in body frame for debugging/custom observations.
        self.virtual_projected_gravity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.virtual_projected_gravity_b[:, 2] = -1.0
        self.heading_target = torch.zeros(self.num_envs, device=self.device)

    @property
    def is_in_climb_phase(self):
        return self.is_in_skill_phase("climb")

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)
        self.heading_target[env_ids] = self._planned_yaw[env_ids]

        seg_len = torch.clamp(self._jump_end_dist[env_ids] - self._jump_start_dist[env_ids], min=1.0e-3)
        pitch = torch.atan(self._jump_height[env_ids] / seg_len)
        active = self._state[env_ids] == self.STATE_JUMPING
        g = torch.zeros(len(env_ids), 3, device=self.device)
        g[:, 0] = torch.sin(pitch)
        g[:, 2] = -torch.cos(pitch)
        # Only publish synthetic slope gravity for active climb windows.
        default_g = torch.zeros_like(g)
        default_g[:, 2] = -1.0
        self.virtual_projected_gravity_b[env_ids] = torch.where(active[:, None], g, default_g)
