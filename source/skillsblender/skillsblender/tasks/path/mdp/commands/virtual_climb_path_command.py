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
    # Legacy height range (unused for Z, kept for curriculum compatibility).
    climb_height_range: Tuple[float, float] = (0.0, 0.0)
    climb_length_range: Tuple[float, float] = (1.0, 1.8)
    # Target pitch angle magnitude in degrees (± range).
    pitch_deg_range: Tuple[float, float] = (15.0, 15.0)
    pitch_random_sign: bool = True

    def __post_init__(self):
        if getattr(self, "climb_prob", 0.0):
            self.virtual_prob = float(self.climb_prob)
        self.gap_width_range = self.climb_length_range
        self.jump_length_range = self.climb_length_range
        self.skill_name = "climb"
        # Keep path Z unchanged; climb is enforced via pitch target only.
        self.profile_type = "none"
        # No vertical hallucination for climb (kept for curriculum compatibility).
        self.jump_height_range = self.climb_height_range
        super().__post_init__()
        if getattr(self, "class_type", None) in (None, MISSING):
            self.class_type = VirtualClimbPathCommand


class VirtualClimbPathCommand(VirtualJumpPathCommand):
    """Virtual climb command with synthetic slope-state outputs."""

    cfg: "VirtualClimbPathCommandCfg"

    def __init__(self, cfg: "VirtualClimbPathCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Pitch target (radians) used by the virtual climb reward.
        self.pitch_target = torch.zeros(self.num_envs, device=self.device)

    @property
    def is_in_climb_phase(self):
        return self.is_in_skill_phase("climb")

    def _on_trigger(
        self,
        env_ids: torch.Tensor,
        jump_start: torch.Tensor,
        jump_end: torch.Tensor,
        jump_height: torch.Tensor,
        stairs_steps: torch.Tensor,
    ):
        # Sample pitch target in degrees, convert to radians.
        pitch_deg = self._sample_uniform(len(env_ids), *self.cfg.pitch_deg_range)
        pitch_rad = torch.deg2rad(pitch_deg)
        if self.cfg.pitch_random_sign:
            sign = torch.where(
                torch.rand(len(env_ids), device=self.device) < 0.5,
                torch.full((len(env_ids),), -1.0, device=self.device),
                torch.ones(len(env_ids), device=self.device),
            )
            pitch_rad = pitch_rad * sign
        self.pitch_target[env_ids] = pitch_rad

    def _on_finish(self, env_ids: torch.Tensor):
        self.pitch_target[env_ids] = 0.0

    def _on_reset(self, env_ids: torch.Tensor):
        self.pitch_target[env_ids] = 0.0

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)
        active = self._state[env_ids] == self.STATE_JUMPING
        if self.has_seg_params and self._seg_params is not None:
            slope_idx = self.SEG_PARAM.get("slope_angle_ref", None)
            if slope_idx is not None and slope_idx < self.num_seg_params:
                env_sel = env_ids[active]
                if env_sel.numel() > 0:
                    # climb segment is the second segment (index 1) when active.
                    self._seg_params[env_sel, 1, slope_idx] = self.pitch_target[env_sel]
