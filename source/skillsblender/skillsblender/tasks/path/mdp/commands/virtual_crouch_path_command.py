from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Tuple

from isaaclab.utils import configclass

from .virtual_jump_path_command import VirtualJumpPathCommand, VirtualJumpPathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class VirtualCrouchPathCommandCfg(VirtualJumpPathCommandCfg):
    """Virtual crouch command config (flat terrain hallucination)."""

    crouch_prob: float = 0.02
    # Positive depth magnitude; command applies negative Z offset in crouch segment.
    crouch_depth_range: Tuple[float, float] = (0.22, 0.34)
    crouch_length_range: Tuple[float, float] = (1.2, 2.0)

    def __post_init__(self):
        # Backward-compat: allow crouch_prob to set virtual_prob.
        if getattr(self, "crouch_prob", 0.0):
            self.virtual_prob = float(self.crouch_prob)
        self.jump_height_range = self.crouch_depth_range
        self.gap_width_range = self.crouch_length_range
        self.jump_length_range = self.crouch_length_range
        self.skill_name = "crouch"
        self.profile_type = "smooth_down"
        super().__post_init__()
        if getattr(self, "class_type", None) in (None, MISSING):
            self.class_type = VirtualCrouchPathCommand


class VirtualCrouchPathCommand(VirtualJumpPathCommand):
    """Virtual crouch command based on the shared hallucination state-machine."""

    cfg: VirtualCrouchPathCommandCfg

    def __init__(self, cfg: VirtualCrouchPathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    @property
    def is_in_crouch_phase(self):
        return self.is_in_skill_phase("crouch")
