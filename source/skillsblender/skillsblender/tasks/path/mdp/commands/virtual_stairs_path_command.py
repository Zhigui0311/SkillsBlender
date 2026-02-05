from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal, Tuple

from isaaclab.utils import configclass

from .virtual_jump_path_command import VirtualJumpPathCommand, VirtualJumpPathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class VirtualStairsPathCommandCfg(VirtualJumpPathCommandCfg):
    """Virtual stairs command config (flat terrain hallucination)."""

    stairs_prob: float = 0.02
    step_height_range: Tuple[float, float] = (0.12, 0.20)
    step_width_range: Tuple[float, float] = (0.25, 0.40)
    num_steps_range: Tuple[int, int] = (3, 6)
    direction: Literal["up", "down"] = "up"

    def __post_init__(self):
        self.jump_prob = self.stairs_prob
        self.jump_height_range = self.step_height_range
        min_len = self.step_width_range[0] * max(1, int(self.num_steps_range[0]))
        max_len = self.step_width_range[1] * max(1, int(self.num_steps_range[1]))
        self.jump_length_range = (min_len, max_len)
        self.stairs_steps_range = self.num_steps_range
        if self.direction == "down":
            self.skill_name = "stairs_down"
            self.profile_type = "stairs_down"
        else:
            self.skill_name = "stairs_up"
            self.profile_type = "stairs_up"
        super().__post_init__()
        if getattr(self, "class_type", None) in (None, MISSING):
            self.class_type = VirtualStairsPathCommand


class VirtualStairsPathCommand(VirtualJumpPathCommand):
    """Virtual stairs command using step-function Z hallucination."""

    cfg: VirtualStairsPathCommandCfg
    SKILL_STAIRS = VirtualJumpPathCommand.SKILL_STAIRS_UP

    def __init__(self, cfg: VirtualStairsPathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    @property
    def is_in_stairs_phase(self):
        return self.is_in_skill_phase("stairs_up") | self.is_in_skill_phase("stairs_down")
