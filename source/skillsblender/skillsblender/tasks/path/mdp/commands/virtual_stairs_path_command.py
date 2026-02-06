from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.utils import configclass

from .virtual_jump_path_command import VirtualJumpPathCommand, VirtualJumpPathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class VirtualStairsPathCommandCfg(VirtualJumpPathCommandCfg):
    """Virtual stairs command config (flat terrain hallucination)."""

    run_prob: float = 0.02
    stairs_up_prob: float = 0.5
    step_height: float = 0.15
    step_width: float = 0.30
    stairs_length_range: Tuple[float, float] = (1.2, 2.4)
    stairs_steps_range: Tuple[int, int] = (3, 6)
    max_up_height: float = 0.40
    max_down_depth: float = 0.25
    down_pitch_deg: float = -10.0

    def __post_init__(self):
        self.jump_prob = self.run_prob
        self.jump_length_range = self.stairs_length_range
        # Height handled per-trigger; keep range non-zero for base sampling.
        self.jump_height_range = (self.step_height, self.step_height)
        self.skill_name = "stairs"
        self.profile_type = "stairs"
        super().__post_init__()
        if getattr(self, "class_type", None) in (None, MISSING):
            self.class_type = VirtualStairsPathCommand


class VirtualStairsPathCommand(VirtualJumpPathCommand):
    """Virtual stairs command using step-function Z hallucination."""

    cfg: VirtualStairsPathCommandCfg

    def __init__(self, cfg: VirtualStairsPathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stairs_dir = torch.ones(self.num_envs, device=self.device)
        self.pitch_target = torch.zeros(self.num_envs, device=self.device)

    @property
    def is_in_stairs_phase(self):
        return self.is_in_skill_phase("stairs_up") | self.is_in_skill_phase("stairs_down")

    def _on_trigger(
        self,
        env_ids: torch.Tensor,
        jump_start: torch.Tensor,
        jump_end: torch.Tensor,
        jump_height: torch.Tensor,
        stairs_steps: torch.Tensor,
    ):
        # Determine direction per-env.
        up = torch.rand(len(env_ids), device=self.device) < float(self.cfg.stairs_up_prob)
        dir_sign = torch.where(
            up,
            torch.ones(len(env_ids), device=self.device),
            -torch.ones(len(env_ids), device=self.device),
        )
        self._stairs_dir[env_ids] = dir_sign

        # Derive steps from run length and step width.
        run_len = torch.clamp(jump_end - jump_start, min=1.0e-3)
        step_w = max(float(self.cfg.step_width), 1.0e-3)
        n_steps = torch.clamp(torch.floor(run_len / step_w).to(torch.long), min=1)
        lo, hi = self.cfg.stairs_steps_range
        n_steps = torch.clamp(n_steps, min=int(lo), max=int(hi))

        # Limit total height for up/down.
        total_height = n_steps.to(torch.float32) * float(self.cfg.step_height)
        max_total = torch.where(
            up,
            torch.full_like(total_height, float(self.cfg.max_up_height)),
            torch.full_like(total_height, float(self.cfg.max_down_depth)),
        )
        scale = torch.clamp(max_total / torch.clamp(total_height, min=1.0e-6), max=1.0)
        total_height = total_height * scale

        self._stairs_steps[env_ids] = n_steps
        # Use total height (not per-step) for the step-function profile.
        self._jump_height[env_ids] = total_height

        down_pitch = torch.deg2rad(
            torch.full_like(total_height, float(self.cfg.down_pitch_deg))
        )
        self.pitch_target[env_ids] = torch.where(up, torch.zeros_like(total_height), down_pitch)

    def _on_finish(self, env_ids: torch.Tensor):
        self._stairs_dir[env_ids] = 1.0
        self.pitch_target[env_ids] = 0.0

    def _on_reset(self, env_ids: torch.Tensor):
        self._stairs_dir[env_ids] = 1.0
        self.pitch_target[env_ids] = 0.0

    def _resample_command(self, env_ids: torch.Tensor):
        # Keep remaining stairs distance before resetting the planned frame.
        prev_s = self._dist_along_planned()[env_ids]
        remaining_s0 = self._jump_start_dist[env_ids] - prev_s
        remaining_s1 = self._jump_end_dist[env_ids] - prev_s

        self._set_planned_frame_from_robot(env_ids)
        pos_traj, yaw_traj, total_len = self._generate_trajectory(
            env_ids,
            remaining_s0=remaining_s0,
            remaining_s1=remaining_s1,
        )

        self._path_len[env_ids] = total_len
        self.pos_path_w[env_ids] = pos_traj
        self.heading_path_w[env_ids, :, 0] = yaw_traj

        # Build segment table with per-env stairs direction.
        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)
        stairs_active = self._state[env_ids] == self.STATE_JUMPING
        stairs_s0 = self._jump_start_dist[env_ids]
        stairs_s1 = self._jump_end_dist[env_ids]

        lead_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(lead_params, "v_ref", 1.0)
        self._append_segment(
            env_ids,
            self.SKILL_ID.get("walk", 0),
            zero,
            torch.where(stairs_active, stairs_s0, total_len),
            params=lead_params,
        )

        stairs_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(stairs_params, "v_ref", 1.2)
        self._set_seg_param(stairs_params, "clearance_ref", 0.16)
        # Force high step-height reference regardless of direction.
        self._set_seg_param(stairs_params, "step_height_ref", float(self.cfg.step_height))
        self._set_seg_param(stairs_params, "misc", self._stairs_steps[env_ids].to(torch.float32))

        up_mask = stairs_active & (self._stairs_dir[env_ids] >= 0.0)
        down_mask = stairs_active & (self._stairs_dir[env_ids] < 0.0)
        if torch.any(up_mask):
            env_up = env_ids[up_mask]
            self._append_segment(
                env_up,
                self.SKILL_STAIRS_UP,
                stairs_s0[up_mask],
                stairs_s1[up_mask],
                params=stairs_params[up_mask],
            )
        if torch.any(down_mask):
            env_down = env_ids[down_mask]
            self._append_segment(
                env_down,
                self.SKILL_STAIRS_DOWN,
                stairs_s0[down_mask],
                stairs_s1[down_mask],
                params=stairs_params[down_mask],
            )

        trail_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(trail_params, "v_ref", 1.0)
        self._append_segment(
            env_ids,
            self.SKILL_ID.get("walk", 0),
            torch.where(stairs_active, stairs_s1, total_len),
            total_len,
            params=trail_params,
        )

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
