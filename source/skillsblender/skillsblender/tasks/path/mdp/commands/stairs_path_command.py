from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StairsPathCommand(SegmentPathCommand):
    """Path with STAIRS transitions (planned, not sensed)."""

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv, stairs_up: bool = True):
        super().__init__(cfg, env)
        self._stairs_up = stairs_up

    @property
    def is_in_stairs_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("stairs_up") | self.is_in_skill_phase("stairs_down")

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        sp = self.cfg.stairs_params
        total_len = torch.full((len(env_ids),), float(self.cfg.ranges.default_path_len), device=self.device)

        s_min, s_max = sp.start_dist_range
        first_s0 = torch.empty(len(env_ids), device=self.device).uniform_(s_min, s_max)

        stairs_len = float(sp.stairs_len)
        # Reserve space for both stair segments and a tiny trailing walk region.
        first_s1_cap = (total_len - stairs_len - 0.2).clamp(min=first_s0 + 0.6)
        first_s1 = torch.minimum(first_s0 + stairs_len, first_s1_cap)
        second_s0 = first_s1
        second_s1 = (second_s0 + stairs_len).clamp(max=total_len - 1e-3)

        self._path_len[env_ids] = total_len

        start = self._planned_start_pos[env_ids].clone()
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha

        dist_at_wp = alpha.squeeze(-1) * total_len[:, None]
        step_len = float(sp.step_length)
        step_h = abs(float(sp.step_height))

        # Default behavior: first go up, then down.
        first_is_up = bool(self._stairs_up)

        inside_first = (dist_at_wp >= first_s0[:, None]) & (dist_at_wp <= first_s1[:, None])
        inside_second = (dist_at_wp >= second_s0[:, None]) & (dist_at_wp <= second_s1[:, None])

        first_local = (dist_at_wp - first_s0[:, None]).clamp(min=0.0)
        first_step_idx = torch.floor(first_local / step_len)
        first_num_steps = torch.floor((first_s1 - first_s0) / step_len).clamp(min=1.0)
        first_top_h = first_num_steps * step_h

        first_up_profile = torch.minimum(first_step_idx * step_h, first_top_h[:, None])
        first_down_profile = -first_up_profile
        first_profile = first_up_profile if first_is_up else first_down_profile

        second_local = (dist_at_wp - second_s0[:, None]).clamp(min=0.0)
        second_step_idx = torch.floor(second_local / step_len)
        second_down_profile = (first_top_h[:, None] - second_step_idx * step_h).clamp(min=0.0)
        second_up_profile = -second_down_profile
        second_profile = second_down_profile if first_is_up else second_up_profile

        z_off = torch.zeros_like(dist_at_wp)
        z_off = torch.where(inside_first, first_profile, z_off)
        z_off = torch.where(inside_second, second_profile, z_off)
        pos[..., 2] = pos[..., 2] + z_off

        yaw_wp = self._planned_yaw[env_ids][:, None].repeat(1, self.num_waypoints)
        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)

        lead_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(lead_params, "v_ref", 1.0)
        self._append_segment(env_ids, self.SKILL_ID["walk"], zero, first_s0, params=lead_params)

        first_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(first_params, "v_ref", 0.8)
        self._set_seg_param(first_params, "clearance_ref", step_h * 2.0)
        self._set_seg_param(first_params, "misc", float(sp.step_length))
        first_skill = self.SKILL_ID["stairs_up"] if first_is_up else self.SKILL_ID["stairs_down"]
        self._append_segment(env_ids, int(first_skill), first_s0, first_s1, params=first_params)

        second_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(second_params, "v_ref", 0.75)
        self._set_seg_param(second_params, "clearance_ref", step_h * 2.0)
        self._set_seg_param(second_params, "misc", float(sp.step_length))
        second_skill = self.SKILL_ID["stairs_down"] if first_is_up else self.SKILL_ID["stairs_up"]
        self._append_segment(env_ids, int(second_skill), second_s0, second_s1, params=second_params)

        trail_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(trail_params, "v_ref", 1.0)
        self._append_segment(env_ids, self.SKILL_ID["walk"], second_s1, total_len, params=trail_params)

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
