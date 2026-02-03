from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StairsPathCommand(SegmentPathCommand):
    """Path with a STAIRS_UP or STAIRS_DOWN segment (planned, not sensed)."""

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv, stairs_up: bool = True):
        super().__init__(cfg, env)
        self._stairs_up = stairs_up

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        sp = self.cfg.stairs_params
        total_len = torch.full((len(env_ids),), float(self.cfg.ranges.default_path_len), device=self.device)

        s_min, s_max = sp.start_dist_range
        stairs_s0 = torch.empty(len(env_ids), device=self.device).uniform_(s_min, s_max)
        stairs_s1 = (stairs_s0 + float(sp.stairs_len)).clamp(max=total_len - 1e-3)

        self._path_len[env_ids] = total_len

        start = self._planned_start_pos[env_ids].clone()
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha

        dist_at_wp = alpha.squeeze(-1) * total_len[:, None]
        inside = (dist_at_wp >= stairs_s0[:, None]) & (dist_at_wp <= stairs_s1[:, None])

        step_len = float(sp.step_length)
        step_h = float(sp.step_height) * (1.0 if self._stairs_up else -1.0)
        local_s = (dist_at_wp - stairs_s0[:, None]).clamp(min=0.0)
        step_idx = torch.floor(local_s / step_len)
        z_off = step_idx * step_h
        pos[..., 2] = pos[..., 2] + torch.where(inside, z_off, torch.zeros_like(z_off))

        yaw_wp = self._planned_yaw[env_ids][:, None].repeat(1, self.num_waypoints)
        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)

        lead_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        lead_params[:, self.SEG_PARAM["v_ref"]] = 1.0
        self._append_segment(env_ids, self.SKILL_ID["walk"], zero, stairs_s0, params=lead_params)

        st_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        st_params[:, self.SEG_PARAM["v_ref"]] = 0.9
        st_params[:, self.SEG_PARAM["clearance_ref"]] = abs(step_h) * 2.0
        st_params[:, self.SEG_PARAM["misc"]] = float(sp.step_length)

        skill = self.SKILL_ID["stairs_up"] if self._stairs_up else self.SKILL_ID["stairs_down"]
        self._append_segment(env_ids, int(skill), stairs_s0, stairs_s1, params=st_params)

        trail_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        trail_params[:, self.SEG_PARAM["v_ref"]] = 1.0
        self._append_segment(env_ids, self.SKILL_ID["walk"], stairs_s1, total_len, params=trail_params)

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
