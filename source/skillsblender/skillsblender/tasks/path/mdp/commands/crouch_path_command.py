from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CrouchPathCommand(SegmentPathCommand):
    """Path with a CROUCH segment (low stance)."""

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        cp = self.cfg.crouch_params
        total_len = torch.full((len(env_ids),), float(self.cfg.ranges.default_path_len), device=self.device)

        s_min, s_max = cp.start_dist_range
        crouch_s0 = torch.empty(len(env_ids), device=self.device).uniform_(s_min, s_max)
        crouch_s1 = (crouch_s0 + float(cp.crouch_len)).clamp(max=total_len - 1e-3)

        self._path_len[env_ids] = total_len

        start = self._planned_start_pos[env_ids].clone()
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha

        yaw_wp = self._planned_yaw[env_ids][:, None].repeat(1, self.num_waypoints)
        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)

        # Leading walk segment
        lead_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        lead_params[:, self.SEG_PARAM["v_ref"]] = 1.0
        self._append_segment(env_ids, self.SKILL_ID["walk"], zero, crouch_s0, params=lead_params)

        # Crouch segment
        crouch_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        crouch_params[:, self.SEG_PARAM["v_ref"]] = 0.7  # Slower speed
        crouch_params[:, self.SEG_PARAM["base_height_ref"]] = float(cp.base_height_ref)  # Lower height
        self._append_segment(env_ids, self.SKILL_ID["crouch"], crouch_s0, crouch_s1, params=crouch_params)

        # Trailing walk segment
        trail_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        trail_params[:, self.SEG_PARAM["v_ref"]] = 1.0
        self._append_segment(env_ids, self.SKILL_ID["walk"], crouch_s1, total_len, params=trail_params)

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
