from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ClimbPathCommand(SegmentPathCommand):
    """Path with a CLIMB segment (ramp up/down, max 30 deg)."""

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    @property
    def is_in_climb_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("climb")

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        sp = self.cfg.climb_params
        s_min, s_max = sp.start_dist_range
        min_len = float(s_max + sp.climb_len + 0.2)
        total_len = self._sample_path_length(env_ids, self._planned_start_pos[env_ids], self._planned_yaw[env_ids], min_len=min_len)
        slope_s0 = torch.empty(len(env_ids), device=self.device).uniform_(s_min, s_max)
        slope_s1 = (slope_s0 + float(sp.climb_len)).clamp(max=total_len - 1e-3)

        self._path_len[env_ids] = total_len

        start = self._planned_start_pos[env_ids].clone()
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha

        dist_at_wp = alpha.squeeze(-1) * total_len[:, None]
        inside = (dist_at_wp >= slope_s0[:, None]) & (dist_at_wp <= slope_s1[:, None])
        after = dist_at_wp > slope_s1[:, None]

        # Sample slope angle within ±max_slope_deg.
        max_rad = torch.deg2rad(torch.full((len(env_ids),), float(sp.max_slope_deg), device=self.device))
        angle = torch.empty(len(env_ids), device=self.device).uniform_(-1.0, 1.0) * max_rad
        slope = torch.tan(angle)

        # Linear ramp within segment, then keep plateau height.
        seg_len = (slope_s1 - slope_s0).clamp(min=1e-3)
        t = ((dist_at_wp - slope_s0[:, None]) / seg_len[:, None]).clamp(0.0, 1.0)
        z_ramp = slope[:, None] * (t * seg_len[:, None])
        z_plateau = slope[:, None] * seg_len[:, None]
        z_off = torch.where(inside, z_ramp, torch.where(after, z_plateau, torch.zeros_like(z_ramp)))
        pos[..., 2] = pos[..., 2] + z_off

        yaw0 = self._planned_yaw[env_ids]
        # Keep yaw fixed for non-walk skills.
        yaw_wp = self._build_fixed_yaw_traj(yaw0)
        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)

        lead_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(lead_params, "v_ref", 1.0)
        self._append_segment(env_ids, self.SKILL_ID["walk"], zero, slope_s0, params=lead_params)

        climb_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(climb_params, "v_ref", 0.8)
        self._set_seg_param(climb_params, "slope_angle_ref", angle)
        self._append_segment(env_ids, self.SKILL_ID["climb"], slope_s0, slope_s1, params=climb_params)

        trail_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(trail_params, "v_ref", 1.0)
        self._append_segment(env_ids, self.SKILL_ID["walk"], slope_s1, total_len, params=trail_params)

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
