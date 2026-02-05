from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ClimbPathCommand(SegmentPathCommand):
    """Path with a CLIMB segment (step-like rise + top traversal)."""

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    @property
    def is_in_climb_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("climb")

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        cp = self.cfg.climb_params
        total_len = torch.full((len(env_ids),), float(self.cfg.ranges.default_path_len), device=self.device)

        s_min, s_max = cp.start_dist_range
        climb_s0 = torch.empty(len(env_ids), device=self.device).uniform_(s_min, s_max)
        climb_s1 = (climb_s0 + float(cp.climb_len)).clamp(max=total_len - 1e-3)

        self._path_len[env_ids] = total_len

        start = self._planned_start_pos[env_ids].clone()
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha

        dist_at_wp = alpha.squeeze(-1) * total_len[:, None]
        inside = (dist_at_wp >= climb_s0[:, None]) & (dist_at_wp <= climb_s1[:, None])
        t = ((dist_at_wp - climb_s0[:, None]) / (climb_s1[:, None] - climb_s0[:, None]).clamp(min=1e-3)).clamp(0.0, 1.0)

        # Step-like rise: climb most of the height early, then keep a top plateau.
        rise_ratio = 0.35
        t_rise = (t / rise_ratio).clamp(0.0, 1.0)
        smooth = 3.0 * t_rise * t_rise - 2.0 * t_rise * t_rise * t_rise
        z_rise = float(cp.climb_height) * smooth
        z_off = torch.where(t <= rise_ratio, z_rise, torch.full_like(z_rise, float(cp.climb_height)))
        pos[..., 2] = pos[..., 2] + torch.where(inside, z_off, torch.zeros_like(z_off))

        yaw_wp = self._planned_yaw[env_ids][:, None].repeat(1, self.num_waypoints)
        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)

        lead_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(lead_params, "v_ref", 1.0)
        self._append_segment(env_ids, self.SKILL_ID["walk"], zero, climb_s0, params=lead_params)

        cl_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(cl_params, "v_ref", 0.65)
        self._set_seg_param(cl_params, "clearance_ref", 0.18)
        self._set_seg_param(cl_params, "misc", float(cp.climb_height))
        self._append_segment(env_ids, self.SKILL_ID["climb"], climb_s0, climb_s1, params=cl_params)

        trail_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(trail_params, "v_ref", 1.0)
        self._append_segment(env_ids, self.SKILL_ID["walk"], climb_s1, total_len, params=trail_params)

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
