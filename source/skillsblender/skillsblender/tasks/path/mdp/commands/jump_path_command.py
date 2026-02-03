from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class JumpPathCommand(SegmentPathCommand):
    """Path with an optional JUMP segment discovered by scanner (privileged) or fallback raycast.

    Segment sequence is variable:
      - If no gap: one WALK segment
      - If gap: may have [WALK] + JUMP + [WALK], empty leading/trailing WALK segments are skipped
    """

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.heading_offset = torch.zeros(self.num_envs, device=self.device)

        # optional height scanner (RayCaster term in scene)
        self.height_scanner = None
        if hasattr(env.scene, "sensors") and isinstance(getattr(env.scene, "sensors", None), dict):
            # user may attach a sensor named "height_scanner"
            self.height_scanner = env.scene.sensors.get("height_scanner", None)

        # buffers for debug/termination gating
        self._has_gap = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._jump_start_dist = torch.zeros(self.num_envs, device=self.device)
        self._jump_end_dist = torch.zeros(self.num_envs, device=self.device)

    @property
    def is_in_jump_phase(self) -> torch.Tensor:
        s = self._dist_along_planned()
        return self._has_gap & (s >= self._jump_start_dist) & (s <= self._jump_end_dist)

    def _resample_command(self, env_ids: torch.Tensor):
        # planned frame
        self._set_planned_frame_from_robot(env_ids)

        # heading offset
        off_min, off_max = self.cfg.jump_params.heading_offset_range
        if off_min == off_max:
            self.heading_offset[env_ids] = off_min
        else:
            self.heading_offset[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(off_min, off_max)

        # apply offset to planned yaw and axes for this episode
        self._planned_yaw[env_ids] = wrap_to_pi(self._planned_yaw[env_ids] + self.heading_offset[env_ids])
        yaw = self._planned_yaw[env_ids]
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        left = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
        self._planned_forward_dir[env_ids] = fwd
        self._planned_left_dir[env_ids] = left

        # detect gap (distance along forward)
        gap_s0 = torch.zeros(len(env_ids), device=self.device)
        gap_s1 = torch.zeros(len(env_ids), device=self.device)
        has_gap = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        # simple scanner-based detection if available
        if self.height_scanner is not None and hasattr(self.height_scanner, "data"):
            ray_hits_w = self.height_scanner.data.ray_hits_w[env_ids]  # (B,R,3)
            robot_pos = self._planned_start_pos[env_ids][:, :3]
            rel = ray_hits_w - robot_pos[:, None, :]
            # project to planned frame
            cy = torch.cos(yaw)[:, None]
            sy = torch.sin(yaw)[:, None]
            dx = rel[..., 0]
            dy = rel[..., 1]
            x_fwd = cy * dx + sy * dy
            y_lat = -sy * dx + cy * dy
            z_w = ray_hits_w[..., 2]

            valid = (x_fwd >= 0.0) & (x_fwd <= self.cfg.jump_params.scan_dist) & (torch.abs(y_lat) <= self.cfg.jump_params.scan_width)

            for bi in range(len(env_ids)):
                m = valid[bi]
                if not torch.any(m):
                    continue
                xv = x_fwd[bi, m]
                zv = z_w[bi, m]
                order = torch.argsort(xv)
                xv = xv[order]
                zv = zv[order]
                # reference height from near samples
                ref_window = min(7, zv.numel())
                ref_samples = zv[:ref_window]
                ref_k = max(1, ref_window // 2)
                ref_h = torch.topk(ref_samples, k=ref_k).values.median()
                is_gap = (zv - ref_h) < self.cfg.jump_params.gap_threshold
                if not torch.any(is_gap):
                    continue
                gi = torch.nonzero(is_gap, as_tuple=False).squeeze(-1)
                first = gi[0]
                # contiguous block
                breaks = torch.nonzero(torch.diff(gi) > 1, as_tuple=False).squeeze(-1)
                last = gi[breaks[0]] if breaks.numel() > 0 else gi[-1]
                gap_s0[bi] = xv[first]
                gap_s1[bi] = xv[last]
                has_gap[bi] = True

        # margins & arc parameters
        gap_w = (gap_s1 - gap_s0).clamp(min=0.0)
        takeoff = torch.clamp(0.2 + 0.4 * gap_w, min=self.cfg.jump_params.takeoff_margin_min, max=self.cfg.jump_params.takeoff_margin_max)
        landing = torch.clamp(0.25 + 0.3 * gap_w, min=self.cfg.jump_params.landing_margin_min, max=self.cfg.jump_params.landing_margin_max)
        extension = torch.clamp(1.0 + 0.5 * gap_w, min=self.cfg.jump_params.endpoint_extension_min, max=self.cfg.jump_params.endpoint_extension_max)
        jump_h = torch.clamp(0.3 + 0.6 * gap_w, min=self.cfg.jump_params.jump_height_min, max=self.cfg.jump_params.jump_height_max)

        jump_s0 = torch.where(has_gap, gap_s0 - takeoff, gap_s0)
        jump_s1 = torch.where(has_gap, gap_s1 + landing, gap_s1)

        # total path length
        total_len = torch.where(has_gap & (jump_s1 > 0.0), jump_s1 + extension, torch.full_like(jump_s1, float(self.cfg.ranges.default_path_len)))
        if self.cfg.jump_params.post_jump_distance > 0:
            total_len = torch.where(has_gap & (jump_s1 > 0.0), total_len + self.cfg.jump_params.post_jump_distance, total_len)

        self._path_len[env_ids] = total_len

        # start/end in world
        start = self._planned_start_pos[env_ids].clone()
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        # base linear trajectory
        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha

        # apply parabolic arc over jump segment
        dist_at_wp = alpha.squeeze(-1) * total_len[:, None]
        den = (jump_s1[:, None] - jump_s0[:, None]).clamp(min=1e-3)
        t = ((dist_at_wp - jump_s0[:, None]) / den).clamp(0.0, 1.0)
        arc = jump_h[:, None] * 4.0 * t * (1.0 - t)

        mask = (dist_at_wp >= jump_s0[:, None]) & (dist_at_wp <= jump_s1[:, None]) & has_gap[:, None]
        pos[..., 2] = pos[..., 2] + torch.where(mask, arc, torch.zeros_like(arc))

        # yaw along path: keep planned yaw constant here (can be upgraded to curvature)
        yaw_wp = self._planned_yaw[env_ids][:, None].repeat(1, self.num_waypoints)

        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        # segment table
        self._clear_segments(env_ids)

        s_zero = torch.zeros(len(env_ids), device=self.device)

        # leading walk
        lead_s0 = s_zero
        lead_s1 = torch.where(has_gap, jump_s0, total_len)
        lead_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        lead_params[:, 0] = 1.0  # v_ref
        self._append_segment(env_ids, self.SKILL_WALK, lead_s0, lead_s1, params=lead_params)

        # jump segment
        j_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        j_params[:, 0] = 1.0
        j_params[:, 4] = jump_h  # jump_height_ref slot
        self._append_segment(env_ids, self.SKILL_JUMP, jump_s0, jump_s1, params=j_params)

        # trailing walk
        trail_s0 = torch.where(has_gap, jump_s1, total_len)
        trail_s1 = total_len
        trail_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        trail_params[:, 0] = 1.0
        self._append_segment(env_ids, self.SKILL_WALK, trail_s0, trail_s1, params=trail_params)

        # ensure at least one segment (no-gap case -> lead walk kept)
        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)

        # store jump buffers for external use
        self._has_gap[env_ids] = has_gap
        self._jump_start_dist[env_ids] = jump_s0
        self._jump_end_dist[env_ids] = jump_s1

        # reset trackers
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False