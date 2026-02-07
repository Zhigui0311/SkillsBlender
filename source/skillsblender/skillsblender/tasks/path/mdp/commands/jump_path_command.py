from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import wrap_to_pi

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
        self._jump_approach_start_dist = torch.zeros(self.num_envs, device=self.device)

    @property
    def is_in_jump_phase(self) -> torch.Tensor:
        s = self._dist_along_planned()
        return self._has_gap & (s >= self._jump_start_dist) & (s <= self._jump_end_dist)

    @property
    def is_in_jump_approach_phase(self) -> torch.Tensor:
        s = self._dist_along_planned()
        return self._has_gap & (s >= self._jump_approach_start_dist) & (s < self._jump_start_dist)

    def _resample_command(self, env_ids: torch.Tensor):
        # planned frame
        self._set_planned_frame_from_robot(env_ids)
        jp = self.cfg.jump_params

        # heading offset
        off_min, off_max = jp.heading_offset_range
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

        # sample a goal distance within env bounds (path end is inside bounds)
        start_pos = self._planned_start_pos[env_ids].clone()
        min_len = float(jp.min_jump_start_dist + jp.min_gap_width + jp.min_landing_runout + jp.endpoint_extension_min)
        total_len = self._sample_path_length(env_ids, start_pos, yaw, min_len=min_len)
        env_bounds = self._build_env_bounds(start_pos)
        max_len = self._compute_max_path_length(start_pos, yaw, env_bounds)

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

            valid = (x_fwd >= 0.0) & (x_fwd <= jp.scan_dist) & (torch.abs(y_lat) <= jp.scan_width)

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
                is_gap = (zv - ref_h) < jp.gap_threshold
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

        # Fallback: when scanner doesn't find a gap, synthesize one in the middle.
        # This keeps jump-skill training from degenerating into pure straight walking.
        missing = ~has_gap
        if torch.any(missing):
            mid = float(jp.fallback_gap_center_ratio) * total_len[missing]
            gap_w = torch.clamp(0.18 * total_len[missing], min=jp.min_gap_width, max=jp.max_gap_width)
            s0 = torch.clamp(mid - 0.5 * gap_w, min=jp.min_gap_start_dist)
            s1 = s0 + gap_w
            max_gap_end = total_len[missing] - max(float(jp.min_landing_runout), 0.6)
            s1 = torch.minimum(s1, max_gap_end)
            s0 = torch.minimum(s0, s1 - jp.min_gap_width)
            gap_s0[missing] = s0
            gap_s1[missing] = s1
            has_gap[missing] = True

        # sanitize detected/synthetic gaps so jump always has a valid takeoff window.
        if torch.any(has_gap):
            hs0 = gap_s0[has_gap]
            hw = (gap_s1[has_gap] - hs0).clamp(min=jp.min_gap_width, max=jp.max_gap_width)
            hs0 = torch.clamp(hs0, min=jp.min_gap_start_dist)
            hs1 = hs0 + hw
            max_gap_end = total_len[has_gap] - max(float(jp.min_landing_runout), 0.6)
            hs1 = torch.minimum(hs1, max_gap_end)
            hs0 = torch.minimum(hs0, hs1 - jp.min_gap_width)
            gap_s0[has_gap] = hs0
            gap_s1[has_gap] = hs1

        # margins & arc parameters
        gap_w = (gap_s1 - gap_s0).clamp(min=0.0)
        if jp.takeoff_margin is None:
            takeoff = torch.clamp(0.2 + 0.4 * gap_w, min=jp.takeoff_margin_min, max=jp.takeoff_margin_max)
        else:
            takeoff = torch.full_like(gap_w, float(jp.takeoff_margin))

        if jp.landing_margin is None:
            landing = torch.clamp(0.25 + 0.3 * gap_w, min=jp.landing_margin_min, max=jp.landing_margin_max)
        else:
            landing = torch.full_like(gap_w, float(jp.landing_margin))

        extension = torch.clamp(1.0 + 0.5 * gap_w, min=jp.endpoint_extension_min, max=jp.endpoint_extension_max)

        if jp.jump_height is None:
            jump_h = torch.clamp(0.3 + 0.6 * gap_w, min=jp.jump_height_min, max=jp.jump_height_max)
        else:
            jump_h = torch.full_like(gap_w, float(jp.jump_height))

        # enforce a minimum approach distance before takeoff to avoid stepping into the pit.
        pre_jump_start = gap_s0 - takeoff
        shift = torch.clamp(float(jp.min_jump_start_dist) - pre_jump_start, min=0.0)
        gap_s0 = torch.where(has_gap, gap_s0 + shift, gap_s0)
        gap_s1 = torch.where(has_gap, gap_s1 + shift, gap_s1)

        jump_s0 = torch.where(has_gap, torch.maximum(gap_s0 - takeoff, torch.full_like(gap_s0, float(jp.min_jump_start_dist))), gap_s0)
        jump_s1 = torch.where(has_gap, gap_s1 + landing, gap_s1)

        # ensure enough landing runout, but keep the goal inside env bounds
        desired_len = torch.maximum(jump_s1 + extension, jump_s1 + float(jp.min_landing_runout))
        if self.cfg.jump_params.post_jump_distance > 0:
            desired_len = desired_len + float(self.cfg.jump_params.post_jump_distance)
        total_len = torch.maximum(total_len, desired_len)
        total_len = torch.minimum(total_len, max_len)
        # re-clamp jump window if the path got shortened
        jump_s1 = torch.minimum(jump_s1, total_len - 1e-3)
        jump_s0 = torch.minimum(jump_s0, jump_s1 - jp.min_gap_width)
        jump_s0 = torch.maximum(jump_s0, torch.full_like(jump_s0, float(jp.min_jump_start_dist)))

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

        # yaw along path: fixed or interpolated based on sampling.yaw_mode
        yaw0 = self._planned_yaw[env_ids]
        # Keep yaw fixed for non-walk skills.
        yaw_wp = self._build_fixed_yaw_traj(yaw0)

        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        # segment table
        self._clear_segments(env_ids)

        s_zero = torch.zeros(len(env_ids), device=self.device)

        # leading walk
        lead_s0 = s_zero
        lead_s1 = torch.where(has_gap, jump_s0, total_len)
        lead_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(lead_params, "v_ref", 1.25)
        self._append_segment(env_ids, self.SKILL_WALK, lead_s0, lead_s1, params=lead_params)

        # jump segment
        j_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(j_params, "v_ref", 1.45)
        self._set_seg_param(j_params, "jump_height_ref", jump_h)
        self._append_segment(env_ids, self.SKILL_JUMP, jump_s0, jump_s1, params=j_params)

        # trailing walk
        trail_s0 = torch.where(has_gap, jump_s1, total_len)
        trail_s1 = total_len
        trail_params = self._new_seg_params(len(env_ids))
        self._set_seg_param(trail_params, "v_ref", 0.95)
        self._append_segment(env_ids, self.SKILL_WALK, trail_s0, trail_s1, params=trail_params)

        # ensure at least one segment (no-gap case -> lead walk kept)
        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)

        # store jump buffers for external use
        self._has_gap[env_ids] = has_gap
        self._jump_approach_start_dist[env_ids] = torch.clamp(jump_s0 - float(jp.approach_phase_window), min=0.0)
        self._jump_start_dist[env_ids] = jump_s0
        self._jump_end_dist[env_ids] = jump_s1

        # reset trackers
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
