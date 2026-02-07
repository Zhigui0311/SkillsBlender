from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.utils import configclass

from .path_command_cfg import PathCommandCfg
from .base_path_command import SegmentPathCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class VirtualJumpPathCommandCfg(PathCommandCfg):
    """Config for virtual skill hallucination command generation on flat terrain."""

    # Whether to inject a virtual segment for this resample.
    virtual_prob: float = 0.5

    jump_prob: float = 0.02
    jump_height_range: Tuple[float, float] = (0.3, 0.5)
    # Length of the virtual segment (gap width / crouch length / stairs run).
    gap_width_range: Tuple[float, float] = (0.8, 1.2)
    jump_length_range: Tuple[float, float] = (0.8, 1.2)
    # Cooldown in seconds between consecutive virtual jumps.
    cool_down: float = 0.8
    # Reusable across skills by changing skill_name/profile_type.
    skill_name: str = "jump"
    # auto -> infer from skill_name; or explicit:
    # none/parabola_up/parabola_down/flat_down/ramp_up/stairs/stairs_up/stairs_down
    profile_type: str = "auto"
    stairs_steps_range: Tuple[int, int] = (3, 6)
    # Trajectory generation (curved paths removed)
    trajectory_mode: str = "line"  # no-op (always line)
    yaw_mode: str = "fixed"  # "fixed" or "interp"
    # Yaw range in degrees relative to start heading (used when yaw_mode="interp").
    yaw_range: Tuple[float, float] = (-45.0, 45.0)
    lateral_range: Tuple[float, float] = (-0.6, 0.6)  # no-op

    def __post_init__(self):
        # Backward-compat: use jump_prob if virtual_prob is not explicitly set.
        if getattr(self, "jump_prob", 0.0) and getattr(self, "virtual_prob", None) is None:
            self.virtual_prob = float(self.jump_prob)
        super().__post_init__()
        if getattr(self, "class_type", None) in (None, MISSING):
            self.class_type = VirtualJumpPathCommand


class VirtualJumpPathCommand(SegmentPathCommand):
    """Purely mathematical jump-hallucination command.

    This command NEVER queries terrain geometry. It generates flat XY trajectories
    and injects virtual Z parabolas according to an Idle/Jumping state-machine.
    """

    STATE_IDLE = 0
    STATE_JUMPING = 1

    cfg: VirtualJumpPathCommandCfg

    def __init__(self, cfg: VirtualJumpPathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Jump state-machine buffers.
        self._state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cooldown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._jump_start_dist = torch.zeros(self.num_envs, device=self.device)
        self._jump_end_dist = torch.zeros(self.num_envs, device=self.device)
        self._jump_height = torch.zeros(self.num_envs, device=self.device)
        self._stairs_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Convert configured cooldown (seconds) to control steps.
        dt = float(getattr(env, "step_dt", 0.02))
        self._cool_down_steps = max(1, int(round(float(cfg.cool_down) / max(dt, 1.0e-6))))

    @property
    def is_in_jump_phase(self) -> torch.Tensor:
        s = self._dist_along_planned()
        return (self._state == self.STATE_JUMPING) & (s >= self._jump_start_dist) & (s <= self._jump_end_dist)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._state[env_ids] = self.STATE_IDLE
        self._cooldown[env_ids] = 0
        self._jump_start_dist[env_ids] = 0.0
        self._jump_end_dist[env_ids] = 0.0
        self._jump_height[env_ids] = 0.0
        self._stairs_steps[env_ids] = 0
        self._on_reset(env_ids)
        return super().reset(env_ids)

    def _sample_uniform(self, n: int, low: float, high: float) -> torch.Tensor:
        lo = float(min(low, high))
        hi = float(max(low, high))
        if hi - lo < 1.0e-6:
            return torch.full((n,), lo, device=self.device)
        return torch.empty((n,), device=self.device).uniform_(lo, hi)

    def _sample_steps(self, n: int, low: int, high: int) -> torch.Tensor:
        lo = int(min(low, high))
        hi = int(max(low, high))
        if hi <= lo:
            return torch.full((n,), max(lo, 1), dtype=torch.long, device=self.device)
        return torch.randint(low=max(lo, 1), high=hi + 1, size=(n,), device=self.device)

    def _sample_path_length(self, n: int) -> torch.Tensor:
        default_len = float(self.cfg.ranges.default_path_len)
        total_len = torch.full((n,), default_len, device=self.device)
        sampling = self.cfg.sampling
        if getattr(sampling, "sample_goal_distance", False):
            lo, hi, _ = sampling.end_to_start_pos
            total_len = self._sample_uniform(n, lo, hi)
        return torch.clamp(total_len, min=1.5)

    def _resolve_profile_type(self) -> str:
        p = str(getattr(self.cfg, "profile_type", "auto"))
        if p != "auto":
            return p
        s = str(getattr(self.cfg, "skill_name", "jump"))
        if s == "jump":
            return "parabola_up"
        if s == "crouch":
            return "flat_down"
        if s == "platform":
            return "ramp_up"
        if s == "climb":
            return "none"
        if s == "stairs":
            return "stairs"
        if s == "stairs_up":
            return "stairs_up"
        if s == "stairs_down":
            return "stairs_down"
        return "none"

    def _generate_base_trajectory(
        self,
        env_ids: torch.Tensor,
        total_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a base trajectory (straight line) before virtual profile injection."""
        n = len(env_ids)
        if total_len is None:
            total_len = self._sample_path_length(n)
        start = self._planned_start_pos[env_ids].clone()
        yaw0 = self._planned_yaw[env_ids]
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = start[:, :2] + fwd * total_len[:, None]
        alpha = self.t_alpha.view(1, -1, 1)
        pos_traj = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha
        # Keep yaw fixed for virtual non-walk skills.
        if str(getattr(self.cfg, "skill_name", "jump")) == "walk":
            sampling = getattr(self.cfg, "sampling", None)
            sampling_mode = str(getattr(sampling, "yaw_mode", "fixed")).lower()
            if sampling_mode in ("interp", "linear", "lerp"):
                yaw_goal = self._sample_goal_yaw(yaw0)
                yaw_traj = self._build_yaw_traj(yaw0, yaw_goal)
            else:
                yaw_traj = yaw0[:, None].repeat(1, self.num_waypoints)
        else:
            yaw_traj = yaw0[:, None].repeat(1, self.num_waypoints)
        return pos_traj, yaw_traj, total_len

    def _apply_profile(
        self,
        env_ids: torch.Tensor,
        pos_traj: torch.Tensor,
        total_len: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the virtual profile to the base trajectory."""
        jumping = self._state[env_ids] == self.STATE_JUMPING
        if not torch.any(jumping):
            return pos_traj

        s0 = self._jump_start_dist[env_ids]
        s1 = self._jump_end_dist[env_ids]
        h = self._jump_height[env_ids]

        dist_at_wp = self.t_alpha.view(1, -1) * total_len[:, None]
        den = (s1 - s0).clamp(min=1.0e-3)
        t = ((dist_at_wp - s0[:, None]) / den[:, None]).clamp(0.0, 1.0)
        profile_type = self._resolve_profile_type()
        if profile_type == "parabola_up":
            z_offset = 4.0 * h[:, None] * t * (1.0 - t)
        elif profile_type == "parabola_down":
            z_offset = -4.0 * h[:, None] * t * (1.0 - t)
        elif profile_type == "smooth_down":
            # Smooth down-up: smoothstep in/out with a flat plateau.
            alpha = 0.2
            t_up = torch.clamp(t / alpha, 0.0, 1.0)
            t_dn = torch.clamp((1.0 - t) / alpha, 0.0, 1.0)
            smooth_up = t_up * t_up * (3.0 - 2.0 * t_up)
            smooth_dn = t_dn * t_dn * (3.0 - 2.0 * t_dn)
            weight = smooth_up * smooth_dn
            z_offset = -h[:, None] * weight
        elif profile_type == "flat_down":
            z_offset = -h[:, None] * ((t >= 0.0) & (t <= 1.0)).to(torch.float32)
        elif profile_type == "ramp_up":
            smooth = 3.0 * t * t - 2.0 * t * t * t
            z_offset = h[:, None] * smooth
        elif profile_type in ("stairs", "stairs_up", "stairs_down"):
            n_steps = torch.clamp(self._stairs_steps[env_ids], min=1).to(torch.float32)
            step_idx = torch.floor(t * n_steps[:, None]).clamp(min=0.0)
            stair = (step_idx / n_steps[:, None]).clamp(max=1.0)
            z_offset = h[:, None] * stair
            if profile_type == "stairs":
                stairs_dir = getattr(self, "_stairs_dir", None)
                if stairs_dir is not None:
                    dir_sign = torch.sign(stairs_dir[env_ids]).clamp(min=-1.0, max=1.0)
                    z_offset = z_offset * dir_sign[:, None]
            elif profile_type == "stairs_down":
                z_offset = -z_offset
        else:
            z_offset = torch.zeros_like(t)
        mask = jumping[:, None] & (dist_at_wp >= s0[:, None]) & (dist_at_wp <= s1[:, None])
        pos_traj[..., 2] = pos_traj[..., 2] + torch.where(mask, z_offset, torch.zeros_like(z_offset))
        return pos_traj

    def _sample_segment_bounds(
        self,
        total_len: torch.Tensor,
        seg_len: torch.Tensor,
        start_ratio_min: float = 0.2,
        start_ratio_max: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample segment start/end distances within [start_ratio_min, start_ratio_max] of path length."""
        min_start = total_len * float(start_ratio_min)
        max_start = total_len * float(start_ratio_max)
        # Keep some room after the segment.
        max_start = torch.minimum(max_start, total_len - seg_len - 0.1 * total_len)
        max_start = torch.maximum(max_start, min_start)
        u = torch.rand_like(total_len)
        s0 = min_start + (max_start - min_start) * u
        s1 = s0 + seg_len
        return s0, s1

    def _resample_command(self, env_ids: torch.Tensor):
        # Keep remaining jump distance before resetting the planned frame.
        prev_s = self._dist_along_planned()[env_ids]
        remaining_s0 = self._jump_start_dist[env_ids] - prev_s
        remaining_s1 = self._jump_end_dist[env_ids] - prev_s

        self._set_planned_frame_from_robot(env_ids)

        # Sample base path length once for consistency.
        total_len = self._sample_path_length(len(env_ids))

        # Determine which envs trigger a virtual segment.
        active_prev = self._state[env_ids] == self.STATE_JUMPING
        prob = float(getattr(self.cfg, "virtual_prob", 0.0))
        do_trigger = active_prev | (torch.rand(len(env_ids), device=self.device) < prob)

        # Preserve ongoing segments across resampling.
        if torch.any(active_prev):
            self._jump_start_dist[env_ids] = torch.where(
                active_prev,
                torch.clamp(remaining_s0, min=0.0),
                self._jump_start_dist[env_ids],
            )
            self._jump_end_dist[env_ids] = torch.where(
                active_prev,
                torch.clamp(remaining_s1, min=0.08),
                self._jump_end_dist[env_ids],
            )

        # Initialize non-trigger envs to pure walk.
        if torch.any(~do_trigger):
            env_nt = env_ids[~do_trigger]
            self._state[env_nt] = self.STATE_IDLE
            self._jump_start_dist[env_nt] = 0.0
            self._jump_end_dist[env_nt] = 0.0
            self._jump_height[env_nt] = 0.0
            self._stairs_steps[env_nt] = 0
            if hasattr(self, "pitch_target"):
                self.pitch_target[env_nt] = 0.0

        # Sample new virtual segment for fresh triggers.
        new_trigger = do_trigger & (~active_prev)
        if torch.any(new_trigger):
            env_t = env_ids[new_trigger]
            n = len(env_t)
            h = self._sample_uniform(n, *self.cfg.jump_height_range)
            seg_len = self._sample_uniform(n, *self.cfg.gap_width_range)
            seg_len = torch.minimum(seg_len, total_len[new_trigger] * 0.6)
            seg_len = torch.clamp(seg_len, min=0.2)
            stairs_steps = self._sample_steps(n, *self.cfg.stairs_steps_range)

            s0, s1 = self._sample_segment_bounds(total_len[new_trigger], seg_len)
            self._state[env_t] = self.STATE_JUMPING
            self._jump_start_dist[env_t] = s0
            self._jump_end_dist[env_t] = torch.maximum(s1, s0 + 0.1)
            self._jump_height[env_t] = h
            self._stairs_steps[env_t] = stairs_steps
            self._on_trigger(env_t, s0, s1, h, stairs_steps)

        pos_traj, yaw_traj, _ = self._generate_base_trajectory(env_ids, total_len=total_len)
        pos_traj = self._apply_profile(env_ids, pos_traj, total_len)

        self._path_len[env_ids] = total_len
        self.pos_path_w[env_ids] = pos_traj
        self.heading_path_w[env_ids, :, 0] = yaw_traj

        # Build segment table for gating/reward compatibility.
        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)
        jump_active = self._state[env_ids] == self.STATE_JUMPING
        jump_s0 = self._jump_start_dist[env_ids]
        jump_s1 = self._jump_end_dist[env_ids]

        # Non-trigger: pure walk path.
        if torch.any(~jump_active):
            env_nt = env_ids[~jump_active]
            params = self._new_seg_params(len(env_nt))
            self._set_seg_param(params, "v_ref", 1.0)
            self._append_segment(env_nt, self.SKILL_WALK, zero[~jump_active], total_len[~jump_active], params=params)

        # Triggered: walk -> skill -> walk.
        if torch.any(jump_active):
            env_t = env_ids[jump_active]
            lead_params = self._new_seg_params(len(env_t))
            self._set_seg_param(lead_params, "v_ref", 1.0)
            self._append_segment(
                env_t,
                self.SKILL_WALK,
                zero[jump_active],
                jump_s0[jump_active],
                params=lead_params,
            )

            jump_skill_id = int(self.SKILL_ID.get(self.cfg.skill_name, self.SKILL_JUMP))
            jump_params = self._new_seg_params(len(env_t))
            self._set_seg_param(jump_params, "v_ref", 1.25)
            self._set_seg_param(jump_params, "jump_height_ref", self._jump_height[env_t])
            if self.cfg.skill_name == "crouch":
                self._set_seg_param(jump_params, "base_height_ref", 0.24)
                self._set_seg_param(jump_params, "crouch_height_ref", -self._jump_height[env_t])
            if self.cfg.skill_name in ("stairs_up", "stairs_down"):
                self._set_seg_param(jump_params, "clearance_ref", 0.16)
                self._set_seg_param(jump_params, "step_height_ref", self._jump_height[env_t])
                self._set_seg_param(jump_params, "misc", self._stairs_steps[env_t].to(torch.float32))
            if self.cfg.skill_name == "platform":
                self._set_seg_param(jump_params, "clearance_ref", 0.2)
                slope = torch.zeros(len(env_t), device=self.device)
                if hasattr(self, "pitch_target"):
                    slope = self.pitch_target[env_t]
                self._set_seg_param(jump_params, "slope_angle_ref", slope)
            if self.cfg.skill_name == "climb":
                slope = torch.zeros(len(env_t), device=self.device)
                if hasattr(self, "pitch_target"):
                    slope = self.pitch_target[env_t]
                self._set_seg_param(jump_params, "slope_angle_ref", slope)

            self._append_segment(
                env_t,
                jump_skill_id,
                jump_s0[jump_active],
                jump_s1[jump_active],
                params=jump_params,
            )

            trail_params = self._new_seg_params(len(env_t))
            self._set_seg_param(trail_params, "v_ref", 1.0)
            self._append_segment(
                env_t,
                self.SKILL_WALK,
                jump_s1[jump_active],
                total_len[jump_active],
                params=trail_params,
            )

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False

    def _update_command(self):
        # Jump completion detection.
        s = self._dist_along_planned()
        jumping = self._state == self.STATE_JUMPING
        finished = jumping & (s >= self._jump_end_dist)
        if torch.any(finished):
            self._state[finished] = self.STATE_IDLE
            self._jump_start_dist[finished] = 0.0
            self._jump_end_dist[finished] = 0.0
            self._jump_height[finished] = 0.0
            self._stairs_steps[finished] = 0
            self._on_finish(finished)

    def _on_trigger(
        self,
        env_ids: torch.Tensor,
        jump_start: torch.Tensor,
        jump_end: torch.Tensor,
        jump_height: torch.Tensor,
        stairs_steps: torch.Tensor,
    ):
        """Hook for subclasses to capture per-trigger state."""
        return

    def _on_finish(self, env_ids: torch.Tensor):
        """Hook for subclasses to clear per-env state when virtual segment ends."""
        return

    def _on_reset(self, env_ids: torch.Tensor):
        """Hook for subclasses to clear per-env state on reset."""
        return
