from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SegmentPathCommand(CommandTerm):
    """Unified path-slice command with a segment table + skill registry.

    Skill ids are centrally registered via SKILL_NAMES. Add a new skill by
    appending a string to SKILL_NAMES (and updating cfg/rules accordingly).
    """

    # -------- skill registry (single source of truth) --------
    SKILL_NAMES = [
        "walk",
        "jump",
        "stairs_up",
        "stairs_down",
        "crouch",
        "platform",  # high platform climb
        "climb",     # ramp/slope traversal
        "sidestep",  # 占位技能
    ]
    SKILL_ID = {name: i for i, name in enumerate(SKILL_NAMES)}
    NUM_SKILLS = len(SKILL_NAMES)
    SKILL_WALK = SKILL_ID["walk"]
    SKILL_JUMP = SKILL_ID["jump"]
    SKILL_STAIRS_UP = SKILL_ID["stairs_up"]
    SKILL_STAIRS_DOWN = SKILL_ID["stairs_down"]
    SKILL_CROUCH = SKILL_ID["crouch"]
    SKILL_SIDESTEP = SKILL_ID["sidestep"]
    SKILL_PLATFORM = SKILL_ID["platform"]
    SKILL_CLIMB = SKILL_ID["climb"]

    # -------- segment parameter slots (single source of truth) --------
    SEG_PARAM = {
        "v_ref": 0,
        "yaw_rate_ref": 1,
        "base_height_ref": 2,
        "clearance_ref": 3,
        "jump_height_ref": 4,
        "misc": 5,
        # aliases used by virtual-skill commands
        "crouch_height_ref": 2,
        "step_height_ref": 3,
        "slope_angle_ref": 5,
    }

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # optional assert for config mismatch
        if cfg.ranges.num_skills is not None:
            assert int(cfg.ranges.num_skills) == int(self.NUM_SKILLS), (
                f"cfg.ranges.num_skills={cfg.ranges.num_skills} "
                f"but registry has {self.NUM_SKILLS} skills: {self.SKILL_NAMES}"
            )

        # sizes
        self.num_waypoints = cfg.ranges.num_waypoints
        self.num_lookahead_waypoints = cfg.ranges.num_lookahead_waypoints
        self.max_segments = cfg.ranges.max_segments
        self.num_skills = self.NUM_SKILLS
        self.num_seg_params = max(int(cfg.ranges.num_seg_params), 0)
        self.has_seg_params = self.num_seg_params > 0

        self.t_alpha = torch.linspace(0, 1, self.num_waypoints, device=self.device)

        # planned-frame buffers
        self._planned_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._planned_yaw = torch.zeros(self.num_envs, device=self.device)
        self._planned_forward_dir = torch.zeros(self.num_envs, 2, device=self.device)  # world XY unit
        self._planned_left_dir = torch.zeros(self.num_envs, 2, device=self.device)     # world XY unit
        self._path_len = torch.ones(self.num_envs, device=self.device) * 1e-3

        # segment table
        self._num_segs = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._seg_skill = torch.zeros(self.num_envs, self.max_segments, dtype=torch.long, device=self.device)
        self._seg_s0 = torch.zeros(self.num_envs, self.max_segments, device=self.device)
        self._seg_s1 = torch.zeros(self.num_envs, self.max_segments, device=self.device)
        self._seg_params = (
            torch.zeros(self.num_envs, self.max_segments, self.num_seg_params, device=self.device)
            if self.has_seg_params
            else None
        )

        # current segment state (derived each step)
        self._cur_seg_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # path buffers (world)
        self.pos_path_w = torch.zeros(self.num_envs, self.num_waypoints, 3, device=self.device)
        self.heading_path_w = torch.zeros(self.num_envs, self.num_waypoints, 1, device=self.device)

        # trackers
        self.current_waypoints_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics (compat with your existing rewards/terminations)
        self.metrics: dict[str, torch.Tensor] = {
            "error_pos_xy": torch.zeros(self.num_envs, device=self.device),
            "error_pos_z": torch.zeros(self.num_envs, device=self.device),
            "error_heading": torch.zeros(self.num_envs, device=self.device),
        }

        # observation outputs
        self.obs_slices = torch.zeros(self.num_envs, self.num_lookahead_waypoints, 4, device=self.device)
        self.obs_meta = torch.zeros(self.num_envs, self._meta_dim(), device=self.device)

        # goal-hold state (freeze command after reaching goal)
        self._goal_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _new_seg_params(self, batch_size: int) -> torch.Tensor | None:
        """Create per-segment param tensor; returns None when params are disabled."""
        if not self.has_seg_params:
            return None
        return torch.zeros(batch_size, self.num_seg_params, device=self.device)

    def _new_seg_param_vec(self) -> torch.Tensor | None:
        """Create a single segment param vector; returns None when params are disabled."""
        if not self.has_seg_params:
            return None
        return torch.zeros(self.num_seg_params, device=self.device)

    def _set_seg_param(self, params: torch.Tensor | None, name: str, value):
        """Set a named param slot if it exists in the current param-width."""
        if params is None:
            return
        idx = self.SEG_PARAM.get(name, None)
        if idx is None or idx >= self.num_seg_params:
            return
        params[:, idx] = value

    def _set_seg_param_vec(self, params: torch.Tensor | None, name: str, value):
        """Set a named param slot on a 1D param vector if the slot exists."""
        if params is None:
            return
        idx = self.SEG_PARAM.get(name, None)
        if idx is None or idx >= self.num_seg_params:
            return
        params[idx] = value

    # ---------------- debug helpers ----------------
    def skill_name(self, skill_id: int) -> str:
        return self.SKILL_NAMES[int(skill_id)]

    def skill_names(self, skill_ids: torch.Tensor) -> list[str]:
        return [self.SKILL_NAMES[int(i)] for i in skill_ids.detach().cpu().tolist()]

    # ---------------- compat: current_alpha ----------------
    @property
    def current_alpha(self) -> torch.Tensor:
        """Path progress alpha in [0,1], compat with older code: (N,1)."""
        a = self.t_alpha[self.current_waypoints_index].unsqueeze(-1)
        return a

    # ----------------- legacy compat properties -----------------
    @property
    def robot_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, :3]

    @property
    def robot_velocity_w(self) -> torch.Tensor:
        return self.robot.data.root_lin_vel_w[:, :3]

    @property
    def target_pos_w(self) -> torch.Tensor:
        # use final goal for legacy reward gating
        return self.pos_path_w[:, -1, :]

    @property
    def target_heading_b(self) -> torch.Tensor:
        # heading error to final goal heading, in body frame
        quat = self.robot.data.root_quat_w
        _, _, robot_yaw = euler_xyz_from_quat(quat)
        target_yaw = self.heading_path_w[:, -1, 0]
        return wrap_to_pi(target_yaw - robot_yaw)

    # ----------------- exposed command -----------------
    def _meta_dim(self) -> int:
        if not getattr(self.cfg.ranges, "include_meta", True):
            return 0
        base = 5  # path_progress, cur_seg_progress, dist_to_cur_start, dist_to_cur_end, in_cur
        if getattr(self.cfg.ranges, "include_skill_onehot", True):
            base += self.num_skills
        base += self.num_seg_params  # cur params
        if getattr(self.cfg.ranges, "include_next_segment", True):
            if getattr(self.cfg.ranges, "include_skill_onehot", True):
                base += self.num_skills
            base += 1 + self.num_seg_params
        return base

    @property
    def command(self) -> torch.Tensor:
        if self.obs_meta.numel() == 0:
            return self.obs_slices.reshape(self.num_envs, -1)
        return torch.cat([self.obs_slices.reshape(self.num_envs, -1), self.obs_meta], dim=-1)

    # ----------------- lifecycle -----------------
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._goal_hold[env_ids] = False
        return super().reset(env_ids)

    def update(self):
        self._update_command()
        self._update_metrics()

    # ----------------- planned frame -----------------
    def _set_planned_frame_from_robot(self, env_ids: torch.Tensor):
        pos = self.robot.data.root_pos_w[env_ids].clone()
        quat = self.robot.data.root_quat_w[env_ids]
        _, _, yaw = euler_xyz_from_quat(quat)

        self._planned_start_pos[env_ids] = pos
        self._planned_yaw[env_ids] = yaw

        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        left = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
        self._planned_forward_dir[env_ids] = fwd
        self._planned_left_dir[env_ids] = left

    def _dist_along_planned(self) -> torch.Tensor:
        rel = self.robot.data.root_pos_w[:, :2] - self._planned_start_pos[:, :2]
        return torch.sum(rel * self._planned_forward_dir, dim=-1)

    # ----------------- sampling helpers -----------------
    def _sample_uniform_range(self, num_envs: int, low: float, high: float, fallback: float) -> torch.Tensor:
        """Sample one scalar per env from [low, high], with robust fallbacks."""
        lo = float(low)
        hi = float(high)
        if not math.isfinite(lo) or not math.isfinite(hi):
            return torch.full((num_envs,), float(fallback), device=self.device)
        if hi < lo:
            lo, hi = hi, lo
        if hi - lo < 1e-6:
            return torch.full((num_envs,), lo, device=self.device)
        return torch.empty((num_envs,), device=self.device).uniform_(lo, hi)

    def _build_env_bounds(self, start_pos: torch.Tensor) -> torch.Tensor:
        """Compute per-env XY bounds based on cfg.path_generator_cfg.env_bounds."""
        bounds = self.cfg.path_generator_cfg.env_bounds
        x_min = start_pos[:, 0] + bounds.x_range[0]
        x_max = start_pos[:, 0] + bounds.x_range[1]
        y_min = start_pos[:, 1] + bounds.y_range[0]
        y_max = start_pos[:, 1] + bounds.y_range[1]
        return torch.stack([x_min, x_max, y_min, y_max], dim=-1)

    def _compute_max_path_length(
        self,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        env_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Max forward distance within env bounds along planned yaw."""
        start_xy = start_pos[:, :2]
        forward_dir = torch.stack([torch.cos(start_yaw), torch.sin(start_yaw)], dim=-1)

        x_min, x_max, y_min, y_max = env_bounds.unbind(dim=-1)
        t_x_min = (x_min - start_xy[:, 0]) / (forward_dir[:, 0] + 1e-6)
        t_x_max = (x_max - start_xy[:, 0]) / (forward_dir[:, 0] + 1e-6)
        t_y_min = (y_min - start_xy[:, 1]) / (forward_dir[:, 1] + 1e-6)
        t_y_max = (y_max - start_xy[:, 1]) / (forward_dir[:, 1] + 1e-6)

        t_all = torch.stack([t_x_min, t_x_max, t_y_min, t_y_max], dim=-1)
        t_all = torch.where(t_all > 0, t_all, torch.full_like(t_all, float("inf")))
        max_len = torch.min(t_all, dim=-1).values

        max_len = max_len - float(self.cfg.path_generator_cfg.env_bounds.boundary_margin)
        return torch.clamp(max_len, min=1.0)

    def _sample_path_length(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        min_len: float | None = None,
    ) -> torch.Tensor:
        """Sample a forward path length, clamped to env bounds."""
        path_len = torch.full(
            (len(env_ids),),
            float(self.cfg.ranges.default_path_len),
            device=self.device,
        )
        sampling = self.cfg.sampling
        if getattr(sampling, "sample_goal_distance", False):
            min_d, max_d, _ = sampling.end_to_start_pos
            path_len = self._sample_uniform_range(
                len(env_ids),
                float(min_d),
                float(max_d),
                fallback=float(self.cfg.ranges.default_path_len),
            )

        env_bounds = self._build_env_bounds(start_pos)
        max_len = self._compute_max_path_length(start_pos, start_yaw, env_bounds)

        if min_len is not None:
            path_len = torch.maximum(path_len, torch.full_like(path_len, float(min_len)))

        path_len = torch.minimum(path_len, max_len)
        return torch.clamp(path_len, min=1.0)

    def _sample_goal_yaw(self, yaw0: torch.Tensor) -> torch.Tensor:
        """Sample a goal yaw when interpolation is enabled; otherwise return yaw0."""
        sampling = self.cfg.sampling
        yaw_mode = str(getattr(sampling, "yaw_mode", "fixed")).lower()
        if yaw_mode not in ("interp", "linear", "lerp"):
            return yaw0
        lo, hi = sampling.end_heading
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            return yaw0
        if hi < lo:
            lo, hi = hi, lo
        if hi - lo < 1e-6:
            return torch.full_like(yaw0, lo)
        return torch.empty_like(yaw0).uniform_(lo, hi)

    def _build_yaw_traj(self, yaw0: torch.Tensor, yaw_goal: torch.Tensor) -> torch.Tensor:
        """Build yaw trajectory for the current path (fixed or interpolated)."""
        sampling = self.cfg.sampling
        yaw_mode = str(getattr(sampling, "yaw_mode", "fixed")).lower()
        if yaw_mode not in ("interp", "linear", "lerp"):
            return yaw0[:, None].repeat(1, self.num_waypoints)
        delta = wrap_to_pi(yaw_goal - yaw0)
        return yaw0[:, None] + delta[:, None] * self.t_alpha[None, :]

    def _build_fixed_yaw_traj(self, yaw0: torch.Tensor) -> torch.Tensor:
        """Always keep yaw fixed along the path."""
        return yaw0[:, None].repeat(1, self.num_waypoints)

    @property
    def current_skill_id(self) -> torch.Tensor:
        """Current active skill-id for each environment."""
        self._select_current_segment()
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self._seg_skill[env_ids, self._cur_seg_idx]

    def is_in_skill_phase(self, skill: str | int) -> torch.Tensor:
        """Return a per-env boolean mask for whether the current segment matches `skill`."""
        if isinstance(skill, str):
            if skill not in self.SKILL_ID:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            skill_id = int(self.SKILL_ID[skill])
        else:
            skill_id = int(skill)
        return self.current_skill_id == skill_id

    @property
    def is_in_walk_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("walk")

    @property
    def is_in_jump_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("jump")

    @property
    def is_in_stairs_up_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("stairs_up")

    @property
    def is_in_stairs_down_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("stairs_down")

    @property
    def is_in_stairs_phase(self) -> torch.Tensor:
        return self.is_in_stairs_up_phase | self.is_in_stairs_down_phase

    @property
    def is_in_crouch_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("crouch")

    @property
    def is_in_platform_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("platform")

    @property
    def is_in_climb_phase(self) -> torch.Tensor:
        return self.is_in_skill_phase("climb")

    # ----------------- segment table API -----------------
    def _clear_segments(self, env_ids: torch.Tensor):
        self._num_segs[env_ids] = 0
        self._seg_skill[env_ids, :] = 0
        self._seg_s0[env_ids, :] = 0.0
        self._seg_s1[env_ids, :] = 0.0
        if self.has_seg_params and self._seg_params is not None:
            self._seg_params[env_ids, :, :] = 0.0

    def _append_segment(
        self,
        env_ids: torch.Tensor,
        skill_id: int,
        s0: torch.Tensor,
        s1: torch.Tensor,
        params: torch.Tensor | None = None,
        min_len: float = 1e-3,
    ):
        seg_len = (s1 - s0)
        keep = seg_len > min_len
        if not torch.any(keep):
            return

        cur_n = self._num_segs[env_ids]
        cap_ok = cur_n < self.max_segments
        mask = keep & cap_ok
        if not torch.any(mask):
            return

        env_sel = env_ids[mask]
        idx = self._num_segs[env_sel]

        self._seg_skill[env_sel, idx] = int(skill_id)
        self._seg_s0[env_sel, idx] = s0[mask]
        self._seg_s1[env_sel, idx] = s1[mask]
        if self.has_seg_params and self._seg_params is not None and params is not None:
            p = params[mask]
            if p.shape[-1] != self.num_seg_params:
                if p.shape[-1] > self.num_seg_params:
                    p = p[:, : self.num_seg_params]
                else:
                    pad = torch.zeros(p.shape[0], self.num_seg_params - p.shape[-1], device=self.device)
                    p = torch.cat([p, pad], dim=-1)
            self._seg_params[env_sel, idx, :] = p
        self._num_segs[env_sel] = self._num_segs[env_sel] + 1

    def _select_current_segment(self):
        s = self._dist_along_planned()
        cur = torch.zeros_like(self._cur_seg_idx)
        num = self._num_segs.clamp(min=1)

        for j in range(self.max_segments):
            s0 = self._seg_s0[:, j]
            s1 = self._seg_s1[:, j]
            inside = (j < num) & (s >= s0) & (s < s1)
            cur = torch.where(inside, torch.full_like(cur, j), cur)

        last = (num - 1).clamp(min=0)
        b = torch.arange(self.num_envs, device=self.device)
        beyond = s >= self._seg_s1[b, last]
        cur = torch.where(beyond, last, cur)
        self._cur_seg_idx = cur

    # ----------------- slice & meta -----------------
    def _update_slice_and_meta(self):
        self._select_current_segment()
        self._advance_waypoint_index()
        # Freeze command at goal to keep robot stationary after reaching endpoint.
        new_hold = self.goal_reached & (~self._goal_hold)
        if torch.any(new_hold):
            env_ids = torch.nonzero(new_hold, as_tuple=False).squeeze(-1)
            self._apply_goal_hold(env_ids)

        N = self.num_envs
        W = self.num_waypoints
        K = self.num_lookahead_waypoints
        device = self.device

        offsets = torch.arange(K, device=device)
        idx = self.current_waypoints_index[:, None] + offsets[None, :]
        idx = torch.clamp(idx, 0, W - 1)

        batch = torch.arange(N, device=device)[:, None]
        pts = self.pos_path_w[batch, idx]                    # (N,K,3)
        yaws = self.heading_path_w[batch, idx].squeeze(-1)   # (N,K)

        robot_pos = self.robot.data.root_pos_w[:, :3]
        d = pts - robot_pos[:, None, :]

        cy = torch.cos(self._planned_yaw)[:, None]
        sy = torch.sin(self._planned_yaw)[:, None]
        dx = d[..., 0]
        dy = d[..., 1]
        x_fwd = cy * dx + sy * dy
        y_lat = -sy * dx + cy * dy
        z_rel = d[..., 2]
        dyaw = wrap_to_pi(yaws - self._planned_yaw[:, None])

        scale = float(self.cfg.ranges.dist_clip)
        z_scale = float(getattr(self.cfg.ranges, 'z_clip', scale))  # backward compatible
        x_fwd = x_fwd / scale
        y_lat = y_lat / scale
        z_rel = z_rel / z_scale  # independent z scaling
        dyaw = dyaw / torch.pi

        self.obs_slices = torch.stack([x_fwd, y_lat, z_rel, dyaw], dim=-1)

        if self.obs_meta.numel() == 0:
            return

        s = self._dist_along_planned()
        path_len = self._path_len.clamp(min=1e-3)
        path_progress = (s / path_len).clamp(0.0, 1.0)

        cur_idx = self._cur_seg_idx
        b = torch.arange(N, device=device)
        cur_s0 = self._seg_s0[b, cur_idx]
        cur_s1 = self._seg_s1[b, cur_idx]
        cur_skill = self._seg_skill[b, cur_idx]
        cur_params = self._seg_params[b, cur_idx, :] if (self.has_seg_params and self._seg_params is not None) else None

        # Transition blending: smooth parameter interpolation near segment boundaries
        transition_window = float(getattr(self.cfg.ranges, 'transition_window_s', 0.0))
        if transition_window > 0.0 and cur_params is not None:
            # Get next segment info for blending
            next_idx = torch.clamp(cur_idx + 1, max=self.max_segments - 1)
            num = self._num_segs.clamp(min=1)
            last = (num - 1).clamp(min=0)
            is_last = cur_idx >= last
            next_idx_blend = torch.where(is_last, cur_idx, next_idx)
            next_params_blend = self._seg_params[b, next_idx_blend, :]

            # Get previous segment info for fade-in
            prev_idx = torch.clamp(cur_idx - 1, min=0)
            is_first = cur_idx == 0
            prev_idx_blend = torch.where(is_first, cur_idx, prev_idx)
            prev_params_blend = self._seg_params[b, prev_idx_blend, :]
            prev_s1 = self._seg_s1[b, prev_idx_blend]

            # Distance to current segment end (fade-out to next)
            dist_to_end = cur_s1 - s
            # Distance from current segment start (fade-in from prev)
            dist_from_start = s - cur_s0

            # Compute blend weights
            # Fade-out: blend cur_params -> next_params as we approach segment end
            fade_out_alpha = torch.clamp(dist_to_end / transition_window, 0.0, 1.0)
            # Fade-in: blend prev_params -> cur_params as we leave segment start
            fade_in_alpha = torch.clamp(dist_from_start / transition_window, 0.0, 1.0)

            # Apply fade-out blending (near segment end)
            in_fade_out = (dist_to_end < transition_window) & (~is_last)
            blended_params = torch.where(
                in_fade_out.unsqueeze(-1),
                fade_out_alpha.unsqueeze(-1) * cur_params + (1.0 - fade_out_alpha.unsqueeze(-1)) * next_params_blend,
                cur_params
            )

            # Apply fade-in blending (near segment start)
            in_fade_in = (dist_from_start < transition_window) & (~is_first)
            blended_params = torch.where(
                in_fade_in.unsqueeze(-1),
                fade_in_alpha.unsqueeze(-1) * blended_params + (1.0 - fade_in_alpha.unsqueeze(-1)) * prev_params_blend,
                blended_params
            )

            cur_params = blended_params

        cur_den = (cur_s1 - cur_s0).clamp(min=1e-3)
        cur_prog = ((s - cur_s0) / cur_den).clamp(0.0, 1.0)
        cur_prog_center = cur_prog * 2.0 - 1.0

        clip = float(self.cfg.ranges.dist_clip)
        dist_to_cur_start = (cur_s0 - s).clamp(-clip, clip) / clip
        dist_to_cur_end = (cur_s1 - s).clamp(-clip, clip) / clip
        in_cur = ((s >= cur_s0) & (s < cur_s1)).to(torch.float32)

        parts = [path_progress[:, None], cur_prog_center[:, None], dist_to_cur_start[:, None], dist_to_cur_end[:, None], in_cur[:, None]]

        if getattr(self.cfg.ranges, "include_skill_onehot", True):
            onehot = torch.zeros(N, self.num_skills, device=device)
            onehot.scatter_(1, cur_skill[:, None].clamp(0, self.num_skills - 1), 1.0)
            parts.append(onehot)

        if cur_params is not None:
            parts.append(cur_params)

        if getattr(self.cfg.ranges, "include_next_segment", True):
            next_idx = torch.clamp(cur_idx + 1, max=self.max_segments - 1)
            num = self._num_segs.clamp(min=1)
            last = (num - 1).clamp(min=0)
            is_last = cur_idx >= last
            next_idx = torch.where(is_last, cur_idx, next_idx)

            next_s0 = self._seg_s0[b, next_idx]
            next_skill = self._seg_skill[b, next_idx]
            next_params = self._seg_params[b, next_idx, :] if (self.has_seg_params and self._seg_params is not None) else None
            dist_to_next_start = (next_s0 - s).clamp(-clip, clip) / clip

            if getattr(self.cfg.ranges, "include_skill_onehot", True):
                n_oh = torch.zeros(N, self.num_skills, device=device)
                n_oh.scatter_(1, next_skill[:, None].clamp(0, self.num_skills - 1), 1.0)
                parts.append(n_oh)

            parts.append(dist_to_next_start[:, None])
            if next_params is not None:
                parts.append(next_params)

        self.obs_meta = torch.cat(parts, dim=-1)

    def _apply_goal_hold(self, env_ids: torch.Tensor):
        """Freeze command at the current robot pose to encourage standing still at the goal."""
        if env_ids.numel() == 0:
            return

        robot_pos = self.robot.data.root_pos_w[env_ids, :3]
        quat = self.robot.data.root_quat_w[env_ids]
        _, _, yaw = euler_xyz_from_quat(quat)

        self._planned_start_pos[env_ids] = robot_pos
        self._planned_yaw[env_ids] = yaw
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        left = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
        self._planned_forward_dir[env_ids] = fwd
        self._planned_left_dir[env_ids] = left

        self._path_len[env_ids] = 1.0
        self.pos_path_w[env_ids] = robot_pos[:, None, :].repeat(1, self.num_waypoints, 1)
        self.heading_path_w[env_ids, :, 0] = yaw[:, None].repeat(1, self.num_waypoints)

        self._clear_segments(env_ids)
        s0 = torch.zeros(len(env_ids), device=self.device)
        s1 = torch.ones(len(env_ids), device=self.device)
        params = self._new_seg_params(len(env_ids))
        self._set_seg_param(params, "v_ref", 0.0)
        self._append_segment(env_ids, self.SKILL_WALK, s0, s1, params=params)
        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)

        self.current_waypoints_index[env_ids] = 0
        self.time_left[env_ids] = 1.0e9
        self._goal_hold[env_ids] = True

    @staticmethod
    def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
        """Convert yaw angles (rad) to wxyz quaternions."""
        half = 0.5 * yaw
        # Note: IsaacLab uses wxyz ordering for quaternions.
        return torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )

    # ----------------- debug visualization -----------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "path_waypoints_visualizer"):
                self.path_waypoints_visualizer = VisualizationMarkers(self.cfg.path_waypoints_visualizer_cfg)
                self.goal_visualizer = VisualizationMarkers(self.cfg.path_goal_visualizer_cfg)
                self.start_visualizer = VisualizationMarkers(self.cfg.path_start_visualizer_cfg)
            if hasattr(self.cfg, "path_heading_visualizer_cfg") and not hasattr(self, "path_heading_visualizer"):
                self.path_heading_visualizer = VisualizationMarkers(self.cfg.path_heading_visualizer_cfg)
            if hasattr(self.cfg, "robot_heading_visualizer_cfg") and not hasattr(self, "robot_heading_visualizer"):
                self.robot_heading_visualizer = VisualizationMarkers(self.cfg.robot_heading_visualizer_cfg)
            self.path_waypoints_visualizer.set_visibility(True)
            self.goal_visualizer.set_visibility(True)
            self.start_visualizer.set_visibility(True)
            if hasattr(self, "path_heading_visualizer"):
                self.path_heading_visualizer.set_visibility(True)
            if hasattr(self, "robot_heading_visualizer"):
                self.robot_heading_visualizer.set_visibility(True)
        else:
            if hasattr(self, "path_waypoints_visualizer"):
                self.path_waypoints_visualizer.set_visibility(False)
                self.goal_visualizer.set_visibility(False)
                self.start_visualizer.set_visibility(False)
            if hasattr(self, "path_heading_visualizer"):
                self.path_heading_visualizer.set_visibility(False)
            if hasattr(self, "robot_heading_visualizer"):
                self.robot_heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.path_waypoints_visualizer.visualize(
            translations=self.pos_path_w.reshape(-1, 3),
        )
        if hasattr(self, "path_heading_visualizer"):
            yaw = self.heading_path_w[..., 0].reshape(-1)
            self.path_heading_visualizer.visualize(
                translations=self.pos_path_w.reshape(-1, 3),
                orientations=self._yaw_to_quat(yaw),
            )
        self.goal_visualizer.visualize(
            translations=self.pos_path_w[torch.arange(self.num_envs), -1],
        )
        self.start_visualizer.visualize(
            translations=self.pos_path_w[torch.arange(self.num_envs), 0],
        )
        if hasattr(self, "robot_heading_visualizer"):
            robot_pos = self.robot.data.root_pos_w[:, :3]
            _, _, robot_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
            self.robot_heading_visualizer.visualize(
                translations=robot_pos,
                orientations=self._yaw_to_quat(robot_yaw),
            )

    # ----------------- waypoint progression + metrics -----------------
    def _advance_waypoint_index(self):
        robot_pos = self.robot.data.root_pos_w[:, :3]
        quat = self.robot.data.root_quat_w
        _, _, robot_yaw = euler_xyz_from_quat(quat)

        b = torch.arange(self.num_envs, device=self.device)
        target = self.pos_path_w[b, self.current_waypoints_index]
        target_yaw = self.heading_path_w[b, self.current_waypoints_index, 0]

        # metrics
        self.metrics["error_pos_xy"] = torch.norm(target[:, :2] - robot_pos[:, :2], dim=-1)
        self.metrics["error_pos_z"] = torch.abs(target[:, 2] - robot_pos[:, 2])
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(target_yaw - robot_yaw))

        reached = self.metrics["error_pos_xy"] < float(self.cfg.ranges.waypoint_reach_threshold)
        self.current_waypoints_index = torch.where(
            reached,
            torch.clamp(self.current_waypoints_index + 1, max=self.num_waypoints - 1),
            self.current_waypoints_index,
        )

        final = self.pos_path_w[b, -1]
        goal_dis = torch.norm(final[:, :2] - robot_pos[:, :2], dim=-1)
        self.goal_reached = self.goal_reached | ((goal_dis < float(self.cfg.ranges.waypoint_reach_threshold)) & (self.current_waypoints_index >= self.num_waypoints - 1))

    # ----------------- derived hooks -----------------
    def _resample_command(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def _update_command(self):
        pass

    def _update_metrics(self):
        # metrics + observations are updated together for path commands
        self._update_slice_and_meta()
