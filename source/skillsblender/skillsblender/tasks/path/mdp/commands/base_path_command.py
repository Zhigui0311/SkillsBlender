from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
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
        "sidestep",
        "climb",
    ]
    SKILL_ID = {name: i for i, name in enumerate(SKILL_NAMES)}
    NUM_SKILLS = len(SKILL_NAMES)
    SKILL_WALK = SKILL_ID["walk"]
    SKILL_JUMP = SKILL_ID["jump"]
    SKILL_STAIRS_UP = SKILL_ID["stairs_up"]
    SKILL_STAIRS_DOWN = SKILL_ID["stairs_down"]
    SKILL_CROUCH = SKILL_ID["crouch"]
    SKILL_SIDESTEP = SKILL_ID["sidestep"]
    SKILL_CLIMB = SKILL_ID["climb"]

    # -------- segment parameter slots (single source of truth) --------
    SEG_PARAM = {
        "v_ref": 0,
        "yaw_rate_ref": 1,
        "base_height_ref": 2,
        "clearance_ref": 3,
        "jump_height_ref": 4,
        "misc": 5,
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
        self.num_seg_params = cfg.ranges.num_seg_params

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
        self._seg_params = torch.zeros(self.num_envs, self.max_segments, self.num_seg_params, device=self.device)

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
        self._resample_command(env_ids)

    def update(self):
        self._update_command()
        self._update_slice_and_meta()

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

    # ----------------- segment table API -----------------
    def _clear_segments(self, env_ids: torch.Tensor):
        self._num_segs[env_ids] = 0
        self._seg_skill[env_ids, :] = 0
        self._seg_s0[env_ids, :] = 0.0
        self._seg_s1[env_ids, :] = 0.0
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
        if params is not None:
            self._seg_params[env_sel, idx, :] = params[mask]
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
        x_fwd = x_fwd / scale
        y_lat = y_lat / scale
        z_rel = z_rel / scale
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
        cur_params = self._seg_params[b, cur_idx, :]

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

        parts.append(cur_params)

        if getattr(self.cfg.ranges, "include_next_segment", True):
            next_idx = torch.clamp(cur_idx + 1, max=self.max_segments - 1)
            num = self._num_segs.clamp(min=1)
            last = (num - 1).clamp(min=0)
            is_last = cur_idx >= last
            next_idx = torch.where(is_last, cur_idx, next_idx)

            next_s0 = self._seg_s0[b, next_idx]
            next_skill = self._seg_skill[b, next_idx]
            next_params = self._seg_params[b, next_idx, :]
            dist_to_next_start = (next_s0 - s).clamp(-clip, clip) / clip

            if getattr(self.cfg.ranges, "include_skill_onehot", True):
                n_oh = torch.zeros(N, self.num_skills, device=device)
                n_oh.scatter_(1, next_skill[:, None].clamp(0, self.num_skills - 1), 1.0)
                parts.append(n_oh)

            parts.append(dist_to_next_start[:, None])
            parts.append(next_params)

        self.obs_meta = torch.cat(parts, dim=-1)

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
