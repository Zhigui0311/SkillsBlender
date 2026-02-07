from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.markers import VisualizationMarkers

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg, sync_skill_params_to_generator_cfg
from skillsblender.tasks.path.utils.path_generator import TerrainAwarePathGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PlannerPathCommand(SegmentPathCommand):
    """Planner-driven path command using TerrainAwarePathGenerator.

    - Uses cfg.path_generator_cfg.skill_sequence to plan segments.
    - Falls back to a single "walk" segment if sequence is empty.
    """

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # optional height scanner (RayCaster term in scene)
        self.height_scanner = None
        if hasattr(env.scene, "sensors") and isinstance(getattr(env.scene, "sensors", None), dict):
            self.height_scanner = env.scene.sensors.get("height_scanner", None)

        gen_cfg = cfg.path_generator_cfg.copy()
        gen_cfg.num_waypoints = cfg.ranges.num_waypoints
        gen_cfg.num_seg_params = max(int(cfg.ranges.num_seg_params), 0)
        sync_skill_params_to_generator_cfg(cfg, generator_cfg=gen_cfg, skill_names=list(self.SKILL_NAMES))
        self._generator_cfg = gen_cfg
        # Build skill-id map (skills are now explicit, no aliases).
        skill_id_map = dict(self.SKILL_ID)

        self._path_generator = TerrainAwarePathGenerator(
            gen_cfg,
            self.device,
            num_seg_params=self.num_seg_params,
            seg_param=self.SEG_PARAM,
            skill_id_map=skill_id_map,
        )

    def _build_env_bounds(self, start_pos: torch.Tensor) -> torch.Tensor:
        bounds = self._generator_cfg.env_bounds
        x_min = start_pos[:, 0] + bounds.x_range[0]
        x_max = start_pos[:, 0] + bounds.x_range[1]
        y_min = start_pos[:, 1] + bounds.y_range[0]
        y_max = start_pos[:, 1] + bounds.y_range[1]
        return torch.stack([x_min, x_max, y_min, y_max], dim=-1)

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

    def _sample_path_length(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        env_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Get per-env target path lengths and clamp to local env bounds."""
        path_len = torch.full(
            (len(env_ids),),
            float(self.cfg.ranges.default_path_len),
            device=self.device,
        )

        sampling = self.cfg.sampling
        if getattr(sampling, "sample_goal_distance", False):
            min_len, max_len, _ = sampling.end_to_start_pos
            path_len = self._sample_uniform_range(
                len(env_ids),
                float(min_len),
                float(max_len),
                fallback=float(self.cfg.ranges.default_path_len),
            )

        max_path_len = self._path_generator.compute_max_path_length(start_pos, start_yaw, env_bounds)
        path_len = torch.minimum(path_len, max_path_len)
        return torch.clamp(path_len, min=1.0)

    def _gather_terrain_data(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        terrain_data: dict[str, torch.Tensor] = {}
        if self.height_scanner is not None and hasattr(self.height_scanner, "data"):
            terrain_data["height_scanner"] = self.height_scanner.data.ray_hits_w[env_ids]
        return terrain_data

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        start_pos = self._planned_start_pos[env_ids]
        # Optionally override planned yaw to be independent of robot yaw
        sampling = self.cfg.sampling
        yaw_type = getattr(sampling, "yaw_type", "along_path")
        if yaw_type in ("fixed", "world", "decoupled", "random"):
            lo, hi = sampling.start_heading
            planned_yaw = self._sample_uniform_range(len(env_ids), float(lo), float(hi), fallback=0.0)
            self._planned_yaw[env_ids] = planned_yaw
            fwd = torch.stack([torch.cos(planned_yaw), torch.sin(planned_yaw)], dim=-1)
            left = torch.stack([-torch.sin(planned_yaw), torch.cos(planned_yaw)], dim=-1)
            self._planned_forward_dir[env_ids] = fwd
            self._planned_left_dir[env_ids] = left

        start_yaw = self._planned_yaw[env_ids]
        env_bounds = self._build_env_bounds(start_pos)
        terrain_data = self._gather_terrain_data(env_ids)
        path_len = self._sample_path_length(env_ids, start_pos, start_yaw, env_bounds)

        skill_sequence = list(getattr(self._generator_cfg, "skill_sequence", []) or [])
        if not skill_sequence:
            skill_sequence = ["walk"]

        path_result = self._path_generator.generate_multi_skill_path(
            env_ids=env_ids,
            skill_sequence=skill_sequence,
            start_pos=start_pos,
            start_yaw=start_yaw,
            env_bounds=env_bounds,
            terrain_data=terrain_data,
            path_length=path_len,
        )

        self._path_len[env_ids] = path_result.path_length
        self.pos_path_w[env_ids] = path_result.waypoints
        self.heading_path_w[env_ids, :, 0] = path_result.headings

        # Optional yaw interpolation override (only for walk-only paths).
        sampling = self.cfg.sampling
        yaw_mode = str(getattr(sampling, "yaw_mode", "fixed")).lower()
        allow_interp = all(s == "walk" for s in skill_sequence)
        if allow_interp and yaw_mode in ("interp", "linear", "lerp"):
            yaw0 = self._planned_yaw[env_ids]
            yaw_goal = self._sample_goal_yaw(yaw0)
            yaw_traj = self._build_yaw_traj(yaw0, yaw_goal)
            self.heading_path_w[env_ids, :, 0] = yaw_traj

        self._clear_segments(env_ids)
        num_segments = int(path_result.num_segments.max().item())
        for seg_idx in range(num_segments):
            skill_id = int(path_result.segment_skills[0, seg_idx].item())
            s0 = path_result.segment_s0[:, seg_idx]
            s1 = path_result.segment_s1[:, seg_idx]
            params = path_result.segment_params[:, seg_idx, :]
            self._append_segment(env_ids, skill_id, s0, s1, params=params)

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "path_waypoints_visualizer"):
                self.path_waypoints_visualizer = VisualizationMarkers(self.cfg.path_waypoints_visualizer_cfg)
                self.goal_visualizer = VisualizationMarkers(self.cfg.path_goal_visualizer_cfg)
                self.start_visualizer = VisualizationMarkers(self.cfg.path_start_visualizer_cfg)
            self.path_waypoints_visualizer.set_visibility(True)
            self.goal_visualizer.set_visibility(True)
            self.start_visualizer.set_visibility(True)
        else:
            if hasattr(self, "path_waypoints_visualizer"):
                self.path_waypoints_visualizer.set_visibility(False)
                self.goal_visualizer.set_visibility(False)
                self.start_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.path_waypoints_visualizer.visualize(
            translations=self.pos_path_w.reshape(-1, 3),
        )
        self.goal_visualizer.visualize(
            translations=self.pos_path_w[torch.arange(self.num_envs), -1],
        )
        self.start_visualizer.visualize(
            translations=self.pos_path_w[torch.arange(self.num_envs), 0],
        )
