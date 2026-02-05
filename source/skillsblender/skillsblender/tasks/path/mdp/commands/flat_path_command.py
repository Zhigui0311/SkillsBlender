from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers
from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FlatPathCommand(SegmentPathCommand):
    """Single-segment WALK path."""

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        total_len = torch.full((len(env_ids),), float(self.cfg.ranges.default_path_len), device=self.device)
        self._path_len[env_ids] = total_len

        start = self._planned_start_pos[env_ids].clone()
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha
        yaw = self._planned_yaw[env_ids][:, None].repeat(1, self.num_waypoints)

        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw

        self._clear_segments(env_ids)
        s0 = torch.zeros(len(env_ids), device=self.device)
        s1 = total_len
        params = self._new_seg_params(len(env_ids))
        self._set_seg_param(params, "v_ref", 1.0)
        self._append_segment(env_ids, self.SKILL_ID["walk"], s0, s1, params=params)
        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)

        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False



    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        Set up or remove debug visualization markers.
        Args:
            debug_vis: Whether to enable debug visualization.
        """
        if debug_vis:
            if not hasattr(self, "path_waypoints_visualizer"):
                self.path_waypoints_visualizer = VisualizationMarkers(self.cfg.path_waypoints_visualizer_cfg)
                self.goal_visualizer = VisualizationMarkers(self.cfg.path_goal_visualizer_cfg)
                self.start_visualizer = VisualizationMarkers(self.cfg.path_start_visualizer_cfg)
            # set their visibility to true
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
        ),
        self.goal_visualizer.visualize(
            translations=self.pos_path_w[torch.arange(self.num_envs), -1],
            
        ),
        self.start_visualizer.visualize(
            translations=self.pos_path_w[torch.arange(self.num_envs), 0],       
        )
