# path_command.py
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import (
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .path_command_cfg import PathCommandCfg


class PathCommand(CommandTerm):
    """
    Decoupled Path Generator & Tracker.
    
    Architecture:
    1. _sample_key_waypoints: Generates sparse key points (Skeleton).
    2. _interpolate_dense_path: Fills in dense points (Flesh).
    3. _update_command: Tracks progress along the dense path.
    4. get_path_slice: Transforms a window of points to Body Frame for RL.
    """

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # --- 1. Buffer Initialization ---
        # "Total Waypoints" / "Dense Points"
        self.num_dense = cfg.num_dense_points
        
        # 存储世界坐标系下的完整路径
        # Pos: (N, num_dense, 3), Yaw: (N, num_dense, 1)
        self.full_path_pos_w = torch.zeros(self.num_envs, self.num_dense, 3, device=self.device)
        self.full_path_yaw_w = torch.zeros(self.num_envs, self.num_dense, 1, device=self.device)

        # 记录机器人当前在路径上的索引 (Progress)
        self.current_path_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 显式存储 Body Frame 下的“当前单一指令” (用于 Reward 计算或 Tensorboard 监控)
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.yaw_command_b = torch.zeros(self.num_envs, 1, device=self.device)

        # --- 2. Interpolation Setup ---
        # 计算分段数 (Key Points - 1)
        # 例如: 3个关键点 = 2段 (Start->Mid, Mid->End)
        self.num_segments = cfg.num_key_waypoints - 1
        if self.num_segments < 1:
            raise ValueError("num_key_waypoints must be >= 2")
            
        # 每一段分配多少个点
        self.points_per_segment = self.num_dense // self.num_segments
        # 重新校准 num_dense 以防止整除误差
        self.num_dense = self.points_per_segment * self.num_segments
        
        # 预计算时间步 t (0 to 1) 用于插值
        self.t_segment = torch.linspace(0, 1, self.points_per_segment, device=self.device)

        # --- 3. Metrics & Vis ---
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

        self._vis_dense = None
        self._vis_keys = None
        
        if self.cfg.debug_vis:
            # 绿色小球：显示稠密路径 (Dense Path)
            marker_cfg = SPHERE_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/PathDense"
            marker_cfg.markers["sphere"].scale = (0.08, 0.08, 0.08)
            marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self._vis_dense = VisualizationMarkers(marker_cfg)
            
            # 红色大球：显示关键规划点 (Key Waypoints)
            key_cfg = SPHERE_MARKER_CFG.copy()
            key_cfg.prim_path = "/Visuals/PathKeys"
            key_cfg.markers["sphere"].scale = (0.2, 0.2, 0.2)
            key_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
            self._vis_keys = VisualizationMarkers(key_cfg)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        """Called during environment reset."""
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        """
        [Generation Phase]
        Executes: Plan -> Interpolate -> Store
        """
        # 1. Plan Skeleton (Key Waypoints)
        keys_pos, keys_yaw = self._sample_key_waypoints(env_ids)

        # 2. Interpolate Flesh (Dense Path)
        dense_pos, dense_yaw = self._interpolate_dense_path(keys_pos, keys_yaw)

        # 3. Store in Buffer
        self.full_path_pos_w[env_ids] = dense_pos
        self.full_path_yaw_w[env_ids] = dense_yaw
        
        # 4. Reset Progres
        # 5. Visualize Key Waypoints (Optional, just once)
        if self._vis_keys and len(env_ids) > 0:
            # Flatten to visualize all points at once
            self._vis_keys.visualize(keys_pos.view(-1, 3))

    def _update_command(self):
        """
        [Tracking Phase] - Called every step
        """
        robot_pos = self.robot.data.root_pos_w[:, :3].unsqueeze(1) # (N, 1, 3)
        robot_quat = self.robot.data.root_quat_w
        
        # 1. Track Progress: Find closest point index
        # Optimization: Search full path (robust) or local window (fast). Here using full path for simplicity.
        dist_sq = torch.sum((self.full_path_pos_w - robot_pos)**2, dim=-1) # (N, num_dense)
        closest_indices = torch.argmin(dist_sq, dim=1) # (N,)
        
        # Ensure progress is monotonic (cannot go backward)
        self.current_path_index = torch.max(self.current_path_index, closest_indices)
        
        # 2. Compute Immediate Command (for Logging/Reward)
        # Look ahead slightly (e.g., +5 indices) for the "target"
        target_idx = torch.clamp(self.current_path_index + 5, max=self.num_dense - 1)
        
        target_pos_w = self.full_path_pos_w[torch.arange(self.num_envs), target_idx]
        target_yaw_w = self.full_path_yaw_w[torch.arange(self.num_envs), target_idx].squeeze(-1)
        
        # Transform to Body Frame
        self.pos_command_b = quat_rotate_inverse(robot_quat, target_pos_w - robot_pos.squeeze(1))
        
        current_yaw = yaw_quat(robot_quat)
        self.yaw_command_b = wrap_to_pi(target_yaw_w - current_yaw).unsqueeze(-1)
        
        # 3. Update Metrics
        self.metrics["error_pos"] = torch.norm(self.pos_command_b[:, :2], dim=1)
        self.metrics["error_yaw"] = torch.abs(self.yaw_command_b).squeeze()
        
        # 4. Visualize Dense Path (Local view)
        if self._vis_dense:
            # Visualize next 20 points
            vis_len = 20
            idx_start = self.current_path_index
            batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, vis_len)
            step_idx = torch.clamp(idx_start.unsqueeze(1) + torch.arange(vis_len, device=self.device), max=self.num_dense-1)
            
            points_to_show = self.full_path_pos_w[batch_idx, step_idx]
            self._vis_dense.visualize(points_to_show.reshape(-1, 3))

    # -------------------------------------------------------------------------
    # [Level 3] Observation Interface (The Observer)
    # -------------------------------------------------------------------------
    def get_path_slice(self) -> torch.Tensor:
        """
        Get the path slice (waypoints) in Body Frame.
        Returns: (N, num_lookahead_waypoints, 4) -> [x, y, z, yaw_rel]
        """
        window_size = self.cfg.num_lookahead_waypoints
        
        # 1. Create Slicing Indices
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, window_size)
        step_indices = torch.arange(window_size, device=self.device).unsqueeze(0) + self.current_path_index.unsqueeze(1)
        
        # Clamp to avoid out of bounds (pad with last point)
        step_indices = torch.clamp(step_indices, max=self.num_dense - 1)
        
        # 2. Gather World Data
        pos_w = self.full_path_pos_w[batch_indices, step_indices] # (N, W, 3)
        yaw_w = self.full_path_yaw_w[batch_indices, step_indices] # (N, W, 1)
        
        # 3. Transform to Body Frame
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        
        # Position: R_inv * (Target - Current)
        pos_diff = pos_w - robot_pos.unsqueeze(1)
        quat_expanded = robot_quat.unsqueeze(1).expand(-1, window_size, -1)
        pos_b = quat_rotate_inverse(quat_expanded, pos_diff)
        
        # Yaw: Wrap(Target - Current)
        robot_yaw = yaw_quat(robot_quat).unsqueeze(1).unsqueeze(2)
        yaw_b = wrap_to_pi(yaw_w - robot_yaw)
        
        # 4. Concat
        return torch.cat([pos_b, yaw_b], dim=-1)

    # -------------------------------------------------------------------------
    # [Level 1] Planner Logic (Skeleton)
    # -------------------------------------------------------------------------
    def _sample_key_waypoints(self, env_ids: torch.Tensor):
        """Generates sparse Key Waypoints."""
        num_batch = len(env_ids)
        num_keys = self.cfg.num_key_waypoints
        
        keys_pos = torch.zeros(num_batch, num_keys, 3, device=self.device)
        keys_yaw = torch.zeros(num_batch, num_keys, 1, device=self.device)
        
        # Point 0: Start at Robot Current Pose
        keys_pos[:, 0] = self.robot.data.root_pos_w[env_ids].clone()
        start_yaw = yaw_quat(self.robot.data.root_quat_w[env_ids])
        keys_yaw[:, 0] = start_yaw.unsqueeze(-1)
        
        # Generate subsequent points (Random Walk Logic for Demo)
        # NOTE: This is where you would plug in A* or Terrain Logic
        r_range = self.cfg.path_length_range
        
        for i in range(1, num_keys):
            # Random distance and heading change
            dist = torch.empty(num_batch, device=self.device).uniform_(r_range[0]/2, r_range[1]/2)
            delta_heading = torch.empty(num_batch, device=self.device).uniform_(*self.cfg.heading_range)
            
            prev_yaw = keys_yaw[:, i-1, 0]
            new_yaw = prev_yaw + delta_heading
            
            # Calculate Pos
            keys_pos[:, i, 0] = keys_pos[:, i-1, 0] + dist * torch.cos(new_yaw)
            keys_pos[:, i, 1] = keys_pos[:, i-1, 1] + dist * torch.sin(new_yaw)
            
            # Calculate Height (Z)
            if self.cfg.height_change:
                # Random height variation
                h_diff = torch.empty(num_batch, device=self.device).uniform_(-self.cfg.max_height_diff, self.cfg.max_height_diff)
                keys_pos[:, i, 2] = keys_pos[:, i-1, 2] + h_diff
            else:
                keys_pos[:, i, 2] = keys_pos[:, i-1, 2]
                
            # Set Yaw
            keys_yaw[:, i, 0] = wrap_to_pi(new_yaw)
            
        return keys_pos, keys_yaw

    # -------------------------------------------------------------------------
    # [Level 2] Interpolation Logic (Flesh)
    # -------------------------------------------------------------------------
    def _interpolate_dense_path(self, key_pos, key_yaw):
        """Interpolates between key waypoints to generate dense path."""
        segments_pos = []
        segments_yaw = []
        
        # t shape: (1, points_per_seg, 1)
        t = self.t_segment.view(1, -1, 1)
        
        for i in range(self.num_segments):
            # Get Start and End for this segment
            p0 = key_pos[:, i].unsqueeze(1)     # (N, 1, 3)
            p3 = key_pos[:, i+1].unsqueeze(1)   # (N, 1, 3)
            
            y0 = key_yaw[:, i].unsqueeze(1)     # (N, 1, 1)
            y1 = key_yaw[:, i+1].unsqueeze(1)   # (N, 1, 1)
            
            # --- Position Interpolation ---
            if self.cfg.interpolation_type == 'linear':
                seg_pos = p0 + (p3 - p0) * t
                
            elif self.cfg.interpolation_type == 'bezier':
                # Heuristic Control Points
                dist = torch.norm(p3 - p0, dim=-1, keepdim=True)
                
                # Use yaw to determine direction of control points
                # P1 extends from P0 along P0's yaw
                dir_0 = torch.cat([torch.cos(y0), torch.sin(y0), torch.zeros_like(y0)], dim=-1)
                p1 = p0 + dir_0 * (dist * 0.33)
                # Ensure Z logic: for 3D bezier, maybe pitch P1 up? 
                # Keeping simple: P1 z is linear interpolation of z
                p1[..., 2] = p0[..., 2] + (p3[..., 2] - p0[..., 2]) * 0.33

                # P2 extends backward from P3 along P3's yaw
                dir_1 = torch.cat([torch.cos(y1), torch.sin(y1), torch.zeros_like(y1)], dim=-1)
                # Note: control point should be *behind* target if following target yaw
                p2 = p3 - dir_1 * (dist * 0.33)
                p2[..., 2] = p0[..., 2] + (p3[..., 2] - p0[..., 2]) * 0.66

                seg_pos = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
            
            else:
                raise ValueError(f"Unknown interpolation: {self.cfg.interpolation_type}")
            
            # --- Yaw Interpolation (Linear) ---
            delta_yaw = wrap_to_pi(y1 - y0)
            seg_yaw = wrap_to_pi(y0 + delta_yaw * t)
            
            segments_pos.append(seg_pos)
            segments_yaw.append(seg_yaw)
            
        # Concatenate all segments
        # (N, num_dense, 3)
        full_pos = torch.cat(segments_pos, dim=1)
        full_yaw = torch.cat(segments_yaw, dim=1)
        
        return full_pos, full_yaw