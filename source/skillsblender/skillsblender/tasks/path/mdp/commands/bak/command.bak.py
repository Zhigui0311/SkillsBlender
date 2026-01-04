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
    yaw_quat,
    quat_apply
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .path_command_cfg import PathCommandCfg


class PathCommand(CommandTerm):
    """
    Command generator that plans a path (Linear or Bezier) and slices it into waypoints 
    for the robot to follow.
    
    This class handles:
    1. Trajectory Generation: Linear or Bezier curves.
    2. Heading Control: Path-following or Decoupled heading.
    3. Observation Slicing: Providing a window of future waypoints in the robot's body frame.
    """

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        """Configuration for the PathCommand class."""
        super().__init__(cfg, env)

        # -- Robot Asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -- Internal Buffers
        # Total number of interpolation points for the full generated path
        self.num_points = cfg.inpoints.num_points 
        
        # Buffers to store the generated full path in World Frame
        # Shape: (num_envs, num_points, 3) for pos, (num_envs, num_points, 1) for yaw
        self.full_path_pos_w = torch.zeros(self.num_envs, self.num_points, 3, device=self.device)
        self.full_path_yaw_w = torch.zeros(self.num_envs, self.num_points, 1, device=self.device)

        # Buffer to store the current progress (index) along the path
        self.current_path_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Pre-computed time steps for interpolation (0 to 1)
        self.t_steps = torch.linspace(0, 1, self.num_points, device=self.device)
        
        # -- Metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # -- Visualization
        self._vis_marker = None
        if self.cfg.debug_vis:
            # Create a sphere marker configuration
            marker_cfg = SPHERE_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/PathTarget"
            marker_cfg.markers["sphere"].scale = (0.1, 0.1, 0.1)
            marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0) # Green
            self._vis_marker = VisualizationMarkers(marker_cfg)

    def __str__(self) -> str:
        return "PathCommand: Trajectory generation and slicing."
    #作用是什么 需要这个吗？
    
    """
    Operations
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        """Reset the commands for the specified environments."""
        # This calls _resample_command internally for the specified env_ids
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        """
        Resample the command (generate a new path) for the specified environments.
        This is called automatically by reset() or when the command time runs out.
        """
        # 1. Generate new trajectories
        new_pos_traj, new_yaw_traj = self._generate_trajectory(env_ids)

        # 2. Update buffers
        self.full_path_pos_w[env_ids] = new_pos_traj
        self.full_path_yaw_w[env_ids] = new_yaw_traj

        # 3. Reset progress index
        self.current_path_index[env_ids] = 0

    def _update_command(self):
        """
        Update the command state at every step.
        This handles tracking progress along the path and computing metrics.
        """
        # 1. Get Robot Position
        robot_pos = self.robot.data.root_pos_w[:, :3].unsqueeze(1) # (N, 1, 3)
        
        # 2. Find the closest point on the path (Simple Search)
        # Optimization: Only search within a local window around the current index
        path_pos = self.full_path_pos_w # (N, num_points, 3)
        
        dist_sq = torch.sum((path_pos - robot_pos)**2, dim=-1) # (N, num_points)
        closest_indices = torch.argmin(dist_sq, dim=1) # (N,)

        # Update current index (ensure monotonicity - robot can't go backwards)
        # Note: Depending on task, you might allow backward movement, but for locomotion usually not.
        self.current_path_index = torch.max(self.current_path_index, closest_indices)

        # 3. Compute Metrics (Error to the closest point)
        min_dist = torch.min(dist_sq, dim=1).values
        self.metrics["error_pos"] = torch.sqrt(min_dist)
        
        # Compute Yaw error (closest point yaw vs robot yaw)
        # Gather the yaw at the closest index
        batch_idx = torch.arange(self.num_envs, device=self.device)
        target_yaw_w = self.full_path_yaw_w[batch_idx, self.current_path_index].squeeze(-1)
        current_yaw_w = yaw_quat(self.robot.data.root_quat_w)
        self.metrics["error_yaw"] = torch.abs(wrap_to_pi(target_yaw_w - current_yaw_w))

        # 4. Visualization
        if self._vis_marker:
            # Visualize a few lookahead points
            lookahead_idx = torch.clamp(self.current_path_index + 10, max=self.num_points - 1)
            target_pos = self.full_path_pos_w[batch_idx, lookahead_idx]
            self._vis_marker.visualize(target_pos)

    """
    Core Logic: Trajectory Generation
    """

    def _generate_trajectory(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trajectory points based on configuration.
        Returns:
            pos_trajectory: (batch, num_points, 3)
            yaw_trajectory: (batch, num_points, 1)
        """
        num_batch = len(env_ids)
        start_pos = self.robot.data.root_pos_w[env_ids].clone()

        # -- Determine End Position --
        end_pos = torch.empty_like(start_pos)
        
        if self.cfg.inpoints.end_to_start_pos is not None:
            # Range relative to start
            offset_x = torch.empty(num_batch, device=self.device).uniform_(*self.cfg.inpoints.end_to_start_pos)
            offset_y = torch.empty(num_batch, device=self.device).uniform_(*self.cfg.inpoints.end_to_start_pos)
            end_pos[:, 0] = start_pos[:, 0] + offset_x
            end_pos[:, 1] = start_pos[:, 1] + offset_y
        else:
            # Default fallback range
            end_pos[:, 0] = start_pos[:, 0] + torch.empty(num_batch, device=self.device).uniform_(-3.0, 3.0)
            end_pos[:, 1] = start_pos[:, 1] + torch.empty(num_batch, device=self.device).uniform_(-3.0, 3.0)

        # Handle Height
        if not self.cfg.inpoints.height_change:
            end_pos[:, 2] = start_pos[:, 2]
        else:
            # TODO: Add logic to sample height from terrain if needed
            end_pos[:, 2] = start_pos[:, 2] # Placeholder

        # -- Time Steps for Interpolation --
        # Shape: (1, num_points, 1) to broadcast
        t = self.t_steps.view(1, -1, 1)

        # -- Position Generation --
        if self.cfg.inpoints.path_type == 'linear':
            pos_trajectory = self._pos_interpolate(start_pos.unsqueeze(1), end_pos.unsqueeze(1), t)
        elif self.cfg.inpoints.path_type == 'bezier':
            pos_trajectory = self._pos_bezerier_interpolate(start_pos.unsqueeze(1), end_pos.unsqueeze(1), t)
        else:
            raise ValueError(f"Unknown path_type: {self.cfg.inpoints.path_type}")

        # -- Yaw Generation --
        if self.cfg.inpoints.yaw_type == 'decoupled':
            # Sample start/end yaw independently
            if self.cfg.inpoints.start_heading is None or self.cfg.inpoints.end_heading is None:
                raise ValueError("Start/End heading ranges required for decoupled yaw.")
            
            # Start Yaw (Relative to robot or Absolute? Usually relative to current or random)
            # Here we sample relative to current yaw or absolute based on cfg logic.
            # Assuming cfg provides absolute ranges or we offset from current.
            # Simplified: Sample from provided range.
            start_yaw = torch.empty(num_batch, device=self.device).uniform_(*self.cfg.inpoints.start_heading)
            end_yaw = torch.empty(num_batch, device=self.device).uniform_(*self.cfg.inpoints.end_heading)
            
            # Interpolate
            yaw_trajectory = self._yaw_interpolate(
                start_yaw.view(num_batch, 1, 1), 
                end_yaw.view(num_batch, 1, 1), 
                t
            )

        elif self.cfg.inpoints.yaw_type == 'along_path':
            if self.cfg.inpoints.path_type == 'linear':
                # Constant yaw for straight line
                yaw_trajectory = self._yaw_along_linear(start_pos, end_pos)
            elif self.cfg.inpoints.path_type == 'bezier':
                # Tangent based yaw
                # Re-compute control points to get derivatives (a bit redundant but clean)
                # Ideally, _pos_bezerier_interpolate should return derivatives too.
                # Here we implement a derivative calculation or finite difference.
                # For simplicity/speed, using finite difference on the generated pos_trajectory
                yaw_trajectory = self._compute_yaw_from_trajectory(pos_trajectory)
        else:
            raise ValueError(f"Unknown yaw_type: {self.cfg.inpoints.yaw_type}")

        return pos_trajectory, yaw_trajectory

    """
    Helper Math Functions
    """

    def _pos_interpolate(self, start_pos, end_pos, alpha) -> torch.Tensor:
        """Linear interpolation: P = A + t * (B - A)"""
        return start_pos + (end_pos - start_pos) * alpha

    def _pos_bezerier_interpolate(self, start_pos, end_pos, t) -> torch.Tensor:
        """Cubic Bezier interpolation."""
        p0 = start_pos
        p3 = end_pos
        
        # Heuristic for Control Points P1, P2
        # P1: Move out along start direction
        # Since we don't have explicit start velocity, we project forward along x-axis or random
        # Better: use current robot heading for P1 to ensure smooth start
        # robot_yaw = yaw_quat(self.robot.data.root_quat_w[env_ids]) ... 
        # For now, using simple offset as in your snippet, but robustified
        
        dist = torch.norm(end_pos - start_pos, dim=-1, keepdim=True)
        control_dist = dist * 0.33
        
        # Simple heuristic: P1 is forward, P2 is backward from end
        p1 = start_pos.clone()
        p1[..., 0] += control_dist[..., 0] # Simple X offset
        
        p2 = end_pos.clone()
        p2[..., 0] -= control_dist[..., 0]

        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    def _yaw_interpolate(self, start_yaw, end_yaw, alpha) -> torch.Tensor:
        """Linear interpolation of Yaw with wrap-around handling."""
        delta_yaw = wrap_to_pi(end_yaw - start_yaw)
        return wrap_to_pi(start_yaw + delta_yaw * alpha)

    def _yaw_along_linear(self, start_pos, end_pos) -> torch.Tensor:
        """Compute constant yaw for a straight line."""
        delta = end_pos - start_pos
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        # Expand to (batch, num_points, 1)
        return yaw.view(-1, 1, 1).expand(-1, self.num_points, 1)

    def _compute_yaw_from_trajectory(self, pos_traj) -> torch.Tensor:
        """Compute yaw from trajectory using finite differences (Approximation for curves)."""
        # Delta = P[t+1] - P[t]
        delta = pos_traj[:, 1:] - pos_traj[:, :-1]
        
        # Pad the last point to keep shape same
        delta = torch.cat([delta, delta[:, -1:]], dim=1)
        
        yaw = torch.atan2(delta[..., 1], delta[..., 0])
        return yaw.unsqueeze(-1)

    """
    Interface for RL Observations (The most important part!)
    """

    def get_path_slice(self) -> torch.Tensor:
        """
        Get a slice of the path relative to the robot's body frame.
        Used as an observation for the RL policy.
        
        Returns:
            slice_b (Tensor): [Num_Envs, Num_Lookahead, 4] -> (x, y, z, yaw_relative)
        """
        window_size = self.num_lookahead_waypoints
        
        # 1. Create indices for slicing
        # From [current_index] to [current_index + window_size]
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, window_size)
        step_indices = torch.arange(window_size, device=self.device).unsqueeze(0) + self.current_path_index.unsqueeze(1)
        
        # Clamp indices to handle end of path (repeat last point)
        step_indices = torch.clamp(step_indices, max=self.num_points - 1)
        
        # 2. Gather World Frame Data
        pos_w = self.full_path_pos_w[batch_indices, step_indices] # (N, W, 3)
        yaw_w = self.full_path_yaw_w[batch_indices, step_indices] # (N, W, 1)
        
        # 3. Transform to Body Frame
        robot_pos = self.robot.data.root_pos_w # (N, 3)
        robot_quat = self.robot.data.root_quat_w # (N, 4)
        
        # Position Transformation: Rotate (Target - Robot) by inverse robot quat
        pos_diff = pos_w - robot_pos.unsqueeze(1)
        # Expand quat to match window size for broadcasting
        quat_expanded = robot_quat.unsqueeze(1).expand(-1, window_size, -1)
        pos_b = quat_rotate_inverse(quat_expanded, pos_diff)
        
        # Yaw Transformation: Relative Yaw = Wrap(Target - Robot)
        robot_yaw = yaw_quat(robot_quat).unsqueeze(1).unsqueeze(2) # (N, 1, 1)
        yaw_b = wrap_to_pi(yaw_w - robot_yaw)
        
        # 4. Concatenate (x, y, z, yaw)
        # Result shape: (N, W, 4)
        slice_b = torch.cat([pos_b, yaw_b], dim=-1)
        
        return slice_b