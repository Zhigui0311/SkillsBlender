from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers #？作用是什么？
from isaaclab.terrains import TerrainImporter #不用引用地形生成的吗？

from isaaclab.utils.math import (quat_apply_inverse,
                                wrap_to_pi,
                                yaw_quat
                                )
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  
    from .path_command_cfg import PathCommandCfg                                      

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

WAYPOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/PathMarkers",
    markers={
        "waypoint_slice": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), 
        ),
        "start": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)), 
        ),
        "goal": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), 
        ),
    }
)


class PathCommand(CommandTerm):
    """Command generator that plan a path and slices it into waypoints for the robot to follow."""
    #TODO: !Implement path planning and waypoint slicing

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        """Configuration for the PathCommand class.
        Args:
            cfg : The configuration parameters for the PathCommand.
            env : The environment the command generator is used in.
        """
        super().__init__(cfg, env) 
        #？这个里面有self.time_left = torch.zeros(self.num_envs, device=self.device)

        #obtain the robot and terrain
        self.robot : Articulation = env.scene[cfg.asset_name]  #提示：这里的到时候要检查是需要的机器人
        # #import terrain if needed
        # self.terrain:TerrainImporter = env.scene[cfg.terrain_name]  #提示：这里的地形对应检查，现在不需要使用

        # -- parameters
        self.num_waypoints = cfg.ranges.num_waypoints #number of interpolation points, also total number of waypoints
        self.num_lookahead_waypoints = cfg.ranges.num_lookahead_waypoints  #number of lookahead waypoints
        self.t_alpha = torch.linspace(0, 1, self.num_waypoints, device=self.device) # alpha values for the waypoints [0, 1], sucn as (0, 0.1, 0.2, ..., 1.0); Constant for all envs

        # -- Observation buffers
        self.obs_slices = torch.zeros(self.num_envs, self.num_lookahead_waypoints, 4, device=self.device)

        # -- command buffers
        self.current_waypoints_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command = torch.zeros(self.num_envs, 4, device=self.device)  # (x, y, z, yaw)
       
        # -- path:(x,y,z,yaw/heading)
        self.pos_path_w = torch.zeros(self.num_envs, self.num_waypoints, 3, device=self.device)
        self.heading_path_w = torch.zeros(self.num_envs, self.num_waypoints, 1, device=self.device) #？！这个角度需要分析一下是相对于什么的角度，还需要转化成四元数之后使用 这个好一点
        # self.pos_path_b = torch.zeros_like(self.pos_path_w)
        # self.heading_path_b = torch.zeros_like(self.heading_path_w) #？这两个量需要吗



        # -- metrics 后面还需要增加一些需要比较的量
        self.metrics["error_pos_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pos_z"] = torch.zeros(self.num_envs, device=self.device)

        # -- Visualization
        #TODO: visualize the planned path and waypoints ?
        self._vis_path = None
        # if self.cfg.debug_vis:
        #     marker_cfg = WAYPOINT_MARKER_CFG
        #     self._vis_path = VisualizationMarkers(marker_cfg)


    @property
    def command(self) -> torch.Tensor:
        """Get the current command (next waypoint) for each environment.
        Returns:
            The current command tensor of shape (num_envs, 4), where each command consists of (x, y, z, yaw).
        """
        return self._command
        # 先看看这个可不可以用？
    
    @property
    def current_alpha(self) -> torch.Tensor:
        """Get the current path progress alpha [0, 1] for each environment.
        Returns:
            The current alpha tensor of shape (num_envs, 1).
        """
        # self.t_alpha  (num_waypoints,)
        # self.current_waypoints_index  (num_envs,)
        current_alpha = self.t_alpha[self.current_waypoints_index] 
        return current_alpha.unsqueeze(1)  # (num_envs, 1)


    # -- Functions
    def _generate_trajectory(self, env_ids:torch.Tensor) -> torch.Tensor:

        """Generate pos trajectory and yaw trajectory based on interpolation points."""
        num_batch = len(env_ids)
        pos_trajectory = torch.zeros((num_batch, self.num_waypoints, 3), device=self.device)
        yaw_trajectory = torch.zeros((num_batch, self.num_waypoints, 1), device=self.device)
        start_pos = self.robot.data.root_pos_w[env_ids].clone()
        end_pos = torch.empty_like(start_pos)
        if self.cfg.inpoints.end_to_start_pos is not None:
            end_pos[:, 0] = start_pos[:, 0] + torch.empty(num_batch, device=self.device).uniform_(
                self.cfg.inpoints.end_to_start_pos[0], self.cfg.inpoints.end_to_start_pos[1])
            end_pos[:, 1] = start_pos[:, 1] + torch.empty(num_batch, device=self.device).uniform_(
                self.cfg.inpoints.end_to_start_pos[0], self.cfg.inpoints.end_to_start_pos[1])
        else:
            end_pos[:, 0] = start_pos[:, 0] + torch.empty(num_batch, device=self.device).uniform_(-5.0, 5.0)
            end_pos[:, 1] = start_pos[:, 1] + torch.empty(num_batch, device=self.device).uniform_(-5.0, 5.0)
        if not self.cfg.inpoints.height_change:
            end_pos[:, 2] = start_pos[:, 2]
        else:
            pass #这里需要改变高度的，需要根据地形设计，感觉可以另外拉一个类来写
        alpha = self.t_alpha.view(1, -1, 1)

        # -- Position generation
        if self.cfg.inpoints.path_type == 'linear':
            # Randomly sample end positions within some range
            pos_trajectory = self._pos_interpolate(start_pos.unsqueeze(1), end_pos.unsqueeze(1), alpha)
        elif self.cfg.inpoints.path_type == 'bezier':
            # Randomly sample end positions within some range
            pos_trajectory = self._pos_bezerier_interpolate(start_pos.unsqueeze(1), end_pos.unsqueeze(1), alpha)
            
        # -- Yaw generation
        if self.cfg.inpoints.yaw_type == 'decoupled':
            # Randomly sample end yaw independently
            if self.cfg.inpoints.start_heading is None and self.cfg.inpoints.end_heading is None:
                raise NotImplementedError("Start and end heading ranges must be specified for decoupled yaw type.")
            else:  
                # Randomly sample start and end yaw within specified ranges
                start_yaw = torch.empty(num_batch, device=self.device).uniform_(self.cfg.inpoints.start_heading[0], self.cfg.inpoints.start_heading[1])
                end_yaw = torch.empty(num_batch, device=self.device).uniform_(self.cfg.inpoints.end_heading[0], self.cfg.inpoints.end_heading[1])                    
            yaw_trajectory = self._yaw_interpolate(start_yaw.unsqueeze(1).unsqueeze(2), end_yaw.unsqueeze(1).unsqueeze(2), alpha)
        elif self.cfg.inpoints.yaw_type == 'along_path':
            # Compute yaw along the path direction
            if self.cfg.inpoints.path_type == 'linear':  
                yaw_trajectory = self._yaw_along_linear(start_pos.unsqueeze(1), end_pos.unsqueeze(1))
            elif self.cfg.inpoints.path_type == 'bezier':
                yaw_trajectory = self._yaw_along_curve(pos_trajectory)
        else:
            raise ValueError(f"Unknown yaw_type: {self.cfg.inpoints.yaw_type}")
            
        return pos_trajectory, yaw_trajectory  #shape: (num_batch, num_points, 3), (num_batch, num_points, 1)


    def _pos_interpolate(self, start_pos, end_pos, alpha) -> torch.Tensor:
        """Interpolate between two positions, and this function just does linear interpolation (for flat terrain).
        Args:
            start: The current position.
            end: The ending position.
            alpha: The interpolation factor.
        Returns:
            The interpolated position.
        """
        return start_pos + (end_pos - start_pos) * alpha


    def _pos_bezerier_interpolate(self, start_pos, end_pos, beta) -> torch.Tensor:
        """Interpolate between two positions using Bezier curve (for uneven terrain).
        Args:
            start: The current position, must be the starting point of uneven terrain.
            end: The ending position.
            beta: The interpolation factor.
        Returns:
            The interpolated position.
        """
        # Define control points for Bezier curve
        p0 = start_pos
        p3 = end_pos
        # Control points p1 and p2 can be defined based on the desired path shape
        p1 = start_pos + torch.tensor([1.0, 0.0, 0.0], device=self.device)  # Example control point
        p2 = end_pos - torch.tensor([1.0, 0.0, 0.0], device=self.device)    # Example control point

        # Compute Bezier point using the cubic Bezier formula
        return (1 - beta) ** 3 * p0 + 3 * (1 - beta) ** 2 * beta * p1 + 3 * (1 - beta) * beta ** 2 * p2 + beta ** 3 * p3
    

    def _yaw_interpolate(self, start_yaw, end_yaw, alpha) -> torch.Tensor:
        """Interpolate between two yaw angles, this suits for decoupled yaw case.
        Args:
            start_yaw: The current yaw angle.
            end_yaw: The ending yaw angle.
            alpha: The interpolation factor.
        Returns:
            The interpolated yaw angle.
        """
        # Ensure the shortest path is taken
        delta_yaw = wrap_to_pi(end_yaw - start_yaw)
        return start_yaw + delta_yaw * alpha
    
    #这个是先转后走的情况，路径上的heading一定是朝着前走的方向的
    def _yaw_along_linear(self, start_pos, end_pos) -> torch.Tensor:
        """Compute yaw for straight path following, this suits for along_path yaw case.
        Args:
            start_pos: The current yaw position.
            end_pos: The ending yaw position.
        Returns:
            The computed yaw angle.
        """
        delta_x = end_pos[:, 0] - start_pos[:, 0]
        delta_y = end_pos[:, 1] - start_pos[:, 1]
        yaw = torch.atan2(delta_y, delta_x)
        yaw_traj =yaw.view(-1,1,1).expand(-1,self.num_waypoints,1)
        return yaw_traj
    
    #看看这个函数里面两种角度的计算方法一不一样
    def _yaw_along_curve(self, pos_traj) -> torch.Tensor:
        """Compute yaw along the path direction for curved paths.
        Args:
            tangent_vectors: The tangent vectors along the path.
        Returns:
            The computed yaw angles.
        """
        # Delta = P[t+1] - P[t]
        delta = pos_traj[:, 1:] - pos_traj[:, :-1]
        # Pad the last point to keep shape same
        delta = torch.cat([delta, delta[:, -1:]], dim=1)

        yaw = torch.atan2(delta[..., 1], delta[..., 0])
        return yaw.unsqueeze(-1)
        # yaw = torch.atan2(tangent_vectors[:, :, 1], tangent_vectors[:, :, 0])
        # return yaw.unsqueeze(-1)
 

    # -- Observation and command functions
    def _get_cur_slices(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get the path slices (waypoints lookahead) for the given environment 
        ids which need to be resampled.
        Args:
            env_ids: The environment IDs to get the waypoint slices for.
        Returns:
            The current waypoint slices for the given environment IDs.
            (N, num_lookahead_waypoints, 4) -> [x, y, z, yaw_rel]
        """
        window_size = self.num_lookahead_waypoints
        
        # 1. Create Slicing Indices
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, window_size) # (N, window_size)
        step_indices = torch.arange(window_size, device=self.device).unsqueeze(0) + self.current_waypoints_index.unsqueeze(1) # (N, window_size)                                                                        
        
        # Clamp to avoid out of bounds (pad with last point)
        step_indices = torch.clamp(step_indices, max=self.num_waypoints - 1)
        
        # 2. Gather World Data   ！这里的数据要在这个函数运行之前更新
        pos_w = self.pos_path_w[batch_indices, step_indices] # (N, window_size, 3) 
        yaw_w = self.heading_path_w[batch_indices, step_indices] # (N, window_size, 1)
        
        # 3. Transform to Body Frame
        robot_pos = self.robot.data.root_pos_w # (N, 3)
        robot_quat = self.robot.data.root_quat_w # (N, 4)
        
        pos_diff = pos_w - robot_pos.unsqueeze(1) # (N, window_size, 3)    caculate position difference in world frame
        quat_expanded = robot_quat.unsqueeze(1).expand(-1, window_size, -1) # (N, window_size, 4)     calculate robot orientation in world frame
        pos_b = quat_apply_inverse(quat_expanded, pos_diff)   # (N, window_size, 3)
        
        robot_yaw_quat = yaw_quat(robot_quat)
        robot_yaw_angle = 2.0*torch.atan2(robot_yaw_quat[:, 3], robot_yaw_quat[:, 0]).unsqueeze(1).unsqueeze(2).expand(-1,window_size,-1)  # (N, window_size, 1)
        yaw_b = wrap_to_pi(yaw_w - robot_yaw_angle)  # (N, window_size, 1)
        
        return torch.cat([pos_b, yaw_b], dim=-1)


    def _resample_command(self, env_ids):
        """Resample the command for the given environment ids."""
        pos_traj, yaw_traj = self._generate_trajectory(env_ids)
        self.pos_path_w[env_ids] = pos_traj
        self.heading_path_w[env_ids] = yaw_traj
        self.current_waypoints_index[env_ids] = 0


    def _update_metrics(self):
        """Update tracking error metrics.Called every step during tracking phase."""
            
        robot_pos = self.robot.data.root_pos_w[:, :3] 
        robot_yaw_quat = yaw_quat(self.robot.data.root_quat_w)
        robot_yaw = 2.0*torch.atan2(robot_yaw_quat[:, 3], robot_yaw_quat[:, 0])  # (N,)

        # 1. Get Target Waypoint (next waypoint)
        target_idx = torch.clamp(self.current_waypoints_index + 1, max=self.num_waypoints - 1)            
        target_pos_w = self.pos_path_w[torch.arange(self.num_envs), target_idx] # (N, 3)
        target_yaw_w = self.heading_path_w[torch.arange(self.num_envs), target_idx].squeeze(-1) # (N,)

        # 2. Compute Errors
        self.metrics["error_pos_xy"] = torch.norm(target_pos_w[:, :2] - robot_pos[:, :2], dim=1)
        self.metrics["error_pos_z"] = torch.abs(target_pos_w[:, 2] - robot_pos[:, 2])
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(target_yaw_w - robot_yaw))


    def _update_command(self):
        """
        Update the command based on the current robot state and path. 
        """
        robot_pos = self.robot.data.root_pos_w[:, :3] # (N, 3)
        # robot_quat = self.robot.data.root_quat_w
        target_pos = self.pos_path_w[torch.arange(self.num_envs), self.current_waypoints_index]  # (N, 3)

        dis_to_target = torch.norm(target_pos - robot_pos, dim=-1)  # (N,)
        reach_threshold = self.cfg.ranges.waypoint_reach_threshold
        reached = dis_to_target < reach_threshold
        self.current_waypoints_index = torch.where(
            reached,
            torch.clamp(self.current_waypoints_index + 1, max=self.num_waypoints - 1),
            self.current_waypoints_index
        )

        self.obs_slices = self._get_cur_slices(torch.arange(self.num_envs, device=self.device))
        self._command = self.obs_slices[:, 0, :]  # (N, 4) -> first waypoint in the slice


        # # 1. Track Progress: Find closest point index
        # # Optimization: Search full path (robust) or local window (fast). 
        # dists = torch.norm(self.pos_path_w - robot_pos.unsqueeze(1), dim=-1)  # (N, num_waypoints)
        # closest_indices = torch.argmin(dists, dim=1)  # (N,)
        
        # # Ensure progress is monotonic (cannot go backward)
        # self.current_waypoints_index = torch.max(self.current_waypoints_index, closest_indices) #! 这里不太对吧 current_waypoints_index 应该也要慢慢增加吧
        
        # # 2. Update Observations: Get waypoint slices in body frame
        # self.obs_slices = self._get_cur_slices(torch.arange(self.num_envs, device=self.device))
        # self._command = self.obs_slices[:, 0, :]  # (N, 4) -> first waypoint in the slice

        # # ? 画出可视化的路径
        # # # 3. Visualize Dense Path (Local view)
        # # if self._vis_path:
        # #     # 只画第一个环境的路径，避免卡顿
        # #     env_0_idx = 0
        # #     # 画出当前索引之后的所有点
        # #     start_idx = self.current_waypoints_index[env_0_idx].item()
        # #     points = self.pos_path_w[env_0_idx, start_idx:]
        # #     self._vis_path.visualize(points, marker_type="waypoint")


