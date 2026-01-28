from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
import isaaclab.utils.warp as warp_utils
import isaaclab.utils.math as math_utils

from skillsblender.tasks.path.mdp.commands.path_command import PathCommand
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .path_command_cfg import JumpPathCommandCfg

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

class JumpPathCommand(PathCommand):
    """Command that senses terrain to generate parabolic jump trajectories."""
    cfg: JumpPathCommandCfg

    def __init__(self, cfg: JumpPathCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the JumpPathCommand and create warp mesh for terrain queries.

        Args:
            cfg: The configuration for the JumpPathCommand.
            env: The environment the command generator is used in.
        """
        # Call parent class initialization
        super().__init__(cfg, env)

        self.height_scanner = env.scene.sensors.get("height_scanner")
        # Get terrain configuration and recreate TerrainGenerator to access mesh
        terrain_importer: TerrainImporter = env.scene.terrain
        terrain_cfg = terrain_importer.cfg.terrain_generator

        # Recreate terrain generator to get the terrain mesh
        # (TerrainImporter doesn't save the mesh after initialization)
        from isaaclab.terrains import TerrainGenerator
        terrain_generator = TerrainGenerator(cfg=terrain_cfg, device=self.device)

        # Convert trimesh to warp mesh for GPU-accelerated raycasting
        self.warp_mesh = warp_utils.convert_to_warp_mesh(
            points=terrain_generator.terrain_mesh.vertices,
            indices=terrain_generator.terrain_mesh.faces,
            device=self.device
        )

    def _generate_trajectory(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate position and yaw trajectories with parabolic arcs over terrain gaps.

        This function:
        1. Scans terrain ahead to detect gaps
        2. Plans a path that includes jumps over detected gaps
        3. Generates parabolic arcs for jump segments
        4. Computes appropriate yaw angles along the path
        """
        num_batch = len(env_ids)
        start_pos = self.robot.data.root_pos_w[env_ids].clone()

        # Clamp start position to env bounds
        xy_min, xy_max = self._get_env_xy_bounds(env_ids)
        if xy_min is not None:
            start_pos[:, :2] = torch.max(torch.min(start_pos[:, :2], xy_max), xy_min)

        # 1. Terrain Scanning (Detect gaps by scanning forward)
        gap_start_dist = torch.zeros(num_batch, device=self.device)
        gap_end_dist = torch.zeros(num_batch, device=self.device)
        has_gap = torch.zeros(num_batch, dtype=torch.bool, device=self.device)

        base_h = start_pos[:, 2]
        steps = int(self.cfg.jump_params.scan_dist / self.cfg.jump_params.scan_step)

        # for i in range(steps):
        #     d = i * self.cfg.jump_params.scan_step
        #     check_xy = start_pos[:, :2].clone()
        #     check_xy[:, 0] += d  # Scan forward along X-axis

        #     # Clamp to env bounds
        #     if xy_min is not None:
        #         check_xy = torch.max(torch.min(check_xy, xy_max), xy_min)

        #     # Query terrain height at this position using warp raycast
        #     # Create rays starting from high above (100m) and pointing downward
        #     ray_starts = torch.cat([
        #         check_xy,  # XY position
        #         torch.ones(num_batch, 1, device=self.device) * 100.0  # Z = 100m above
        #     ], dim=-1)  # Shape: (num_batch, 3)

        #     # Ray direction: straight down (0, 0, -1)
        #     ray_directions = torch.tensor(
        #         [[0.0, 0.0, -1.0]],
        #         device=self.device
        #     ).expand(num_batch, -1)  # Shape: (num_batch, 3)

        #     # Perform raycast to find ground height
        #     ray_hits, _, _, _ = warp_utils.raycast_mesh(
        #         ray_starts=ray_starts,
        #         ray_directions=ray_directions,
        #         mesh=self.warp_mesh,
        #         max_dist=200.0  # Max raycast distance (100m down)
        #     )

        #     # Extract Z coordinate as terrain height
        #     # If ray misses (returns inf), use base height as fallback
        #     h = torch.where(
        #         torch.isinf(ray_hits[:, 2]),
        #         base_h,  # Fallback to base height if raycast misses
        #         ray_hits[:, 2]  # Use raycast Z coordinate
        #     )

        #     # Detect gap start (height drops)
        #     is_gap = (h - base_h) < self.cfg.jump_params.gap_threshold

        #     # Mark gap start
        #     start_mask = is_gap & (~has_gap)
        #     if start_mask.any():
        #         gap_start_dist[start_mask] = d - self.cfg.jump_params.takeoff_margin
        #         has_gap |= start_mask

        #     # Mark gap end (height recovers)
        #     end_mask = (~is_gap) & has_gap & (gap_end_dist == 0)
        #     if end_mask.any():
        #         gap_end_dist[end_mask] = d + self.cfg.jump_params.landing_margin
        if self.height_scanner is not None:
            ray_hits_w = self.height_scanner.data.ray_hits_w[env_ids]
            robot_pos_w = start_pos[:, :3]
            robot_quat_w = self.robot.data.root_quat_w[env_ids]
            rel_hits = ray_hits_w - robot_pos_w.unsqueeze(1)
            rel_hits_flat = rel_hits.reshape(-1, 3)
            quat_flat = robot_quat_w.repeat_interleave(rel_hits.shape[1], dim=0)
            rel_hits_b = math_utils.quat_apply_inverse(quat_flat, rel_hits_flat).reshape(rel_hits.shape)
            rel_x = rel_hits_b[..., 0]
            rel_y = rel_hits_b[..., 1]
            rel_z = ray_hits_w[..., 2]        

            valid_mask = (
                (rel_x >= 0.0)
                & (rel_x <= self.cfg.jump_params.scan_dist)
                & (torch.abs(rel_y) < 0.05)
            )
                
            for env_i in range(num_batch):
                mask = valid_mask[env_i]
                if not mask.any():
                    continue
                x_vals = rel_x[env_i, mask]
                h_vals = rel_z[env_i, mask]
                order = torch.argsort(x_vals)
                x_vals = x_vals[order]
                h_vals = h_vals[order]
                is_gap = (h_vals - base_h[env_i]) < self.cfg.jump_params.gap_threshold
                if not is_gap.any():
                    continue
                first_gap_idx = torch.nonzero(is_gap, as_tuple=False)[0, 0]
                # Use configured takeoff margin based on gap width (will be determined later)
                # For now, use narrow gap margin as default during detection
                gap_start_dist[env_i] = x_vals[first_gap_idx] - self.cfg.jump_params.narrow_gap_takeoff_margin
                has_gap[env_i] = True
                end_candidates = torch.nonzero(~is_gap & (x_vals > x_vals[first_gap_idx]), as_tuple=False)
                if end_candidates.numel() > 0:
                    end_idx = end_candidates[0, 0]
                    gap_end_dist[env_i] = x_vals[end_idx] + self.cfg.jump_params.landing_margin    
        else:
            for i in range(steps):
                d = i * self.cfg.jump_params.scan_step
                check_xy = start_pos[:, :2].clone()
                check_xy[:, 0] += d  # Scan forward along X-axis

                # Clamp to env bounds
                if xy_min is not None:
                    check_xy = torch.max(torch.min(check_xy, xy_max), xy_min)

                # Query terrain height at this position using warp raycast
                # Create rays starting from high above (100m) and pointing downward
                ray_starts = torch.cat([
                    check_xy,  # XY position
                    torch.ones(num_batch, 1, device=self.device) * 100.0  # Z = 100m above
                ], dim=-1)  # Shape: (num_batch, 3)

                # Ray direction: straight down (0, 0, -1)
                ray_directions = torch.tensor(
                    [[0.0, 0.0, -1.0]],
                    device=self.device
                ).expand(num_batch, -1)  # Shape: (num_batch, 3)

                # Perform raycast to find ground height
                ray_hits, _, _, _ = warp_utils.raycast_mesh(
                    ray_starts=ray_starts,
                    ray_directions=ray_directions,
                    mesh=self.warp_mesh,
                    max_dist=200.0  # Max raycast distance (100m down)
                )

                # Extract Z coordinate as terrain height
                # If ray misses (returns inf), use base height as fallback
                h = torch.where(
                    torch.isinf(ray_hits[:, 2]),
                    base_h,  # Fallback to base height if raycast misses
                    ray_hits[:, 2]  # Use raycast Z coordinate
                )

                # Detect gap start (height drops)
                is_gap = (h - base_h) < self.cfg.jump_params.gap_threshold

                # Mark gap start
                start_mask = is_gap & (~has_gap)
                if start_mask.any():
                    # Use configured takeoff margin (narrow gap margin as default during detection)
                    gap_start_dist[start_mask] = d - self.cfg.jump_params.narrow_gap_takeoff_margin
                    has_gap |= start_mask

                # Mark gap end (height recovers)
                end_mask = (~is_gap) & has_gap & (gap_end_dist == 0)
                if end_mask.any():
                    gap_end_dist[end_mask] = d + self.cfg.jump_params.landing_margin
                    
                    
        # 2. Plan 3D Path
        # Calculate gap width to determine endpoint extension and adjust takeoff margins
        gap_width = gap_end_dist - gap_start_dist
        # Classify gaps using configured threshold
        is_narrow_gap = gap_width < self.cfg.jump_params.gap_width_threshold

        # Adjust takeoff margins based on gap classification
        # (Initial detection used narrow margin, now refine based on actual gap width)
        takeoff_margin_adjustment = torch.where(
            is_narrow_gap,
            torch.tensor(0.0, device=self.device),  # Already using narrow margin
            self.cfg.jump_params.wide_gap_takeoff_margin - self.cfg.jump_params.narrow_gap_takeoff_margin
        )
        # Apply adjustment only where gaps exist
        gap_start_dist = torch.where(
            has_gap,
            gap_start_dist - takeoff_margin_adjustment,
            gap_start_dist
        )

        # Endpoint extension: use configured values for narrow/wide gaps
        endpoint_extension = torch.where(
            is_narrow_gap,
            self.cfg.jump_params.narrow_gap_endpoint_extension,
            self.cfg.jump_params.wide_gap_endpoint_extension
        )

        # Total distance: extend beyond gap end if gap exists, otherwise use default length
        total_len = torch.where(
            has_gap & (gap_end_dist > 0),
            gap_end_dist + endpoint_extension,
            torch.tensor(5.0, device=self.device)
        )

        # Apply optional post-jump distance extension
        if self.cfg.jump_params.post_jump_distance > 0:
            total_len = torch.where(
                has_gap & (gap_end_dist > 0),
                total_len + self.cfg.jump_params.post_jump_distance,
                total_len
            )

        end_pos = start_pos.clone()
        end_pos[:, 0] += total_len

        # Clamp end position to env bounds
        if xy_min is not None:
            end_pos[:, :2] = torch.max(torch.min(end_pos[:, :2], xy_max), xy_min)

        # 3. Generate linear interpolation as base trajectory
        alpha = self.t_alpha.view(1, -1, 1)
        pos_traj = start_pos.unsqueeze(1) + (end_pos.unsqueeze(1) - start_pos.unsqueeze(1)) * alpha

        # 4. Apply Parabolic Arc over detected gaps
        # Calculate distance of each waypoint from start
        dist_at_wp = alpha.squeeze(-1) * total_len.unsqueeze(1)

        # Normalize position within jump segment [0, 1]
        t_jump = (dist_at_wp - gap_start_dist.unsqueeze(1)) / \
                 (gap_end_dist.unsqueeze(1) - gap_start_dist.unsqueeze(1) + 1e-6)
        t_jump = torch.clamp(t_jump, 0.0, 1.0)

        # Parabolic arc: h = 4 * H * t * (1-t) reaches max H at t=0.5
        arc_h = self.cfg.jump_params.jump_height * 4 * t_jump * (1 - t_jump)

        # Apply arc only within jump segment
        jump_mask = (dist_at_wp > gap_start_dist.unsqueeze(1)) & \
                   (dist_at_wp < gap_end_dist.unsqueeze(1)) & \
                   has_gap.unsqueeze(1)

        pos_traj[..., 2] += torch.where(jump_mask, arc_h, torch.zeros_like(arc_h))

        # 5. Compute yaw along the path
        yaw_traj = self._yaw_along_curve(pos_traj)

        return pos_traj, yaw_traj

    def _update_command(self):
        """
        Override: 更新航点索引 - 跳跃任务需要特殊处理空中情况。

        关键改进：
        - 检测机器人是否在空中
        - 在空中时只考虑XY平面距离，避免航点推进过快
        - 在地面时使用正常的3D距离判断
        - 在跳跃阶段使用更严格的航点阈值
        """
        robot_pos = self.robot.data.root_pos_w[:, :3]  # (N, 3)
        robot_height = robot_pos[:, 2]

        # 检测是否在空中（高于站立高度+阈值）
        # GO2站立高度约0.34m，离地0.15m以上认为在空中
        standing_height = 0.34
        air_threshold = 0.15
        is_in_air = robot_height > (standing_height + air_threshold)

        # 获取当前目标航点
        target_pos_cur = self.pos_path_w[torch.arange(self.num_envs), self.current_waypoints_index]  # (N, 3)

        # 计算到目标的距离
        # 在空中时只考虑XY平面，在地面时考虑3D距离
        dis_xy = torch.norm(target_pos_cur[:, :2] - robot_pos[:, :2], dim=-1)  # XY平面距离
        dis_3d = torch.norm(target_pos_cur - robot_pos, dim=-1)  # 3D距离

        # 选择距离判断方式
        dis_to_target = torch.where(is_in_air, dis_xy, dis_3d)

        # 自适应航点阈值：跳跃阶段使用0.5m，其他阶段使用0.8m
        # 判断是否在跳跃阶段：检查当前航点是否在跳跃段内
        # 这需要访问gap_start_dist和gap_end_dist，但它们在_generate_trajectory中
        # 作为简化，我们使用is_in_air作为跳跃阶段的代理
        base_threshold = self.cfg.ranges.waypoint_reach_threshold  # 默认0.8m
        jump_threshold = 0.5  # 跳跃阶段的严格阈值
        reach_threshold = torch.where(is_in_air, jump_threshold, base_threshold)

        # 到达阈值判断
        reached = dis_to_target < reach_threshold

        # 更新航点索引
        self.current_waypoints_index = torch.where(
            reached,
            torch.clamp(self.current_waypoints_index + 1, max=self.num_waypoints - 1),
            self.current_waypoints_index
        )

        # 检查是否到达终点
        target_pos_final = self.pos_path_w[torch.arange(self.num_envs), -1]
        goal_dis = torch.norm(target_pos_final - robot_pos, dim=-1)
        goal_reached = (goal_dis < base_threshold) & (self.current_waypoints_index >= self.num_waypoints - 1)
        self.goal_reached = self.goal_reached | goal_reached

        # 更新观测切片
        self.obs_slices = self._get_cur_slices(torch.arange(self.num_envs, device=self.device))

        # 到达终点时，将观测切片清零
        obs_slices = self.obs_slices.clone()
        if torch.any(goal_reached):
            obs_slices[goal_reached] = 0.0
        self.obs_slices = obs_slices