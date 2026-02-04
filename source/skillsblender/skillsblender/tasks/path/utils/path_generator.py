"""地形感知路径生成器

该模块实现了统一的路径生成器，能够：
1. 根据地形类型和技能序列生成路径
2. 管理技能路径规划器（SkillPathPlanner）
3. 计算环境边界约束
4. 生成单技能或多技能路径
5. 验证生成的路径
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from skillsblender.tasks.path.mdp.commands.path_command_cfg import PathGeneratorCfg
from skillsblender.tasks.path.utils.skill_path_planners import (
    SkillPathPlanner,
    WalkPathPlanner,
    JumpPathPlanner,
    StairsPathPlanner,
    ClimbPathPlanner,
    CrouchPathPlanner,
    SegmentPlan,
)
from skillsblender.tasks.path.utils.path_validator import PathValidator, PathResult


class TerrainAwarePathGenerator:
    """统一的路径生成器，根据地形类型和技能序列生成路径

    该类负责：
    1. 管理技能路径规划器（SkillPathPlanner）
    2. 计算环境边界约束
    3. 生成单技能或多技能路径
    4. 验证生成的路径
    """

    def __init__(self, cfg: PathGeneratorCfg, device: torch.device):
        """初始化路径生成器

        Args:
            cfg: 路径生成器配置
            device: PyTorch 设备（cuda 或 cpu）
        """
        self.cfg = cfg
        self.device = device

        # 注册技能路径规划器
        self.skill_planners: Dict[str, SkillPathPlanner] = {
            "walk": WalkPathPlanner(cfg.walk_params, device),
            "jump": JumpPathPlanner(cfg.jump_params, device),
            "stairs_up": StairsPathPlanner(cfg.stairs_params, device, direction="up"),
            "stairs_down": StairsPathPlanner(cfg.stairs_params, device, direction="down"),
            "climb": ClimbPathPlanner(cfg.climb_params, device),
            "crouch": CrouchPathPlanner(cfg.crouch_params, device),
        }

        # 路径验证器
        self.validator = PathValidator(cfg.validation_params, device)

        # 航点插值器
        self.num_waypoints = cfg.num_waypoints
        self.waypoint_spacing = cfg.waypoint_spacing

    def generate_path(
        self,
        env_ids: torch.Tensor,
        skill_name: str,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        env_bounds: torch.Tensor,
        terrain_data: Optional[Dict[str, torch.Tensor]] = None,
        path_length: Optional[torch.Tensor] = None,
    ) -> PathResult:
        """生成单技能路径

        Args:
            env_ids: 环境 ID，shape (N,)
            skill_name: 技能名称（"walk", "jump" 等）
            start_pos: 起始位置，shape (N, 3)，世界坐标系
            start_yaw: 起始航向角，shape (N,)，单位：弧度
            env_bounds: 环境边界，shape (N, 4)，格式：[x_min, x_max, y_min, y_max]
            terrain_data: 地形数据字典，包含：
                - "height_scanner": RayCaster 扫描结果（用于 jump）
                - "terrain_type": 地形类型（用于自动技能选择）

        Returns:
            PathResult: 路径生成结果
        """
        N = len(env_ids)
        device = start_pos.device

        # 1. 计算最大路径长度（环境边界约束）
        if path_length is None:
            max_length = self.compute_max_path_length(start_pos, start_yaw, env_bounds)
        else:
            max_length = path_length.to(start_pos.device)

        # 2. 获取技能路径规划器
        if skill_name not in self.skill_planners:
            raise ValueError(f"Unknown skill: {skill_name}. Available skills: {list(self.skill_planners.keys())}")
        planner = self.skill_planners[skill_name]

        # 3. 规划路径段
        segment_plan = planner.plan_segment(
            env_ids=env_ids,
            start_pos=start_pos,
            start_yaw=start_yaw,
            terrain_data=terrain_data or {},
            segment_length=max_length,
        )

        # 4. 插值生成航点
        waypoints, headings = self._interpolate_waypoints(
            segment_plan.waypoints,
            segment_plan.headings,
            num_waypoints=self.num_waypoints,
        )

        # 5. 构建 PathResult
        path_result = PathResult(
            waypoints=waypoints,
            headings=headings,
            num_segments=torch.ones(N, dtype=torch.long, device=device),
            segment_skills=torch.full((N, 1), segment_plan.skill_id, dtype=torch.long, device=device),
            segment_s0=torch.zeros(N, 1, device=device),
            segment_s1=segment_plan.segment_length.unsqueeze(-1),
            segment_params=segment_plan.segment_params.unsqueeze(1),
            path_length=segment_plan.segment_length,
            valid=segment_plan.valid,
        )

        # 6. 验证路径
        validation_result = self.validator.validate(path_result)
        path_result.valid = path_result.valid & validation_result.valid

        return path_result

    def generate_multi_skill_path(
        self,
        env_ids: torch.Tensor,
        skill_sequence: List[str],
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        env_bounds: torch.Tensor,
        terrain_data: Optional[Dict[str, torch.Tensor]] = None,
        path_length: Optional[torch.Tensor] = None,
    ) -> PathResult:
        """生成多技能序列路径

        Args:
            env_ids: 环境 ID，shape (N,)
            skill_sequence: 技能序列，如 ["walk", "jump", "walk"]
            start_pos: 起始位置，shape (N, 3)
            start_yaw: 起始航向角，shape (N,)
            env_bounds: 环境边界，shape (N, 4)
            terrain_data: 地形数据字典

        Returns:
            PathResult: 路径生成结果
        """
        N = len(env_ids)
        device = start_pos.device

        # 1. 计算总的最大路径长度
        if path_length is None:
            max_total_length = self.compute_max_path_length(start_pos, start_yaw, env_bounds)
        else:
            max_total_length = path_length.to(start_pos.device)

        # 2. 为每个技能分配长度（均匀分配）
        num_skills = len(skill_sequence)
        segment_lengths = max_total_length / num_skills

        # 3. 依次规划每个技能段
        all_segments = []
        current_pos = start_pos.clone()
        current_yaw = start_yaw.clone()
        current_s = torch.zeros(N, device=device)

        for skill_name in skill_sequence:
            if skill_name not in self.skill_planners:
                raise ValueError(f"Unknown skill: {skill_name}")
            planner = self.skill_planners[skill_name]

            # 规划当前技能段
            segment_plan = planner.plan_segment(
                env_ids=env_ids,
                start_pos=current_pos,
                start_yaw=current_yaw,
                terrain_data=terrain_data or {},
                segment_length=segment_lengths,
            )

            # 记录 segment 信息
            all_segments.append({
                "skill_id": segment_plan.skill_id,
                "s0": current_s.clone(),
                "s1": current_s + segment_plan.segment_length,
                "params": segment_plan.segment_params,
                "waypoints": segment_plan.waypoints,
                "headings": segment_plan.headings,
            })

            # 更新当前位置和航向（下一个 segment 的起点）
            current_pos = segment_plan.waypoints[:, -1, :]  # 最后一个航点
            current_yaw = segment_plan.headings[:, -1]  # 最后一个航向
            current_s = current_s + segment_plan.segment_length

        # 4. 拼接所有 segment 的航点
        all_waypoints = []
        all_headings = []
        for seg in all_segments:
            all_waypoints.append(seg["waypoints"])
            all_headings.append(seg["headings"])

        # 拼接并重新插值到固定数量的航点
        concatenated_waypoints = torch.cat(all_waypoints, dim=1)  # (N, W_total, 3)
        concatenated_headings = torch.cat(all_headings, dim=1)  # (N, W_total)

        waypoints, headings = self._interpolate_waypoints(
            concatenated_waypoints,
            concatenated_headings,
            num_waypoints=self.num_waypoints,
        )

        # 5. 构建 PathResult
        num_segments = len(skill_sequence)
        segment_skills = torch.stack([torch.full((N,), seg["skill_id"], dtype=torch.long, device=device) for seg in all_segments], dim=1)
        segment_s0 = torch.stack([seg["s0"] for seg in all_segments], dim=1)
        segment_s1 = torch.stack([seg["s1"] for seg in all_segments], dim=1)
        segment_params = torch.stack([seg["params"] for seg in all_segments], dim=1)

        path_result = PathResult(
            waypoints=waypoints,
            headings=headings,
            num_segments=torch.full((N,), num_segments, dtype=torch.long, device=device),
            segment_skills=segment_skills,
            segment_s0=segment_s0,
            segment_s1=segment_s1,
            segment_params=segment_params,
            path_length=current_s,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

        # 6. 验证路径
        validation_result = self.validator.validate(path_result)
        path_result.valid = path_result.valid & validation_result.valid

        return path_result

    def compute_max_path_length(
        self,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        env_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """计算在给定方向上的最大路径长度（考虑环境边界）

        Args:
            start_pos: 起始位置，shape (N, 3)
            start_yaw: 起始航向角，shape (N,)
            env_bounds: 环境边界，shape (N, 4) [x_min, x_max, y_min, y_max]

        Returns:
            torch.Tensor: 最大路径长度，shape (N,)
        """
        N = start_pos.shape[0]
        device = start_pos.device

        # 前向方向向量
        forward_dir = torch.stack([
            torch.cos(start_yaw),
            torch.sin(start_yaw),
        ], dim=-1)  # (N, 2)

        # 起始位置 (x, y)
        start_xy = start_pos[:, :2]  # (N, 2)

        # 计算与四条边界的交点距离
        x_min, x_max, y_min, y_max = env_bounds.unbind(dim=-1)

        # 与 x_min 边界的交点
        t_x_min = (x_min - start_xy[:, 0]) / (forward_dir[:, 0] + 1e-6)

        # 与 x_max 边界的交点
        t_x_max = (x_max - start_xy[:, 0]) / (forward_dir[:, 0] + 1e-6)

        # 与 y_min 边界的交点
        t_y_min = (y_min - start_xy[:, 1]) / (forward_dir[:, 1] + 1e-6)

        # 与 y_max 边界的交点
        t_y_max = (y_max - start_xy[:, 1]) / (forward_dir[:, 1] + 1e-6)

        # 只保留正距离（前方的交点）
        t_all = torch.stack([t_x_min, t_x_max, t_y_min, t_y_max], dim=-1)  # (N, 4)
        t_all = torch.where(t_all > 0, t_all, torch.full_like(t_all, float('inf')))

        # 最小的正距离就是最大路径长度
        max_length = torch.min(t_all, dim=-1).values  # (N,)

        # 减去安全边界
        max_length = max_length - self.cfg.env_bounds.boundary_margin
        max_length = torch.clamp(max_length, min=1.0)  # 至少 1m

        return max_length

    def _interpolate_waypoints(
        self,
        waypoints: torch.Tensor,  # (N, W_in, 3)
        headings: torch.Tensor,  # (N, W_in)
        num_waypoints: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将航点插值到固定数量

        Args:
            waypoints: 输入航点，shape (N, W_in, 3)
            headings: 输入航向，shape (N, W_in)
            num_waypoints: 目标航点数量

        Returns:
            interpolated_waypoints: shape (N, num_waypoints, 3)
            interpolated_headings: shape (N, num_waypoints)
        """
        # Waypoints: (N, W_in, 3) -> (N, 3, W_in)
        waypoints_t = waypoints.transpose(1, 2)
        interpolated_waypoints = F.interpolate(
            waypoints_t, size=num_waypoints, mode="linear", align_corners=True
        ).transpose(1, 2)

        # Headings: fallback from input (unit-circle interpolation)
        heading_vec = torch.stack([torch.cos(headings), torch.sin(headings)], dim=1)  # (N, 2, W_in)
        heading_vec_i = F.interpolate(
            heading_vec, size=num_waypoints, mode="linear", align_corners=True
        )
        fallback_headings = torch.atan2(heading_vec_i[:, 1], heading_vec_i[:, 0])  # (N, W)

        # Headings: derive from waypoint direction (auto-follow path)
        if num_waypoints <= 1:
            return interpolated_waypoints, fallback_headings[:, :1]

        diff = interpolated_waypoints[:, 1:, :2] - interpolated_waypoints[:, :-1, :2]  # (N, W-1, 2)
        heading_dir = torch.atan2(diff[..., 1], diff[..., 0])  # (N, W-1)
        last = heading_dir[:, -1:]
        interpolated_headings = torch.cat([heading_dir, last], dim=1)  # (N, W)

        # If movement is too small, fall back to input headings to avoid NaNs/jitter
        dist = torch.linalg.norm(diff, dim=-1)  # (N, W-1)
        moving = dist > 1e-4
        moving = torch.cat([moving, moving[:, -1:]], dim=1)
        interpolated_headings = torch.where(moving, interpolated_headings, fallback_headings)

        return interpolated_waypoints, interpolated_headings

    def _interpolate_angles(
        self,
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """插值角度（处理环绕）"""
        # Deprecated: kept for compatibility; prefer _interpolate_waypoints path.
        complex_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0).unsqueeze(0)
        complex_interp = F.interpolate(
            complex_angles, size=output_indices.numel(), mode="linear", align_corners=True
        ).squeeze(0)
        return torch.atan2(complex_interp[1], complex_interp[0])
