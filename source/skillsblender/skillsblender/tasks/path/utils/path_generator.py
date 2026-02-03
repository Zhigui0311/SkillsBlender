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
        max_length = self.compute_max_path_length(start_pos, start_yaw, env_bounds)

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
        max_total_length = self.compute_max_path_length(start_pos, start_yaw, env_bounds)

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
        N, W_in, _ = waypoints.shape
        device = waypoints.device

        # 输入索引（0 到 W_in-1）
        input_indices = torch.linspace(0, W_in - 1, W_in, device=device)

        # 输出索引（0 到 W_in-1，均匀分布 num_waypoints 个点）
        output_indices = torch.linspace(0, W_in - 1, num_waypoints, device=device)

        # 线性插值
        interpolated_waypoints = torch.zeros(N, num_waypoints, 3, device=device)
        interpolated_headings = torch.zeros(N, num_waypoints, device=device)

        for i in range(N):
            # 插值位置 (x, y, z)
            interpolated_waypoints[i, :, 0] = torch.interp(output_indices, input_indices, waypoints[i, :, 0])
            interpolated_waypoints[i, :, 1] = torch.interp(output_indices, input_indices, waypoints[i, :, 1])
            interpolated_waypoints[i, :, 2] = torch.interp(output_indices, input_indices, waypoints[i, :, 2])

            # 插值航向（需要处理角度环绕）
            interpolated_headings[i, :] = self._interpolate_angles(
                output_indices, input_indices, headings[i, :]
            )

        return interpolated_waypoints, interpolated_headings

    def _interpolate_angles(
        self,
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """插值角度（处理环绕）"""
        # 将角度转换为复数表示
        complex_angles = torch.complex(torch.cos(angles), torch.sin(angles))

        # 插值实部和虚部
        real_interp = torch.interp(output_indices, input_indices, complex_angles.real)
        imag_interp = torch.interp(output_indices, input_indices, complex_angles.imag)

        # 转换回角度
        interpolated_angles = torch.atan2(imag_interp, real_interp)

        return interpolated_angles
