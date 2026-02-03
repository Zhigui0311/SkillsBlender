"""路径验证器

该模块负责验证生成的路径是否满足物理约束和技能要求，包括：
- 路径连续性验证（相邻航点距离）
- 航向连续性验证（相邻航向角变化）
- 高度变化验证（爬升/下降角度）
- Segment 长度验证
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Dict

from isaaclab.utils.math import wrap_to_pi
from skillsblender.tasks.path.mdp.commands.path_command_cfg import ValidationParams


@dataclass
class ValidationResult:
    """路径验证结果"""
    valid: torch.Tensor  # (N,) bool，路径是否有效
    errors: Dict[str, torch.Tensor]  # 各项验证的错误信息
    warnings: Dict[str, torch.Tensor]  # 各项验证的警告信息


@dataclass
class PathResult:
    """完整路径生成结果"""
    waypoints: torch.Tensor  # (N, W, 3) 路径航点 (x, y, z)，世界坐标系
    headings: torch.Tensor   # (N, W) 航向角 (yaw)，单位：弧度
    num_segments: torch.Tensor  # (N,) segment 数量
    segment_skills: torch.Tensor  # (N, S) 每个 segment 的技能 ID
    segment_s0: torch.Tensor  # (N, S) segment 起始位置（沿路径的距离）
    segment_s1: torch.Tensor  # (N, S) segment 结束位置
    segment_params: torch.Tensor  # (N, S, P) segment 参数
    path_length: torch.Tensor  # (N,) 总路径长度
    valid: torch.Tensor  # (N,) bool，路径是否有效


class PathValidator:
    """路径验证器"""

    def __init__(self, cfg: ValidationParams, device: torch.device):
        """初始化验证器

        Args:
            cfg: 验证参数配置
            device: PyTorch 设备
        """
        self.cfg = cfg
        self.device = device

    def validate(self, path_result: PathResult) -> ValidationResult:
        """验证路径

        Args:
            path_result: 路径生成结果

        Returns:
            ValidationResult: 验证结果
        """
        N = path_result.waypoints.shape[0]
        device = path_result.waypoints.device

        errors = {}
        warnings = {}

        # 1. 验证路径连续性
        continuity_valid = self.validate_continuity(path_result.waypoints)
        errors["continuity"] = ~continuity_valid

        # 2. 验证航向连续性
        heading_valid = self.validate_heading_continuity(path_result.headings)
        errors["heading_continuity"] = ~heading_valid

        # 3. 验证高度变化
        height_valid = self.validate_height_change(path_result.waypoints)
        errors["height_change"] = ~height_valid

        # 4. 验证 segment 长度
        segment_valid = self.validate_segment_length(
            path_result.segment_s0,
            path_result.segment_s1,
        )
        errors["segment_length"] = ~segment_valid

        # 5. 综合判断
        all_valid = continuity_valid & heading_valid & height_valid & segment_valid

        return ValidationResult(
            valid=all_valid,
            errors=errors,
            warnings=warnings,
        )

    def validate_continuity(self, waypoints: torch.Tensor) -> torch.Tensor:
        """验证路径连续性

        Args:
            waypoints: 路径航点，shape (N, W, 3)

        Returns:
            torch.Tensor: (N,) bool，True 表示连续性良好
        """
        N, W, _ = waypoints.shape
        device = waypoints.device

        # 计算相邻航点之间的距离
        diffs = waypoints[:, 1:, :] - waypoints[:, :-1, :]  # (N, W-1, 3)
        distances = torch.norm(diffs, dim=-1)  # (N, W-1)

        # 检查是否有距离超过阈值
        max_distances = distances.max(dim=-1).values  # (N,)
        valid = max_distances <= self.cfg.max_waypoint_distance

        return valid

    def validate_heading_continuity(self, headings: torch.Tensor) -> torch.Tensor:
        """验证航向连续性

        Args:
            headings: 航向角，shape (N, W)

        Returns:
            torch.Tensor: (N,) bool，True 表示连续性良好
        """
        N, W = headings.shape
        device = headings.device

        # 计算相邻航向角的变化（考虑角度环绕）
        heading_diffs = headings[:, 1:] - headings[:, :-1]  # (N, W-1)
        heading_diffs = wrap_to_pi(heading_diffs)  # 归一化到 [-pi, pi]

        # 检查是否有变化超过阈值
        max_heading_changes = torch.abs(heading_diffs).max(dim=-1).values  # (N,)
        valid = max_heading_changes <= self.cfg.max_heading_change

        return valid

    def validate_height_change(self, waypoints: torch.Tensor) -> torch.Tensor:
        """验证高度变化合理性

        Args:
            waypoints: 路径航点，shape (N, W, 3)

        Returns:
            torch.Tensor: (N,) bool，True 表示高度变化合理
        """
        N, W, _ = waypoints.shape
        device = waypoints.device

        # 计算相邻航点的水平距离和高度差
        diffs = waypoints[:, 1:, :] - waypoints[:, :-1, :]  # (N, W-1, 3)
        horizontal_dist = torch.norm(diffs[:, :, :2], dim=-1)  # (N, W-1)
        height_diff = diffs[:, :, 2]  # (N, W-1)

        # 计算爬升/下降角度
        angles = torch.atan2(torch.abs(height_diff), horizontal_dist + 1e-6)  # (N, W-1)

        # 分别检查爬升和下降
        climb_angles = torch.where(height_diff > 0, angles, torch.zeros_like(angles))
        descent_angles = torch.where(height_diff < 0, angles, torch.zeros_like(angles))

        max_climb = climb_angles.max(dim=-1).values  # (N,)
        max_descent = descent_angles.max(dim=-1).values  # (N,)

        valid_climb = max_climb <= self.cfg.max_climb_angle
        valid_descent = max_descent <= self.cfg.max_descent_angle

        valid = valid_climb & valid_descent

        return valid

    def validate_segment_length(
        self,
        segment_s0: torch.Tensor,  # (N, S)
        segment_s1: torch.Tensor,  # (N, S)
    ) -> torch.Tensor:
        """验证 segment 长度合理性

        Args:
            segment_s0: segment 起始位置
            segment_s1: segment 结束位置

        Returns:
            torch.Tensor: (N,) bool，True 表示长度合理
        """
        N, S = segment_s0.shape
        device = segment_s0.device

        # 计算每个 segment 的长度
        segment_lengths = segment_s1 - segment_s0  # (N, S)

        # 检查是否有长度小于最小阈值
        min_lengths = segment_lengths.min(dim=-1).values  # (N,)
        valid = min_lengths >= self.cfg.min_segment_length

        return valid
