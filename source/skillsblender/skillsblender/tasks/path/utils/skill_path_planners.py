"""技能路径规划器接口和实现

该模块定义了技能路径规划器的抽象接口，以及各个技能的具体实现：
- WalkPathPlanner: 行走路径规划器（直线路径）
- JumpPathPlanner: 跳跃路径规划器（gap 检测 + 抛物线轨迹）
- StairsPathPlanner: 楼梯路径规划器（上楼/下楼）
- ClimbPathPlanner: 攀爬路径规划器（斜坡）
- CrouchPathPlanner: 蹲伏路径规划器（低姿态）
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from typing import Any, Dict, Optional, Mapping

from skillsblender.tasks.path.mdp.commands.path_command_cfg import (
    WalkParams,
    JumpParams,
    StairsParams,
    ClimbParams,
    CrouchParams,
)


@dataclass
class SegmentPlan:
    """单个技能段的路径规划结果"""
    waypoints: torch.Tensor  # (N, W, 3) 路径航点 (x, y, z)，世界坐标系
    headings: torch.Tensor   # (N, W) 航向角 (yaw)，单位：弧度
    skill_id: int            # 技能 ID
    segment_length: torch.Tensor  # (N,) 实际 segment 长度（沿路径的距离）
    segment_params: torch.Tensor  # (N, P) segment 参数向量
    valid: torch.Tensor      # (N,) bool，是否成功规划


class SkillPathPlanner(ABC):
    """技能路径规划器接口"""

    def __init__(
        self,
        cfg,
        device: torch.device,
        num_seg_params: int = 6,
        seg_param: Mapping[str, int] | None = None,
    ):
        """初始化路径规划器

        Args:
            cfg: 技能特定的配置（如 JumpParams, WalkParams 等）
            device: PyTorch 设备
        """
        self.cfg = cfg
        self.device = device
        self.num_seg_params = max(int(num_seg_params), 0)
        self.seg_param = dict(seg_param or {})

    def _make_segment_params(self, num_envs: int) -> torch.Tensor:
        """Create (N, P) param tensor; P can be zero."""
        return torch.zeros(num_envs, self.num_seg_params, device=self.device)

    def _set_segment_param(self, params: torch.Tensor, name: str, value):
        """Set named slot only when the slot is configured in current width."""
        idx = self.seg_param.get(name, None)
        if idx is None or idx >= self.num_seg_params:
            return
        params[:, idx] = value

    @abstractmethod
    def plan_segment(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
        segment_length: torch.Tensor,
    ) -> SegmentPlan:
        """规划一个技能段的路径

        Args:
            env_ids: 环境 ID，shape (N,)
            start_pos: 起始位置，shape (N, 3)，世界坐标系
            start_yaw: 起始航向角，shape (N,)，单位：弧度
            terrain_data: 地形数据字典，可能包含：
                - "height_scanner": RayCaster 扫描结果
                - "terrain_heights": 地形高度图
                - "terrain_normals": 地形法向量
            segment_length: 期望的 segment 长度，shape (N,)

        Returns:
            SegmentPlan: 路径规划结果
        """
        pass

    @abstractmethod
    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证地形是否适合该技能

        Args:
            terrain_data: 地形数据字典

        Returns:
            torch.Tensor: (N,) bool tensor，True 表示地形适合
        """
        pass


class WalkPathPlanner(SkillPathPlanner):
    """行走路径规划器 - 生成简单的直线路径"""

    def __init__(
        self,
        cfg: WalkParams,
        device: torch.device,
        num_seg_params: int = 6,
        seg_param: Mapping[str, int] | None = None,
    ):
        super().__init__(cfg, device, num_seg_params=num_seg_params, seg_param=seg_param)

    def plan_segment(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
        segment_length: torch.Tensor,
    ) -> SegmentPlan:
        """生成直线行走路径"""
        N = len(env_ids)
        device = start_pos.device

        # 1. 计算终点位置
        forward_dir = torch.stack([
            torch.cos(start_yaw),
            torch.sin(start_yaw),
            torch.zeros_like(start_yaw),
        ], dim=-1)  # (N, 3)

        end_pos = start_pos + forward_dir * segment_length.unsqueeze(-1)

        # 2. 生成航点（线性插值）
        num_waypoints = 32  # 临时航点数量，后续会重新插值
        alpha = torch.linspace(0, 1, num_waypoints, device=device).view(1, -1, 1)
        waypoints = start_pos.unsqueeze(1) + (end_pos - start_pos).unsqueeze(1) * alpha

        # 3. 保持恒定航向
        headings = start_yaw.unsqueeze(-1).repeat(1, num_waypoints)

        # 4. 构建 segment 参数
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", self.cfg.v_ref)
        self._set_segment_param(segment_params, "base_height_ref", self.cfg.base_height_ref)

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=0,  # SKILL_ID["walk"]
            segment_length=segment_length,
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证地形平坦度 - 行走适用于所有地形"""
        if "terrain_heights" in terrain_data:
            heights = terrain_data["terrain_heights"]
            N = heights.shape[0]
            return torch.ones(N, dtype=torch.bool, device=heights.device)
        return torch.tensor([True], device=self.device)


class JumpPathPlanner(SkillPathPlanner):
    """跳跃路径规划器 - 检测 gap 并生成抛物线轨迹"""

    def __init__(
        self,
        cfg: JumpParams,
        device: torch.device,
        num_seg_params: int = 6,
        seg_param: Mapping[str, int] | None = None,
    ):
        super().__init__(cfg, device, num_seg_params=num_seg_params, seg_param=seg_param)

    def plan_segment(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
        segment_length: torch.Tensor,
    ) -> SegmentPlan:
        """生成跳跃路径"""
        N = len(env_ids)
        device = start_pos.device

        # 1. 检测 gap
        gap_s0, gap_s1, has_gap = self._detect_gap(
            start_pos, start_yaw, terrain_data
        )

        # Scanner miss fallback: still synthesize a jump gap so jump skill does not collapse to walk.
        missing = ~has_gap
        if torch.any(missing):
            seg_len_m = segment_length[missing]
            mid = float(self.cfg.fallback_gap_center_ratio) * seg_len_m
            gap_w = torch.clamp(0.18 * seg_len_m, min=self.cfg.min_gap_width, max=self.cfg.max_gap_width)
            s0 = torch.clamp(mid - 0.5 * gap_w, min=self.cfg.min_gap_start_dist)
            s1 = s0 + gap_w
            max_gap_end = seg_len_m - max(float(self.cfg.min_landing_runout), 0.5)
            s1 = torch.minimum(s1, max_gap_end)
            s0 = torch.minimum(s0, s1 - self.cfg.min_gap_width)
            gap_s0[missing] = s0
            gap_s1[missing] = s1
            has_gap[missing] = True

        # Sanitize detected/synthetic gaps.
        if torch.any(has_gap):
            hs0 = torch.clamp(gap_s0[has_gap], min=self.cfg.min_gap_start_dist)
            hw = (gap_s1[has_gap] - hs0).clamp(min=self.cfg.min_gap_width, max=self.cfg.max_gap_width)
            hs1 = hs0 + hw
            max_gap_end = segment_length[has_gap] - max(float(self.cfg.min_landing_runout), 0.5)
            hs1 = torch.minimum(hs1, max_gap_end)
            hs0 = torch.minimum(hs0, hs1 - self.cfg.min_gap_width)
            gap_s0[has_gap] = hs0
            gap_s1[has_gap] = hs1

        # 2. 计算跳跃参数
        gap_w = (gap_s1 - gap_s0).clamp(min=0.0)
        if self.cfg.takeoff_margin is None:
            takeoff = torch.clamp(
                0.2 + 0.4 * gap_w,
                min=self.cfg.takeoff_margin_min,
                max=self.cfg.takeoff_margin_max,
            )
        else:
            takeoff = torch.full_like(gap_w, float(self.cfg.takeoff_margin))

        if self.cfg.landing_margin is None:
            landing = torch.clamp(
                0.25 + 0.3 * gap_w,
                min=self.cfg.landing_margin_min,
                max=self.cfg.landing_margin_max,
            )
        else:
            landing = torch.full_like(gap_w, float(self.cfg.landing_margin))

        if self.cfg.jump_height is None:
            jump_height = torch.clamp(
                0.3 + 0.6 * gap_w,
                min=self.cfg.jump_height_min,
                max=self.cfg.jump_height_max,
            )
        else:
            jump_height = torch.full_like(gap_w, float(self.cfg.jump_height))

        # Keep enough approach distance before takeoff.
        pre_jump_start = gap_s0 - takeoff
        shift = torch.clamp(float(self.cfg.min_jump_start_dist) - pre_jump_start, min=0.0)
        gap_s0 = torch.where(has_gap, gap_s0 + shift, gap_s0)
        gap_s1 = torch.where(has_gap, gap_s1 + shift, gap_s1)

        jump_s0 = torch.where(
            has_gap,
            torch.maximum(gap_s0 - takeoff, torch.full_like(gap_s0, float(self.cfg.min_jump_start_dist))),
            torch.zeros_like(gap_s0),
        )
        jump_s1 = torch.where(has_gap, gap_s1 + landing, segment_length)

        extension = torch.clamp(1.0 + 0.5 * gap_w, min=self.cfg.endpoint_extension_min, max=self.cfg.endpoint_extension_max)
        planned_len = torch.where(
            has_gap,
            torch.maximum(jump_s1 + extension, jump_s1 + float(self.cfg.min_landing_runout)),
            segment_length,
        )
        if self.cfg.post_jump_distance > 0.0:
            planned_len = planned_len + float(self.cfg.post_jump_distance)

        # 3. 生成基础线性路径
        forward_dir = torch.stack([
            torch.cos(start_yaw),
            torch.sin(start_yaw),
            torch.zeros_like(start_yaw),
        ], dim=-1)

        end_pos = start_pos + forward_dir * planned_len.unsqueeze(-1)

        num_waypoints = 32
        alpha = torch.linspace(0, 1, num_waypoints, device=device).view(1, -1, 1)
        waypoints = start_pos.unsqueeze(1) + (end_pos - start_pos).unsqueeze(1) * alpha

        # 4. 在跳跃段添加抛物线轨迹
        dist_at_wp = alpha.squeeze(-1) * planned_len.unsqueeze(-1)  # (N, W)
        den = (jump_s1.unsqueeze(-1) - jump_s0.unsqueeze(-1)).clamp(min=1e-3)
        t = ((dist_at_wp - jump_s0.unsqueeze(-1)) / den).clamp(0.0, 1.0)
        arc = jump_height.unsqueeze(-1) * 4.0 * t * (1.0 - t)

        mask = (dist_at_wp >= jump_s0.unsqueeze(-1)) & (dist_at_wp <= jump_s1.unsqueeze(-1)) & has_gap.unsqueeze(-1)
        waypoints[..., 2] = waypoints[..., 2] + torch.where(mask, arc, torch.zeros_like(arc))

        # 5. 保持恒定航向
        headings = start_yaw.unsqueeze(-1).repeat(1, num_waypoints)

        # 6. 构建 segment 参数
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 1.45)
        self._set_segment_param(segment_params, "jump_height_ref", jump_height)

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=1,  # SKILL_ID["jump"]
            segment_length=planned_len,
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def _detect_gap(
        self,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """检测 gap（复用 JumpPathCommand 逻辑）"""
        N = start_pos.shape[0]
        device = start_pos.device

        gap_s0 = torch.zeros(N, device=device)
        gap_s1 = torch.zeros(N, device=device)
        has_gap = torch.zeros(N, dtype=torch.bool, device=device)

        if "height_scanner" not in terrain_data:
            return gap_s0, gap_s1, has_gap

        ray_hits_w = terrain_data["height_scanner"]  # (N, R, 3)

        # 转换到规划坐标系
        rel = ray_hits_w - start_pos.unsqueeze(1)
        cy = torch.cos(start_yaw).unsqueeze(-1)
        sy = torch.sin(start_yaw).unsqueeze(-1)
        dx = rel[..., 0]
        dy = rel[..., 1]
        x_fwd = cy * dx + sy * dy
        y_lat = -sy * dx + cy * dy
        z_w = ray_hits_w[..., 2]

        valid = (x_fwd >= 0.0) & (x_fwd <= self.cfg.scan_dist) & (torch.abs(y_lat) <= self.cfg.scan_width)

        for bi in range(N):
            m = valid[bi]
            if not torch.any(m):
                continue
            xv = x_fwd[bi, m]
            zv = z_w[bi, m]
            order = torch.argsort(xv)
            xv = xv[order]
            zv = zv[order]

            # 参考高度
            ref_window = min(7, zv.numel())
            ref_samples = zv[:ref_window]
            ref_k = max(1, ref_window // 2)
            ref_h = torch.topk(ref_samples, k=ref_k).values.median()

            # 检测 gap
            is_gap = (zv - ref_h) < self.cfg.gap_threshold
            if not torch.any(is_gap):
                continue

            gi = torch.nonzero(is_gap, as_tuple=False).squeeze(-1)
            first = gi[0]
            breaks = torch.nonzero(torch.diff(gi) > 1, as_tuple=False).squeeze(-1)
            last = gi[breaks[0]] if breaks.numel() > 0 else gi[-1]

            gap_s0[bi] = xv[first]
            gap_s1[bi] = xv[last]
            has_gap[bi] = True

        return gap_s0, gap_s1, has_gap

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证是否有 gap"""
        if "height_scanner" not in terrain_data:
            N = 1
            return torch.zeros(N, dtype=torch.bool, device=self.device)

        # 简单检测：是否有高度低于阈值的点
        ray_hits_w = terrain_data["height_scanner"]
        N = ray_hits_w.shape[0]
        z_w = ray_hits_w[..., 2]
        ref_h = z_w[:, :7].median(dim=1).values
        has_low_points = torch.any((z_w - ref_h.unsqueeze(-1)) < self.cfg.gap_threshold, dim=1)

        return has_low_points


class StairsPathPlanner(SkillPathPlanner):
    """楼梯路径规划器 - 生成楼梯路径（上楼/下楼）"""

    def __init__(
        self,
        cfg: StairsParams,
        device: torch.device,
        direction: str = "up",
        num_seg_params: int = 6,
        seg_param: Mapping[str, int] | None = None,
    ):
        super().__init__(cfg, device, num_seg_params=num_seg_params, seg_param=seg_param)
        self.direction = direction  # "up" or "down"

    def plan_segment(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
        segment_length: torch.Tensor,
    ) -> SegmentPlan:
        """生成楼梯路径"""
        N = len(env_ids)
        device = start_pos.device

        # 1. 计算楼梯参数
        stairs_len = min(self.cfg.stairs_len, segment_length.min().item())
        num_steps = int(stairs_len / self.cfg.step_length)
        step_height = self.cfg.step_height if self.direction == "up" else -self.cfg.step_height

        # 2. 生成楼梯航点
        num_waypoints = num_steps * 2  # 每个台阶 2 个航点（水平 + 垂直）
        waypoints = torch.zeros(N, num_waypoints, 3, device=device)

        forward_dir = torch.stack([
            torch.cos(start_yaw),
            torch.sin(start_yaw),
        ], dim=-1)

        for i in range(num_steps):
            # 水平移动
            wp_idx = i * 2
            dist = i * self.cfg.step_length
            height = i * step_height
            waypoints[:, wp_idx, :2] = start_pos[:, :2] + forward_dir * dist
            waypoints[:, wp_idx, 2] = start_pos[:, 2] + height

            # 垂直移动
            wp_idx = i * 2 + 1
            waypoints[:, wp_idx, :2] = waypoints[:, wp_idx - 1, :2]
            waypoints[:, wp_idx, 2] = start_pos[:, 2] + (i + 1) * step_height

        # 3. 保持恒定航向
        headings = start_yaw.unsqueeze(-1).repeat(1, num_waypoints)

        # 4. 构建 segment 参数
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 0.5)
        self._set_segment_param(segment_params, "base_height_ref", self.cfg.step_height)

        skill_id = 2 if self.direction == "up" else 3  # stairs_up / stairs_down

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=skill_id,
            segment_length=torch.full((N,), stairs_len, device=device),
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证楼梯地形"""
        return torch.tensor([True], device=self.device)


class ClimbPathPlanner(SkillPathPlanner):
    """攀爬路径规划器 - 生成斜坡路径"""

    def __init__(
        self,
        cfg: ClimbParams,
        device: torch.device,
        num_seg_params: int = 6,
        seg_param: Mapping[str, int] | None = None,
    ):
        super().__init__(cfg, device, num_seg_params=num_seg_params, seg_param=seg_param)

    def plan_segment(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
        segment_length: torch.Tensor,
    ) -> SegmentPlan:
        """生成斜坡路径"""
        N = len(env_ids)
        device = start_pos.device

        # 1. 计算斜坡参数
        climb_len = min(self.cfg.climb_len, segment_length.min().item())
        climb_height = self.cfg.climb_height

        # 2. 生成斜坡航点（线性高度增加）
        num_waypoints = 32
        alpha = torch.linspace(0, 1, num_waypoints, device=device).view(1, -1)

        forward_dir = torch.stack([
            torch.cos(start_yaw),
            torch.sin(start_yaw),
        ], dim=-1)

        waypoints = torch.zeros(N, num_waypoints, 3, device=device)
        waypoints[:, :, :2] = start_pos[:, :2].unsqueeze(1) + forward_dir.unsqueeze(1) * (alpha.unsqueeze(-1) * climb_len)
        waypoints[:, :, 2] = start_pos[:, 2].unsqueeze(1) + alpha * climb_height

        # 3. 保持恒定航向
        headings = start_yaw.unsqueeze(-1).repeat(1, num_waypoints)

        # 4. 构建 segment 参数
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 0.7)
        self._set_segment_param(segment_params, "base_height_ref", climb_height)

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=6,  # SKILL_ID["climb"]
            segment_length=torch.full((N,), climb_len, device=device),
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证斜坡地形"""
        return torch.tensor([True], device=self.device)


class CrouchPathPlanner(SkillPathPlanner):
    """蹲伏路径规划器 - 生成低姿态路径"""

    def __init__(
        self,
        cfg: CrouchParams,
        device: torch.device,
        num_seg_params: int = 6,
        seg_param: Mapping[str, int] | None = None,
    ):
        super().__init__(cfg, device, num_seg_params=num_seg_params, seg_param=seg_param)

    def plan_segment(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
        segment_length: torch.Tensor,
    ) -> SegmentPlan:
        """生成蹲伏路径"""
        N = len(env_ids)
        device = start_pos.device

        # 1. 计算蹲伏段长度
        crouch_len = min(self.cfg.crouch_len, segment_length.min().item())

        # 2. 生成直线路径（低姿态）
        num_waypoints = 32
        alpha = torch.linspace(0, 1, num_waypoints, device=device).view(1, -1, 1)

        forward_dir = torch.stack([
            torch.cos(start_yaw),
            torch.sin(start_yaw),
            torch.zeros_like(start_yaw),
        ], dim=-1)

        end_pos = start_pos + forward_dir * crouch_len
        waypoints = start_pos.unsqueeze(1) + (end_pos - start_pos).unsqueeze(1) * alpha

        # 3. 保持恒定航向
        headings = start_yaw.unsqueeze(-1).repeat(1, num_waypoints)

        # 4. 构建 segment 参数（低姿态）
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 0.5)
        self._set_segment_param(segment_params, "base_height_ref", self.cfg.base_height_ref)

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=4,  # SKILL_ID["crouch"]
            segment_length=torch.full((N,), crouch_len, device=device),
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证蹲伏地形"""
        return torch.tensor([True], device=self.device)


@dataclass(frozen=True)
class SkillPlannerRegistration:
    """Planner registration entry used by TerrainAwarePathGenerator."""

    planner_cls: type[SkillPathPlanner]
    params_attr: str
    planner_kwargs: dict[str, Any] = field(default_factory=dict)


def build_default_skill_planner_registry() -> dict[str, SkillPlannerRegistration]:
    """Default registry for built-in skill planners.

    Add new skills here (name + planner + params attr) to avoid touching the generator.
    """
    return {
        "walk": SkillPlannerRegistration(WalkPathPlanner, "walk_params"),
        "jump": SkillPlannerRegistration(JumpPathPlanner, "jump_params"),
        "stairs_up": SkillPlannerRegistration(
            StairsPathPlanner,
            "stairs_params",
            planner_kwargs={"direction": "up"},
        ),
        "stairs_down": SkillPlannerRegistration(
            StairsPathPlanner,
            "stairs_params",
            planner_kwargs={"direction": "down"},
        ),
        "climb": SkillPlannerRegistration(ClimbPathPlanner, "climb_params"),
        "crouch": SkillPlannerRegistration(CrouchPathPlanner, "crouch_params"),
    }


DEFAULT_SKILL_PLANNER_REGISTRY = build_default_skill_planner_registry()
