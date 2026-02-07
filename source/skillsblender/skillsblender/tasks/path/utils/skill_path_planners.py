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
from typing import Any, Dict, Optional, Mapping, Tuple

from skillsblender.tasks.path.mdp.commands.path_command_cfg import (
    WalkParams,
    JumpParams,
    StairsParams,
    PlatformParams,
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


def _trace_terrain_path(
    start_pos: torch.Tensor,
    start_yaw: torch.Tensor,
    segment_length: torch.Tensor,
    terrain_data: Dict[str, torch.Tensor],
    num_waypoints: int = 64,
    scan_width: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace a terrain-aware path using height_scanner, fallback to flat.

    Returns:
        waypoints: (N, W, 3)
        headings: (N, W)
        dist: (N, W) distance along path
        z_profile: (N, W)
        has_scanner: (N,) bool, True when valid scanner data is used
    """
    N = start_pos.shape[0]
    device = start_pos.device

    alpha = torch.linspace(0, 1, num_waypoints, device=device).view(1, -1)
    dist = alpha * segment_length.unsqueeze(-1)

    fwd = torch.stack([torch.cos(start_yaw), torch.sin(start_yaw)], dim=-1)
    xy = start_pos[:, None, :2] + fwd[:, None, :] * dist.unsqueeze(-1)

    z_profile = start_pos[:, 2].unsqueeze(-1).repeat(1, num_waypoints)
    headings = start_yaw.unsqueeze(-1).repeat(1, num_waypoints)

    has_scanner = torch.zeros(N, dtype=torch.bool, device=device)
    if "height_scanner" in terrain_data:
        ray_hits_w = terrain_data["height_scanner"]  # (N, R, 3)
        for bi in range(N):
            rays = ray_hits_w[bi]
            if rays.numel() == 0:
                continue
            rel = rays - start_pos[bi].unsqueeze(0)
            cy = torch.cos(start_yaw[bi])
            sy = torch.sin(start_yaw[bi])
            x_fwd = cy * rel[:, 0] + sy * rel[:, 1]
            y_lat = -sy * rel[:, 0] + cy * rel[:, 1]
            valid = (x_fwd >= 0.0) & (x_fwd <= segment_length[bi]) & (torch.abs(y_lat) <= scan_width)
            if not torch.any(valid):
                continue
            x = x_fwd[valid]
            z = rays[valid, 2]
            # Nearest-neighbor sampling along forward distance to preserve sharp edges (stairs).
            s = dist[bi]
            diff = torch.abs(s[:, None] - x[None, :])
            idx = torch.argmin(diff, dim=1)
            z_profile[bi] = z[idx]
            has_scanner[bi] = True

    waypoints = torch.zeros(N, num_waypoints, 3, device=device)
    waypoints[:, :, :2] = xy
    waypoints[:, :, 2] = z_profile
    return waypoints, headings, dist, z_profile, has_scanner


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
        """生成跳跃路径（Dual Mode: Real / Virtual）"""
        N = len(env_ids)
        device = start_pos.device

        # 判断是否有真实地形数据
        has_scanner = "height_scanner" in terrain_data

        # 1) Real Terrain: gap 检测
        if has_scanner:
            gap_s0, gap_s1, has_gap = self._detect_gap(start_pos, start_yaw, terrain_data)

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
        else:
            # 2) Virtual / Flat: sample a synthetic gap
            has_gap = torch.ones(N, dtype=torch.bool, device=device)
            gap_w = torch.empty((N,), device=device).uniform_(self.cfg.min_gap_width, self.cfg.max_gap_width)
            min_start = torch.full((N,), float(self.cfg.min_jump_start_dist), device=device)
            max_start = segment_length - gap_w - float(self.cfg.min_landing_runout)
            max_start = torch.maximum(max_start, min_start)
            u = torch.rand((N,), device=device)
            gap_s0 = min_start + (max_start - min_start) * u
            gap_s1 = gap_s0 + gap_w

        # 3) 跳跃参数（Real / Virtual 共用）
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

        # 4) Terrain-aware base path (flat if no scanner)
        waypoints, headings, dist_at_wp, z_profile, _ = _trace_terrain_path(
            start_pos=start_pos,
            start_yaw=start_yaw,
            segment_length=planned_len,
            terrain_data=terrain_data if has_scanner else {},
            num_waypoints=64,
            scan_width=float(self.cfg.scan_width),
        )

        # 5) 在跳跃段添加抛物线轨迹（起落点对齐地形高度）
        den = (jump_s1.unsqueeze(-1) - jump_s0.unsqueeze(-1)).clamp(min=1e-3)
        t = ((dist_at_wp - jump_s0.unsqueeze(-1)) / den).clamp(0.0, 1.0)

        # Sample takeoff/landing heights from traced terrain.
        z0 = torch.zeros(N, device=device)
        z1 = torch.zeros(N, device=device)
        for bi in range(N):
            idx0 = torch.argmin(torch.abs(dist_at_wp[bi] - jump_s0[bi]))
            idx1 = torch.argmin(torch.abs(dist_at_wp[bi] - jump_s1[bi]))
            z0[bi] = z_profile[bi, idx0]
            z1[bi] = z_profile[bi, idx1]

        z_lin = z0.unsqueeze(-1) + (z1 - z0).unsqueeze(-1) * t
        arc = jump_height.unsqueeze(-1) * 4.0 * t * (1.0 - t)
        z_jump = z_lin + arc

        mask = (dist_at_wp >= jump_s0.unsqueeze(-1)) & (dist_at_wp <= jump_s1.unsqueeze(-1)) & has_gap.unsqueeze(-1)
        waypoints[..., 2] = torch.where(mask, z_jump, z_profile)

        # 6) 构建 segment 参数（只输出标量参数，不暴露扫描原始数据）
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 1.45)
        self._set_segment_param(segment_params, "jump_height_ref", jump_height)
        # Store gap width in misc for downstream use (if any).
        self._set_segment_param(segment_params, "misc", gap_w)

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
        """验证是否可用（Real 或 Virtual）"""
        if "height_scanner" not in terrain_data:
            N = 1
            return torch.ones(N, dtype=torch.bool, device=self.device)

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
        """生成楼梯路径（Real Mode Only）"""
        N = len(env_ids)
        device = start_pos.device

        stairs_len = torch.minimum(
            segment_length,
            torch.full_like(segment_length, float(self.cfg.stairs_len)),
        )

        if "height_scanner" not in terrain_data:
            # No virtual mode for contact-dominant skills, fallback to Walk.
            waypoints, headings, _, _, _ = _trace_terrain_path(
                start_pos=start_pos,
                start_yaw=start_yaw,
                segment_length=stairs_len,
                terrain_data={},
                num_waypoints=64,
                scan_width=0.6,
            )
            params = self._make_segment_params(N)
            self._set_segment_param(params, "v_ref", 1.0)
            return SegmentPlan(
                waypoints=waypoints,
                headings=headings,
                skill_id=0,  # walk
                segment_length=stairs_len,
                segment_params=params,
                valid=torch.ones(N, dtype=torch.bool, device=device),
            )

        # 1) Terrain-aware path (dense samples to preserve step edges)
        num_waypoints = max(64, int(float(self.cfg.stairs_len) / float(self.cfg.step_length)) * 4)
        waypoints, headings, _, z_profile, _ = _trace_terrain_path(
            start_pos=start_pos,
            start_yaw=start_yaw,
            segment_length=stairs_len,
            terrain_data=terrain_data,
            num_waypoints=num_waypoints,
            scan_width=0.6,
        )

        # 2) 提取 step_height（从地形高度变化估计）
        step_height = torch.full((N,), float(self.cfg.step_height), device=device)
        for bi in range(N):
            dz = z_profile[bi, 1:] - z_profile[bi, :-1]
            if self.direction == "down":
                dz = -dz
            candidates = dz[dz > 0.02]
            if candidates.numel() > 0:
                step_height[bi] = candidates.median()

        # 3) 构建 segment 参数（高抬腿）
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 0.6)
        self._set_segment_param(segment_params, "step_height_ref", step_height)
        self._set_segment_param(segment_params, "misc", float(self.cfg.step_length))

        skill_id = 2 if self.direction == "up" else 3  # stairs_up / stairs_down

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=skill_id,
            segment_length=stairs_len,
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证楼梯地形（仅真实模式）"""
        if "height_scanner" not in terrain_data:
            return torch.tensor([False], device=self.device)
        N = terrain_data["height_scanner"].shape[0]
        return torch.ones(N, dtype=torch.bool, device=self.device)


class PlatformPathPlanner(SkillPathPlanner):
    """高台攀爬路径规划器 - 生成平地->垂直上升->高台路径"""

    def __init__(
        self,
        cfg: PlatformParams,
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
        """生成高台攀爬路径（Real Mode Only）"""
        N = len(env_ids)
        device = start_pos.device

        climb_len = torch.minimum(
            segment_length,
            torch.full_like(segment_length, float(self.cfg.climb_len)),
        )

        if "height_scanner" not in terrain_data:
            # No virtual mode for contact-dominant skills, fallback to Walk.
            waypoints, headings, _, _, _ = _trace_terrain_path(
                start_pos=start_pos,
                start_yaw=start_yaw,
                segment_length=climb_len,
                terrain_data={},
                num_waypoints=64,
                scan_width=0.6,
            )
            params = self._make_segment_params(N)
            self._set_segment_param(params, "v_ref", 1.0)
            return SegmentPlan(
                waypoints=waypoints,
                headings=headings,
                skill_id=0,  # walk
                segment_length=climb_len,
                segment_params=params,
                valid=torch.ones(N, dtype=torch.bool, device=device),
            )

        # 1) Terrain-aware path
        waypoints, headings, dist_at_wp, z_profile, _ = _trace_terrain_path(
            start_pos=start_pos,
            start_yaw=start_yaw,
            segment_length=climb_len,
            terrain_data=terrain_data,
            num_waypoints=64,
            scan_width=0.6,
        )

        # 2) 提取 climb_height（高台高度）
        climb_height = (z_profile.max(dim=1).values - z_profile.min(dim=1).values).clamp(min=0.0)

        # 强化为“平地 -> 垂直上升 -> 高台平地”
        for bi in range(N):
            dz = z_profile[bi, 1:] - z_profile[bi, :-1]
            if dz.numel() == 0:
                continue
            idx = torch.argmax(dz)
            z0 = z_profile[bi, 0]
            z1 = z0 + climb_height[bi]
            waypoints[bi, : idx + 1, 2] = z0
            waypoints[bi, idx + 1 :, 2] = z1

        # 3) 构建 segment 参数
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 0.7)
        self._set_segment_param(segment_params, "misc", climb_height)

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=6,  # SKILL_ID["platform"]
            segment_length=climb_len,
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证高台地形（仅真实模式）"""
        if "height_scanner" not in terrain_data:
            return torch.tensor([False], device=self.device)
        N = terrain_data["height_scanner"].shape[0]
        return torch.ones(N, dtype=torch.bool, device=self.device)


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
        """生成蹲伏路径（Dual Mode: Real / Virtual）"""
        N = len(env_ids)
        device = start_pos.device

        crouch_len = min(self.cfg.crouch_len, segment_length.min().item())
        seg_len = torch.full((N,), crouch_len, device=device)

        has_scanner = "height_scanner" in terrain_data

        # 1) Real Terrain: 贴地轨迹
        if has_scanner:
            waypoints, headings, dist_at_wp, z_profile, has_valid = _trace_terrain_path(
                start_pos=start_pos,
                start_yaw=start_yaw,
                segment_length=seg_len,
                terrain_data=terrain_data,
                num_waypoints=64,
                scan_width=0.6,
            )
            depth = None
        else:
            # 2) Virtual / Flat: 先生成绝对平直路径
            waypoints, headings, dist_at_wp, z_profile, has_valid = _trace_terrain_path(
                start_pos=start_pos,
                start_yaw=start_yaw,
                segment_length=seg_len,
                terrain_data={},
                num_waypoints=64,
                scan_width=0.6,
            )

            # 在平地上生成下潜曲线（cosine 平滑）
            # 目标下潜深度：从配置中采样，没有则用 base_height_ref 的比例
            if hasattr(self.cfg, "crouch_height_range"):
                h_min, h_max = self.cfg.crouch_height_range
                depth = torch.empty((N,), device=device).uniform_(float(h_min), float(h_max))
            else:
                depth = torch.full((N,), max(0.08, 0.5 * float(self.cfg.base_height_ref)), device=device)

            t = dist_at_wp / torch.clamp(seg_len.unsqueeze(-1), min=1e-3)
            smooth = 0.5 - 0.5 * torch.cos(2.0 * torch.pi * t)  # 0->1->0
            z_offset = -depth.unsqueeze(-1) * smooth
            waypoints[..., 2] = z_profile + z_offset

        # 3) 参数提取：障碍高度（简化为地形起伏高度）
        if has_scanner:
            obs_height = (z_profile.max(dim=1).values - z_profile.min(dim=1).values).clamp(min=0.0)
            target_base_h = torch.clamp(
                start_pos[:, 2] - obs_height,
                min=0.12,
                max=start_pos[:, 2],
            )
        else:
            # Virtual: base height target follows the hallucinated crouch depth.
            target_base_h = torch.clamp(start_pos[:, 2] - depth, min=0.12, max=start_pos[:, 2])

        # 4) 构建 segment 参数（低姿态）
        segment_params = self._make_segment_params(N)
        self._set_segment_param(segment_params, "v_ref", 0.5)
        self._set_segment_param(segment_params, "base_height_ref", target_base_h)

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=4,  # SKILL_ID["crouch"]
            segment_length=seg_len,
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """验证蹲伏地形"""
        return torch.tensor([True], device=self.device)


class ClimbPathPlanner(SkillPathPlanner):
    """爬坡路径规划器 - 生成坡道路径（Real Mode Only, max 30 deg）"""

    def __init__(
        self,
        cfg,
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
        N = len(env_ids)
        device = start_pos.device

        climb_len = torch.minimum(
            segment_length,
            torch.full_like(segment_length, float(self.cfg.climb_len)),
        )

        if "height_scanner" not in terrain_data:
            # No virtual mode for contact-dominant skills, fallback to Walk.
            waypoints, headings, _, _, _ = _trace_terrain_path(
                start_pos=start_pos,
                start_yaw=start_yaw,
                segment_length=climb_len,
                terrain_data={},
                num_waypoints=64,
                scan_width=0.6,
            )
            params = self._make_segment_params(N)
            self._set_segment_param(params, "v_ref", 1.0)
            return SegmentPlan(
                waypoints=waypoints,
                headings=headings,
                skill_id=0,  # walk
                segment_length=climb_len,
                segment_params=params,
                valid=torch.ones(N, dtype=torch.bool, device=device),
            )

        waypoints, headings, dist_at_wp, z_profile, _ = _trace_terrain_path(
            start_pos=start_pos,
            start_yaw=start_yaw,
            segment_length=climb_len,
            terrain_data=terrain_data,
            num_waypoints=64,
            scan_width=0.6,
        )

        # Estimate slope angle and clamp to max_slope_deg.
        dz = z_profile[:, -1] - z_profile[:, 0]
        angle = torch.atan2(dz, torch.clamp(climb_len, min=1e-3))
        max_rad = torch.deg2rad(torch.full((N,), float(self.cfg.max_slope_deg), device=device))
        angle = torch.clamp(angle, -max_rad, max_rad)

        # If terrain slope is steeper than max, scale z profile.
        scale = torch.clamp(max_rad / torch.clamp(torch.abs(angle), min=1e-6), max=1.0)
        z0 = z_profile[:, :1]
        z_profile = z0 + (z_profile - z0) * scale[:, None]
        waypoints[..., 2] = z_profile

        params = self._make_segment_params(N)
        self._set_segment_param(params, "v_ref", 0.8)
        self._set_segment_param(params, "slope_angle_ref", angle)

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=7,  # SKILL_ID["climb"]
            segment_length=climb_len,
            segment_params=params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(
        self,
        terrain_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "height_scanner" not in terrain_data:
            return torch.tensor([False], device=self.device)
        N = terrain_data["height_scanner"].shape[0]
        return torch.ones(N, dtype=torch.bool, device=self.device)


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
        "platform": SkillPlannerRegistration(PlatformPathPlanner, "platform_params"),
        "climb": SkillPlannerRegistration(ClimbPathPlanner, "climb_params"),
        "crouch": SkillPlannerRegistration(CrouchPathPlanner, "crouch_params"),
    }


DEFAULT_SKILL_PLANNER_REGISTRY = build_default_skill_planner_registry()
