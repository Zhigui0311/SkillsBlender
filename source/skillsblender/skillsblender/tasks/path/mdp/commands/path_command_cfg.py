from __future__ import annotations
from typing import Tuple, Literal, List
from dataclasses import MISSING, field

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from skillsblender.tasks.path.config import WAYPOINTS_MARKER_CFG, START_SPHERE_MARKER_CFG, GOAL_SPHERE_MARKER_CFG


@configclass
class JumpParams:
    scan_dist: float = 6.0
    scan_step: float = 0.1
    scan_width: float = 0.6
    gap_threshold: float = -0.15  # height drop threshold (m)

    # optional fixed params (when provided, override min/max logic)
    takeoff_margin: float | None = None
    landing_margin: float | None = None
    jump_height: float | None = None

    # used when scanner is absent
    takeoff_margin_min: float = 0.2
    takeoff_margin_max: float = 0.6
    landing_margin_min: float = 0.25
    landing_margin_max: float = 0.6

    endpoint_extension_min: float = 1.0
    endpoint_extension_max: float = 2.0

    jump_height_min: float = 0.3
    jump_height_max: float = 0.8

    post_jump_distance: float = 0.0

    # heading jitter
    heading_offset_range: Tuple[float, float] = (0.0, 0.0)

@configclass
class StairsParams:
    # planned geometry parameters (not terrain sensing)
    stairs_len: float = 2.5
    step_length: float = 0.25
    step_height: float = 0.07
    # segment placement on the path
    start_dist_range: Tuple[float, float] = (1.0, 2.0)  # where the stairs segment starts along the path

@configclass
class CrouchParams:
    # planned constraint (not sensing)
    crouch_len: float = 2.0
    base_height_ref: float = 0.24
    start_dist_range: Tuple[float, float] = (1.0, 2.0)

@configclass
class ClimbParams:
    # planned climb geometry: continuous ramp-like height gain
    climb_len: float = 2.0
    climb_height: float = 0.5
    start_dist_range: Tuple[float, float] = (1.0, 2.0)

@configclass
class WalkParams:
    """行走路径参数"""
    v_ref: float = 1.0  # 参考速度 (m/s)
    base_height_ref: float = 0.34  # 参考高度 (m)

@configclass
class EnvironmentBounds:
    """环境边界配置"""
    x_range: Tuple[float, float] = (-5.0, 5.0)  # X 方向范围 (m)
    y_range: Tuple[float, float] = (-5.0, 5.0)  # Y 方向范围 (m)
    z_range: Tuple[float, float] = (0.0, 2.0)   # Z 方向范围 (m)
    boundary_margin: float = 0.5  # 距离边界的安全距离 (m)

@configclass
class ValidationParams:
    """路径验证参数"""
    max_waypoint_distance: float = 0.2  # 相邻航点最大距离 (m)
    max_heading_change: float = 0.5  # 相邻航向最大变化 (rad)，约 28.6 度
    max_climb_angle: float = 0.6  # 最大爬升角度 (rad)，约 34.4 度
    max_descent_angle: float = 0.6  # 最大下降角度 (rad)
    min_segment_length: float = 0.5  # 最小 segment 长度 (m)


@configclass
class InterpolationPoints:
    """Legacy path interpolation inputs (kept for backward compatibility)."""
    path_type: str = "linear"
    height_change: bool = False
    # interpreted as (min_len, max_len, z_offset); only len range used in current commands
    end_to_start_pos: Tuple[float, float, float] = (5.0, 5.0, 0.0)
    yaw_type: str = "along_path"
    start_heading: Tuple[float, float] = (0.0, 0.0)
    end_heading: Tuple[float, float] = (0.0, 0.0)

@configclass
class PathGeneratorCfg:
    """路径生成器配置"""
    # 环境边界
    env_bounds: EnvironmentBounds = field(default_factory=EnvironmentBounds)

    # 路径采样参数
    num_waypoints: int = 80  # 路径航点数量
    waypoint_spacing: float = 0.1  # 航点间距 (m)

    # 技能参数
    walk_params: WalkParams = field(default_factory=WalkParams)
    jump_params: JumpParams = field(default_factory=JumpParams)
    stairs_params: StairsParams = field(default_factory=StairsParams)
    climb_params: ClimbParams = field(default_factory=ClimbParams)
    crouch_params: CrouchParams = field(default_factory=CrouchParams)

    # 验证参数
    validation_params: ValidationParams = field(default_factory=ValidationParams)

    # 多技能路径规划
    enable_multi_skill: bool = False  # 是否启用多技能路径规划
    skill_sequence: List[str] = field(default_factory=lambda: ["walk"])  # 技能序列

@configclass
class PathRanges:
    num_waypoints: int = 80
    num_lookahead_waypoints: int = 24
    waypoint_reach_threshold: float = 0.8

    # path length default (when no special segment)
    default_path_len: float = 5.0

    # command output options
    include_meta: bool = True
    include_skill_onehot: bool = True
    include_next_segment: bool = True

    # segment-table sizing
    max_segments: int = 8
    num_skills: int = 7  # keep in sync with PathCommand skill ids

    # segment param vector (shared across skills)
    num_seg_params: int = 6

    # normalization / clipping for meta distances
    dist_clip: float = 5.0
    

@configclass
class PathCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    debug_vis: bool = False
    ranges: PathRanges = PathRanges()
    path_generator_cfg: PathGeneratorCfg = PathGeneratorCfg()

    # legacy compatibility for older configs
    inpoints: InterpolationPoints = InterpolationPoints()

    # per-skill config
    walk_params: WalkParams = WalkParams()
    jump_params: JumpParams = JumpParams()
    stairs_params: StairsParams = StairsParams()
    crouch_params: CrouchParams = CrouchParams()
    climb_params: ClimbParams = ClimbParams()
    
    path_waypoints_visualizer_cfg: VisualizationMarkersCfg = WAYPOINTS_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_waypoints"
    )
    """The configuration for the path waypoints visualization marker. Defaults to WAYPOINT_MARKER_CFG.
    """

    path_start_visualizer_cfg: VisualizationMarkersCfg = START_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_start"
    )
    """The configuration for the path start visualization marker. Defaults to START_SPHERE_MARKER_CFG.
    """

    path_goal_visualizer_cfg: VisualizationMarkersCfg = GOAL_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_goal"
    )
    """The configuration for the path goal visualization marker. Defaults to GOAL_SPHERE_MARKER_CFG.
    """

    def __post_init__(self):
        if getattr(self, "class_type", None) in (None, MISSING):
            # local import avoids circular dependency
            from .planner_path_command import PlannerPathCommand

            self.class_type = PlannerPathCommand
        # sync generator cfg with command cfg
        self.path_generator_cfg.num_waypoints = self.ranges.num_waypoints
        self.path_generator_cfg.walk_params = self.walk_params
        self.path_generator_cfg.jump_params = self.jump_params
        self.path_generator_cfg.stairs_params = self.stairs_params
        self.path_generator_cfg.climb_params = self.climb_params
        self.path_generator_cfg.crouch_params = self.crouch_params
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()


@configclass
class JumpPathCommandCfg(PathCommandCfg):
    def __post_init__(self):
        super().__post_init__()
        from .jump_path_command import JumpPathCommand

        self.class_type = JumpPathCommand


@configclass
class StairsPathCommandCfg(PathCommandCfg):
    def __post_init__(self):
        super().__post_init__()
        from .stairs_path_command import StairsPathCommand

        self.class_type = StairsPathCommand


@configclass
class ClimbPathCommandCfg(PathCommandCfg):
    def __post_init__(self):
        super().__post_init__()
        from .climb_path_command import ClimbPathCommand

        self.class_type = ClimbPathCommand


@configclass
class CrouchPathCommandCfg(PathCommandCfg):
    def __post_init__(self):
        super().__post_init__()
        from .crouch_path_command import CrouchPathCommand

        self.class_type = CrouchPathCommand


# Backward-compatible nested names used in older configs
PathCommandCfg.InterpolationPoints = InterpolationPoints
PathCommandCfg.Ranges = PathRanges
PathCommandCfg.PathGeneratorCfg = PathGeneratorCfg
PathCommandCfg.WalkParams = WalkParams
PathCommandCfg.JumpParams = JumpParams
PathCommandCfg.StairsParams = StairsParams
PathCommandCfg.ClimbParams = ClimbParams
PathCommandCfg.CrouchParams = CrouchParams

JumpPathCommandCfg.InterpolationPoints = InterpolationPoints
JumpPathCommandCfg.Ranges = PathRanges
JumpPathCommandCfg.JumpParams = JumpParams
