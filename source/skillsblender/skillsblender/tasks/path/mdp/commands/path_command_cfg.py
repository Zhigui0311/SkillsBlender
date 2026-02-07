from __future__ import annotations
from typing import Tuple, Literal, List
from dataclasses import MISSING, field

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from skillsblender.tasks.path.config import (
    WAYPOINTS_MARKER_CFG,
    START_SPHERE_MARKER_CFG,
    GOAL_SPHERE_MARKER_CFG,
    PATH_HEADING_MARKER_CFG,
    ROBOT_HEADING_MARKER_CFG,
)


_DEFAULT_SKILL_PARAM_ALIASES: dict[str, str] = {
    "stairs_up": "stairs_params",
    "stairs_down": "stairs_params",
}


def resolve_skill_params_attr(cfg_obj, skill_name: str, aliases: dict[str, str] | None = None) -> str | None:
    """Resolve a skill name to a `*_params` attribute on `cfg_obj`."""
    candidate_attrs: list[str] = []
    alias_map = dict(_DEFAULT_SKILL_PARAM_ALIASES)
    if aliases:
        alias_map.update(aliases)
    if skill_name in alias_map:
        candidate_attrs.append(alias_map[skill_name])
    candidate_attrs.append(f"{skill_name}_params")
    if "_" in skill_name:
        candidate_attrs.append(f"{skill_name.rsplit('_', 1)[0]}_params")

    for attr_name in candidate_attrs:
        if hasattr(cfg_obj, attr_name):
            return attr_name
    return None


def get_registered_skill_names() -> list[str]:
    """Read skill names from SegmentPathCommand registry if available."""
    try:
        from .base_path_command import SegmentPathCommand

        return list(SegmentPathCommand.SKILL_NAMES)
    except Exception:
        return []


def sync_skill_params_to_generator_cfg(
    command_cfg,
    generator_cfg=None,
    skill_names: list[str] | None = None,
):
    """Sync all recognized `*_params` from command cfg to generator cfg."""
    gen_cfg = command_cfg.path_generator_cfg if generator_cfg is None else generator_cfg
    aliases = getattr(gen_cfg, "skill_param_aliases", None)
    if aliases is None:
        aliases = {}

    names = list(skill_names or get_registered_skill_names())
    if not names:
        names = list(getattr(gen_cfg, "skill_sequence", []) or [])

    for skill_name in names:
        attr_name = resolve_skill_params_attr(command_cfg, skill_name, aliases=aliases)
        if attr_name is None:
            continue
        setattr(gen_cfg, attr_name, getattr(command_cfg, attr_name))

    return gen_cfg


@configclass
class JumpParams:
    scan_dist: float = 6.0
    scan_step: float = 0.1
    scan_width: float = 0.6
    gap_threshold: float = -0.15  # height drop threshold (m)
    # geometric guards to avoid "immediate pit" or degenerate tiny gaps
    min_gap_start_dist: float = 1.2
    min_gap_width: float = 0.35
    max_gap_width: float = 1.00
    min_jump_start_dist: float = 0.9
    min_landing_runout: float = 1.0
    # jump approach phase window for reward gating
    approach_phase_window: float = 1.0
    # fallback synthetic-gap placement when scanner misses gaps
    fallback_gap_center_ratio: float = 0.58

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
class PlatformParams:
    # planned platform climb geometry: step up to a high ledge
    climb_len: float = 2.0
    climb_height: float = 0.5
    start_dist_range: Tuple[float, float] = (1.0, 2.0)


@configclass
class ClimbParams:
    # planned climb (slope/ramp) traversal parameters
    climb_len: float = 2.0
    max_slope_deg: float = 30.0
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
class PathSamplingCfg:
    """Path sampling and heading strategy configuration."""
    path_type: str = "linear"
    height_change: bool = False
    # interpreted as (min_len, max_len, z_offset); only len range used in current commands
    end_to_start_pos: Tuple[float, float, float] = (5.0, 5.0, 0.0)
    yaw_type: str = "along_path"
    start_heading: Tuple[float, float] = (0.0, 0.0)
    end_heading: Tuple[float, float] = (0.0, 0.0)
    # yaw_mode: "fixed" keeps yaw constant; "interp" linearly interpolates to end_heading.
    yaw_mode: str = "fixed"
    sample_goal_distance: bool = False

@configclass
class PathGeneratorCfg:
    """路径生成器配置"""
    # 环境边界
    env_bounds: EnvironmentBounds = field(default_factory=EnvironmentBounds)

    # 路径采样参数
    num_waypoints: int = 80  # 路径航点数量
    waypoint_spacing: float = 0.1  # 航点间距 (m)
    num_seg_params: int = 6

    # 技能参数
    walk_params: WalkParams = field(default_factory=WalkParams)
    jump_params: JumpParams = field(default_factory=JumpParams)
    stairs_params: StairsParams = field(default_factory=StairsParams)
    platform_params: PlatformParams = field(default_factory=PlatformParams)
    climb_params: ClimbParams = field(default_factory=ClimbParams)
    crouch_params: CrouchParams = field(default_factory=CrouchParams)

    # 验证参数
    validation_params: ValidationParams = field(default_factory=ValidationParams)

    # 多技能路径规划
    enable_multi_skill: bool = False  # 是否启用多技能路径规划
    skill_sequence: List[str] = field(default_factory=lambda: ["walk"])  # 技能序列
    # optional aliases when a skill name maps to shared params (e.g. stairs_up/down -> stairs_params)
    skill_param_aliases: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_SKILL_PARAM_ALIASES))

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
    num_skills: int = 8  # keep in sync with PathCommand skill ids

    # segment param vector (shared across skills)
    # Set to 0 to disable per-segment param table (smaller command/meta and slightly faster updates).
    num_seg_params: int = 6

    # normalization / clipping for meta distances
    dist_clip: float = 5.0
    

@configclass
class PathCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    debug_vis: bool = False
    ranges: PathRanges = PathRanges()
    path_generator_cfg: PathGeneratorCfg = PathGeneratorCfg()

    sampling: PathSamplingCfg = PathSamplingCfg()

    # per-skill config
    walk_params: WalkParams = WalkParams()
    jump_params: JumpParams = JumpParams()
    stairs_params: StairsParams = StairsParams()
    crouch_params: CrouchParams = CrouchParams()
    platform_params: PlatformParams = PlatformParams()
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

    path_heading_visualizer_cfg: VisualizationMarkersCfg = PATH_HEADING_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_heading"
    )
    """The configuration for desired heading visualization markers (arrows)."""

    robot_heading_visualizer_cfg: VisualizationMarkersCfg = ROBOT_HEADING_MARKER_CFG.replace(
        prim_path="/Visuals/Command/robot_heading"
    )
    """The configuration for robot heading visualization markers (arrows)."""

    def __post_init__(self):
        if getattr(self, "class_type", None) in (None, MISSING):
            # local import avoids circular dependency
            from .planner_path_command import PlannerPathCommand

            self.class_type = PlannerPathCommand
        # sync generator cfg with command cfg
        self.path_generator_cfg.num_waypoints = self.ranges.num_waypoints
        self.path_generator_cfg.num_seg_params = max(int(self.ranges.num_seg_params), 0)
        sync_skill_params_to_generator_cfg(self)
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
class PlatformPathCommandCfg(PathCommandCfg):
    def __post_init__(self):
        super().__post_init__()
        from .platform_path_command import PlatformPathCommand

        self.class_type = PlatformPathCommand


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


# Nested helper names used in configs
PathCommandCfg.Sampling = PathSamplingCfg
PathCommandCfg.Ranges = PathRanges
PathCommandCfg.PathGeneratorCfg = PathGeneratorCfg
PathCommandCfg.WalkParams = WalkParams
PathCommandCfg.JumpParams = JumpParams
PathCommandCfg.StairsParams = StairsParams
PathCommandCfg.PlatformParams = PlatformParams
PathCommandCfg.ClimbParams = ClimbParams
PathCommandCfg.CrouchParams = CrouchParams

JumpPathCommandCfg.Sampling = PathSamplingCfg
JumpPathCommandCfg.Ranges = PathRanges
JumpPathCommandCfg.JumpParams = JumpParams
