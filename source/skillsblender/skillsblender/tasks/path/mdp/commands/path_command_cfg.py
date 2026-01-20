from __future__ import annotations
#未来特性导入

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal
from isaaclab.utils import configclass

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from skillsblender.tasks.path.config import WAYPOINTS_MARKER_CFG, START_SPHERE_MARKER_CFG, GOAL_SPHERE_MARKER_CFG


from .path_command import PathCommand
from .jump_path_command import JumpPathCommand

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv
    


@configclass
class PathCommandCfg(CommandTermCfg):
    """Configuration for the PathCommand class.
    """

    class_type: type = PathCommand

    resampling_time_range: tuple[float, float] = MISSING
    
    asset_name: str = "robot"

    # -- parameters for path planning
    @configclass
    class InterpolationPoints:
        """Interpolation points for path planning.
        """
        # resolution for path interpolation
        path_type: Literal['linear','bezier'] = MISSING
        height_change: bool = MISSING #whether to consider height change in path planning
        end_to_start_pos: tuple[float, float, float] = None #range for end position sampling  .= path length range
        yaw_type: Literal['decoupled','along_path'] = MISSING #how to determine the yaw along the path
            #along_path: yaw is determined by the path direction
            #decoupled: yaw is sampled independently 可能会横着走
        start_heading: tuple[float, float] = None #range for start heading sampling, only for decoupled yaw type
        end_heading: tuple[float, float] = None #range for end heading sampling, only for decoupled yaw type
            #这里是需要开始的角度和结束的角度还是结束相对开始的角度差？
            
    inpoints: InterpolationPoints = InterpolationPoints()

 
    @configclass
    class Ranges:
        num_waypoints: int = MISSING  #total number of waypoints
        num_lookahead_waypoints: int = MISSING  #number of lookahead waypoints
        waypoint_reach_threshold: float = MISSING  #distance threshold to consider a waypoint reached
    
    ranges: Ranges = Ranges()
    
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


@configclass
class JumpPathCommandCfg(CommandTermCfg):
    """Configuration for the JumpPathCommand class.

    跳跃命令通过地形扫描生成轨迹，不需要 inpoints 配置。
    """

    class_type : type = JumpPathCommand

    resampling_time_range: tuple[float, float] = MISSING

    asset_name: str = "robot"

    # 虚拟的 inpoints 配置（用于兼容性，不会被使用）
    @configclass
    class DummyInterpolationPoints:
        """Dummy interpolation points (not used by JumpPathCommand)."""
        path_type: Literal['linear','bezier'] = 'linear'
        height_change: bool = False
        end_to_start_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        yaw_type: Literal['decoupled','along_path'] = 'along_path'
        start_heading: tuple[float, float] = None
        end_heading: tuple[float, float] = None

    inpoints: DummyInterpolationPoints = DummyInterpolationPoints()

    @configclass
    class JumpParams:
        jump_height: float = 0.35            # Max height of the parabolic arc
        gap_threshold: float = -0.4          # Height drop to identify a gap (meters)
        scan_dist: float = 6.0               # How far to look ahead for gaps
        scan_step: float = 0.1               # Resolution of terrain scanning
        takeoff_margin: float = 0.2          # Distance before gap to start arc
        landing_margin: float = 0.3          # Distance after gap to end arc

    jump_params: JumpParams = JumpParams()

    @configclass
    class Ranges:
        num_waypoints: int = MISSING         # total number of waypoints
        num_lookahead_waypoints: int = MISSING  # number of lookahead waypoints
        waypoint_reach_threshold: float = MISSING  # distance threshold to consider a waypoint reached

    ranges: Ranges = Ranges()

    # 可视化标记配置（复用 PathCommandCfg 的）
    path_waypoints_visualizer_cfg: VisualizationMarkersCfg = WAYPOINTS_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_waypoints"
    )

    path_start_visualizer_cfg: VisualizationMarkersCfg = START_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_start"
    )

    path_goal_visualizer_cfg: VisualizationMarkersCfg = GOAL_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_goal"
    )

    # debug_vis: bool = False

