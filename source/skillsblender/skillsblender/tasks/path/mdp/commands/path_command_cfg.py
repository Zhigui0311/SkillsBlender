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
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
        #TODO：应该有路径的插值点的设置（依据地形信息）
        # terrain_type: str = None  #terrain type to consider for path planning
        # num_key_path_points: int = MISSING # number of path points to generate #加入地形以及路径规划算法后启用
        # key_path_points : dict = MISSING  # list of key path points [(x1,y1),(x2,y2),...] 
        # curvature_factor: float = MISSING
        #曲线弯曲程度 或者说是不使用贝塞尔曲线的插值 还是说直接生成直线


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

    @property
    def slice_nums(self) -> int:
        return 4*self.ranges.num_lookahead_waypoints


@configclass
class JumpPathCommandCfg(PathCommandCfg):

    class_type: type = PathCommand

    resampling_time_range: tuple[float, float] = MISSING
    
    asset_name: str = "robot"
    
    # @configclass
    # class JumpParams:
    #     """Parameters for jump path planning.
    #     """
    #     jump_height_range: tuple[float, float] = (0.25, 0.4)
    #     # 判定为沟壑的高度降幅阈值 (米)
    #     gap_threshold: float = -0.3 
    #     # 扫描地形时的前向最大距离 (米)
    #     scan_dist: float = 8.0
    #     # 扫描精度 (米)
    #     scan_step: float = 0.1
    #     # 起跳前的预留距离 (起跳点距离沟壑边缘的距离)
    #     takeoff_buffer: float = 0.3
    #     # 落地后的缓冲距离
    #     landing_buffer: float = 0.5
        
    # jump_params: JumpParams = JumpParams()
    
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
        num_waypoints: int = MISSING  #total number of waypoints
        num_lookahead_waypoints: int = MISSING  #number of lookahead waypoints
        waypoint_reach_threshold: float = MISSING  #distance threshold to consider a waypoint reached
    
    ranges: Ranges = MISSING
    

