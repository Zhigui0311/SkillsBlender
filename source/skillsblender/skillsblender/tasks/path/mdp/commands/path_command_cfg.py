from __future__ import annotations
#未来特性导入

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal
from isaaclab.utils import configclass

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG


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
        end_to_start_pos: tuple[float, float] = None #range for end position sampling  .= path length range
        yaw_type: Literal['decoupled','along_path'] = MISSING #how to determine the yaw along the path
            #along_path: yaw is determined by the path direction
            #decoupled: yaw is sampled independently 可能会横着走
        start_heading: tuple[float, float] = None #range for start heading sampling, only for decoupled yaw type
        end_heading: tuple[float, float] = None #range for end heading sampling, only for decoupled yaw type
            #这里是需要开始的角度和结束的角度还是结束相对开始的角度差？
            
    inpoints: InterpolationPoints = MISSING

 
    @configclass
    class Ranges:
        num_waypoints: int = MISSING  #total number of waypoints
        num_lookahead_waypoints: int = MISSING  #number of lookahead waypoints
        waypoint_reach_threshold: float = MISSING  #distance threshold to consider a waypoint reached
    
    ranges: Ranges = MISSING




    
