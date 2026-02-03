from __future__ import annotations
from typing import Tuple, Literal
from isaaclab.utils import configclass
from dataclasses import field 
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from skillsblender.tasks.path.config import WAYPOINTS_MARKER_CFG, START_SPHERE_MARKER_CFG, GOAL_SPHERE_MARKER_CFG


@configclass
class JumpParams:
    scan_dist: float = 6.0
    scan_step: float = 0.1
    scan_width: float = 0.6
    gap_threshold: float = -0.15  # height drop threshold (m)

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
class PathRanges:
    num_waypoints: int = 64
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
    num_seg_params: int = 6

    # normalization / clipping for meta distances
    dist_clip: float = 5.0
    

@configclass
class PathCommandCfg:
    asset_name: str = "robot"
    debug_vis: bool = False
    ranges: PathRanges = PathRanges()

    # per-skill config
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


