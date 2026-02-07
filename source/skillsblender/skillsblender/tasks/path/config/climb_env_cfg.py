"""Climb (slope) skill environment configuration."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg
from skillsblender.tasks.path.utils.terrain import CLIMB_TERRAIN_CFG
import skillsblender.tasks.path.mdp as mdp


@configclass
class MyClimbSceneCfg(InteractiveSceneCfg):
    """Scene configuration for climb (slope) terrain with height scanner."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CLIMB_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = None

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[6.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ClimbPathEnvCfg(PathEnvCfg):
    """Climb (slope) skill training environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Use climb (slope) scene with height scanner
        self.scene: MyClimbSceneCfg = MyClimbSceneCfg(num_envs=4096, env_spacing=2.5)
        self.scene.env_spacing = 4.0
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk", "climb", "walk"]
        self.commands.path_tracking.sampling.yaw_type = "fixed"
        self.commands.path_tracking.sampling.start_heading = (0.0, 0.0)
        self.commands.path_tracking.sampling.end_to_start_pos = (3.5, 6.0, 0.0)
        self.commands.path_tracking.sampling.sample_goal_distance = True

        # Climb-specific parameters
        self.commands.path_tracking.climb_params.start_dist_range = (1.0, 1.5)
        self.commands.path_tracking.climb_params.climb_len = 2.0
        self.commands.path_tracking.climb_params.max_slope_deg = 30.0

        # Rewards: encourage forward progress with stability
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity_along_path_exp.weight = 1.0
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.9
        self.rewards.flat_orientation.weight = -1.2
        self.rewards.base_height_l2.weight = 0.0

        if self.__class__.__name__ == "ClimbPathEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class ClimbPathEnvCfg_PLAY(ClimbPathEnvCfg):
    """Play configuration for climb debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0
        self.events.push_robot = None
