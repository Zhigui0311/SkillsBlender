"""Platform climb skill environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Platform Terrain Configuration
# ==============================================================================

# Pit geometry helpers (estimated depth used to align command height with the rim/top).
# Start easy; curriculum can raise toward PIT_SLOPE_RANGE_FINAL.
PIT_SLOPE_RANGE_INIT = (0.20, 0.30)
PIT_SLOPE_RANGE_FINAL = (0.55, 0.85)
PIT_SLOPE_STEP = 0.05
PIT_SLOPE_RANGE = PIT_SLOPE_RANGE_INIT
PIT_PLATFORM_WIDTH = 2.0
PIT_BORDER_WIDTH = 0.5
PIT_SIZE = 8.0
PIT_RUN = (PIT_SIZE * 0.5) - (PIT_PLATFORM_WIDTH * 0.5) - PIT_BORDER_WIDTH
PIT_DEPTH_EST = ((PIT_SLOPE_RANGE[0] + PIT_SLOPE_RANGE[1]) * 0.5) * PIT_RUN
PLATFORM_TOP_HEIGHT = 0.56

PLATFORM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=PIT_BORDER_WIDTH,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Central pit: surrounding ground higher than the pit floor.
        # This keeps the robot initialized in a depression and encourages climbing out.
        "pit": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=PIT_SLOPE_RANGE,  # curriculum will raise this toward PIT_SLOPE_RANGE_FINAL
            platform_width=PIT_PLATFORM_WIDTH,
            border_width=PIT_BORDER_WIDTH,
        ),
    },
)


@configclass
class MyPlatformSceneCfg(MySceneCfg):
    """Platform climb scene configuration with flat terrain."""

    # Steep ledge obstacle (near-vertical front face) to train front-leg lift and pull-up behavior.
    # Keep this cuboid tall so curriculum can move its top higher while still touching ground.
    climb_step = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ClimbStep",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.8, 0.0, -0.04),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 1.6, 1.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Top platform after the ledge.
    climb_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ClimbTop",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.9, 0.0, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.6, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[6.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground", "{ENV_REGEX_NS}/ClimbStep", "{ENV_REGEX_NS}/ClimbTop"],
    )

    def __post_init__(self):
        super().__post_init__()
        # Update terrain for platform climb task
        self.terrain.terrain_generator = PLATFORM_TERRAIN_CFG


# ==============================================================================
# Platform Environment Configuration
# ==============================================================================

@configclass
class PlatformPathEnvCfg(PathEnvCfg):
    """Platform climb skill training environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Use platform scene
        self.scene: MyPlatformSceneCfg = MyPlatformSceneCfg(num_envs=4096, env_spacing=2.5)
        self.scene.env_spacing = 4.0
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk", "platform", "walk"]
        self.commands.path_tracking.sampling.yaw_type = "fixed"
        self.commands.path_tracking.sampling.start_heading = (0.0, 0.0)
        self.commands.path_tracking.sampling.end_to_start_pos = (4.0, 6.0, 0.0)
        self.commands.path_tracking.sampling.sample_goal_distance = True

        # Platform-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 5.2
        self.commands.path_tracking.platform_params.start_dist_range = (1.0, 1.3)
        self.commands.path_tracking.platform_params.climb_len = 1.3
        # Align commanded climb height with pit depth + platform top.
        self.commands.path_tracking.platform_params.climb_height = PIT_DEPTH_EST + PLATFORM_TOP_HEIGHT

        # Keep starts aligned with the ledge.
        self.events.reset_base.params["pose_range"]["x"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["y"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)
        self.events.push_robot = None

        # Platform-specific rewards
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity_along_path_exp.weight = 1.0

        # Forward progress reward
        self.rewards.climb_track_velocity_phase.weight = 4.0
        self.rewards.climb_track_velocity_phase.params["desired_speed"] = 0.65
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.65

        # Stability on slopes
        self.rewards.flat_orientation.weight = -1.5
        self.rewards.base_height_l2.weight = 0.0

        # Prevent slipping
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*_foot"]

        # Smooth motion
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7

        # Camera: keep robot visible during platform climb.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)
