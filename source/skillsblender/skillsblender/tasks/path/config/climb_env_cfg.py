"""Climb skill environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Climb Terrain Configuration
# ==============================================================================

CLIMB_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=4.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Flat ground; climb is defined by the step and top platform props.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class MyClimbSceneCfg(MySceneCfg):
    """Climb scene configuration with sloped terrain."""

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

    def __post_init__(self):
        super().__post_init__()
        # Update terrain for climb task
        self.terrain.terrain_generator = CLIMB_TERRAIN_CFG


# ==============================================================================
# Climb Environment Configuration
# ==============================================================================

@configclass
class ClimbPathEnvCfg(PathEnvCfg):
    """Climb skill training environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Use climb scene
        self.scene: MyClimbSceneCfg = MyClimbSceneCfg(num_envs=4096, env_spacing=2.5)
        self.scene.env_spacing = 4.0
        self.commands.path_tracking.class_type = mdp.commands.ClimbPathCommand
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk", "climb", "walk"]
        self.commands.path_tracking.sampling.yaw_type = "fixed"
        self.commands.path_tracking.sampling.start_heading = (0.0, 0.0)

        # Climb-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 5.2
        self.commands.path_tracking.climb_params.start_dist_range = (1.0, 1.3)
        self.commands.path_tracking.climb_params.climb_len = 1.3
        self.commands.path_tracking.climb_params.climb_height = 0.56

        # Keep starts aligned with the ledge.
        self.events.reset_base.params["pose_range"]["x"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["y"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)
        self.events.push_robot = None

        # Climb-specific rewards
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

        # Camera: keep robot visible during climb.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)
