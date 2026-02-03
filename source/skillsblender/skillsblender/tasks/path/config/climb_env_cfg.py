"""Climb skill environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Climb Terrain Configuration
# ==============================================================================

CLIMB_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # 20% flat terrain for approach
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # 60% pyramid slopes (up)
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.6,
            slope_range=(0.10, 0.45),  # 10-45 degree slopes
            platform_width=2.0,
            border_width=0.25
        ),
        # 20% inverted pyramid slopes (down)
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.10, 0.45),
            platform_width=2.0,
            border_width=0.25
        ),
    },
)


@configclass
class MyClimbSceneCfg(MySceneCfg):
    """Climb scene configuration with sloped terrain."""

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

        # Climb-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 64
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.climb_params.climb_len = 2.0
        self.commands.path_tracking.climb_params.climb_height = 0.5

        # Climb-specific rewards
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity.weight = 3.0

        # Forward progress reward
        self.rewards.track_velocity.weight = 4.0

        # Stability on slopes
        self.rewards.flat_orientation.weight = -1.5
        self.rewards.base_height_l2.weight = -1.0

        # Prevent slipping
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*_foot"]

        # Smooth motion
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7
