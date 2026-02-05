"""Stairs skill environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Stairs Terrain Configuration
# ==============================================================================

STAIRS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        # 60% pyramid stairs (up)
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.6,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 20% inverted pyramid stairs (down)
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class MyStairsSceneCfg(MySceneCfg):
    """Stairs scene configuration with pyramid stairs terrain."""

    def __post_init__(self):
        super().__post_init__()
        # Update terrain for stairs task
        self.terrain.terrain_generator = STAIRS_TERRAIN_CFG


# ==============================================================================
# Stairs Environment Configuration
# ==============================================================================

@configclass
class StairsPathEnvCfg(PathEnvCfg):
    """Stairs skill training environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Use stairs scene
        self.scene: MyStairsSceneCfg = MyStairsSceneCfg(num_envs=4096, env_spacing=2.5)
        self.commands.path_tracking.class_type = mdp.commands.StairsPathCommand
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk", "stairs_up", "stairs_down", "walk"]
        self.commands.path_tracking.sampling.yaw_type = "fixed"
        self.commands.path_tracking.sampling.start_heading = (0.0, 0.0)

        # Stairs-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 5.5
        self.commands.path_tracking.stairs_params.start_dist_range = (0.8, 1.2)
        self.commands.path_tracking.stairs_params.stairs_len = 1.8
        self.commands.path_tracking.stairs_params.step_length = 0.25
        self.commands.path_tracking.stairs_params.step_height = 0.10

        # Keep starts aligned with the stair axis for stable up/down training.
        self.events.reset_base.params["pose_range"]["x"] = (-0.25, 0.25)
        self.events.reset_base.params["pose_range"]["y"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)

        # Stairs-specific rewards
        self.rewards.track_xy.weight = 6.0
        self.rewards.track_yaw.weight = 3.0
        self.rewards.track_velocity_along_path_exp.weight = 2.0
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.7

        # Clearance reward for lifting feet
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.stairs_feet_air_time_phase.weight = 2.5
        self.rewards.stairs_feet_air_time_phase.params["threshold"] = 0.3

        # Stability rewards
        self.rewards.flat_orientation.weight = -2.0
        # Disable constant-height penalty: climbing stairs needs changing base height.
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.feet_slide.weight = -2.0

        # Smooth motion
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7
