"""Blender (multi-skill fusion) environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Blender Terrain Configuration (Mixed Skills)
# ==============================================================================

BLENER_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        # 15% flat terrain
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),

        # 20% gaps for jumping
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.3, 0.8),
            platform_width=2.0,
        ),

        # 20% stairs
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # 15% slopes for climbing
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.10, 0.40),
            platform_width=2.0,
            border_width=0.25
        ),

        # 15% rough terrain
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25
        ),

        # 15% boxes
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.20),
            platform_width=2.0
        ),
    },
)


@configclass
class MyBlenerSceneCfg(MySceneCfg):
    """Blender scene configuration with mixed terrain types."""

    def __post_init__(self):
        super().__post_init__()
        # Update terrain for blender task
        self.terrain.terrain_generator = BLENER_TERRAIN_CFG


# ==============================================================================
# Blender Environment Configuration
# ==============================================================================

@configclass
class BlenerPathEnvCfg(PathEnvCfg):
    """Multi-skill blender training environment configuration.

    This environment trains the robot to handle multiple skills in sequence,
    such as walk -> jump -> stairs -> climb.
    """

    def __post_init__(self):
        super().__post_init__()

        # Use blender scene
        self.scene: MyBlenerSceneCfg = MyBlenerSceneCfg(num_envs=4096, env_spacing=2.5)

        # Blender-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 8.0  # Longer paths
        self.commands.path_tracking.path_generator_cfg.skill_sequence = [
            "walk",
            "jump",
            "stairs_up",
            "climb",
            "crouch",
        ]

        # Balanced rewards for all skills
        self.rewards.track_xy.weight = 6.0
        self.rewards.track_yaw.weight = 3.0
        self.rewards.track_velocity_along_path_exp.weight = 4.0

        # Base stability (important for skill transitions)
        self.rewards.flat_orientation.weight = -1.0
        self.rewards.base_height_l2.weight = -1.0

        # Feet rewards (for all skills)
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.feet_air_time.params["threshold"] = 0.4

        # Prevent undesired contacts
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_hip", ".*_thigh", ".*_calf"
        ]

        # Smooth motion (important for transitions)
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7

        # Joint limits
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0

        # Termination: more lenient for complex terrain
        self.terminations.bad_orientation.params["limit_angle"] = 1.2  # ~69 degrees
