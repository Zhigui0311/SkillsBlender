"""Crouch skill environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
from skillsblender.tasks.path.utils.crouch_terrain_spawner import spawn_crouch_obstacles
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Crouch Terrain Configuration
# ==============================================================================

CROUCH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        # 30% flat terrain
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        # 70% flat with low obstacles (for crouching under)
        # Note: Actual obstacles will be spawned separately
        "crouch_area": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
    },
)


@configclass
class MyCrouchSceneCfg(MySceneCfg):
    """Crouch scene configuration with low obstacles."""

    def __post_init__(self):
        super().__post_init__()
        # Update terrain for crouch task
        self.terrain.terrain_generator = CROUCH_TERRAIN_CFG

        # Note: Low obstacles for crouching will be spawned using
        # spawn_crouch_obstacles() function in the environment


# ==============================================================================
# Crouch Environment Configuration
# ==============================================================================

@configclass
class CrouchPathEnvCfg(PathEnvCfg):
    """Crouch skill training environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Use crouch scene
        self.scene: MyCrouchSceneCfg = MyCrouchSceneCfg(num_envs=4096, env_spacing=2.5)

        # Crouch-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 64
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.crouch_params.crouch_len = 2.0
        self.commands.path_tracking.crouch_params.base_height_ref = 0.24  # Lower height

        # Crouch-specific rewards
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity.weight = 2.0  # Slower speed

        # Low height reward (encourage crouching)
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.24

        # Stability in low stance
        self.rewards.flat_orientation.weight = -1.5

        # Avoid hitting obstacles with body
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", ".*_hip", ".*_thigh"
        ]

        # Smooth motion
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7
