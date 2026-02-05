"""Walk skill environment configuration."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Walk Terrain Configuration
# ==============================================================================

WALK_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        # 70% flat terrain for basic walking
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
        # 30% slightly rough terrain for robustness
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25
        ),
    },
)


@configclass
class MyWalkSceneCfg(MySceneCfg):
    """Walk scene configuration with flat and slightly rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        # Update terrain for walk task
        self.terrain.terrain_generator = WALK_TERRAIN_CFG


# ==============================================================================
# Walk Environment Configuration
# ==============================================================================

@configclass
class WalkPathEnvCfg(PathEnvCfg):
    """Walk skill training environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Use walk scene
        self.scene: MyWalkSceneCfg = MyWalkSceneCfg(num_envs=4096, env_spacing=2.5)

        # Walk-specific command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 5.0
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk"]
        self.commands.path_tracking.sampling.yaw_type = "decoupled"
        self.commands.path_tracking.sampling.start_heading = (-3.1415926, 3.1415926)
        self.commands.path_tracking.sampling.end_to_start_pos = (3.0, 6.0, 0.0)
        self.commands.path_tracking.sampling.sample_goal_distance = True

        # Randomized spawn position (local per-env) as randomized path starts
        self.events.reset_base.params["pose_range"]["x"] = (-1.2, 1.2)
        self.events.reset_base.params["pose_range"]["y"] = (-1.2, 1.2)

        # Walk-specific rewards
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity_along_path_exp.weight = 3.0

        # Base stability rewards
        self.rewards.flat_orientation.weight = -1.0
        self.rewards.base_height_l2.weight = -1.0

        # Smooth motion rewards
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7

        # Gait rewards for trotting
        self.rewards.feet_air_time.weight = 0.8
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_gait.weight = 1.5
        self.rewards.joint_mirror.weight = -0.4
        self.rewards.air_time_variance.weight = -0.8
        self.rewards.feet_contact_balance.weight = -1.0
