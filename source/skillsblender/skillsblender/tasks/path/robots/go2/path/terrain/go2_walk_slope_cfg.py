"""Unitree Go2 walk skill with mild slope terrain."""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .go2_walk_cfg import Go2WalkEnvCfg


WALK_SLOPE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        # Keep most terrain easy so walk remains dominant.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.55),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        # Mild up/down slopes only (not steep climb terrain).
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.04, 0.18),
            platform_width=2.5,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.04, 0.18),
            platform_width=2.5,
            border_width=0.25,
        ),
    },
)


@configclass
class Go2WalkSlopeEnvCfg(Go2WalkEnvCfg):
    """Go2 walk training on flat + mild slopes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = WALK_SLOPE_TERRAIN_CFG

        # Slightly lower desired speed for slope robustness.
        self.rewards.track_velocity_along_path_exp.weight = 4.0
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.9
        self.rewards.flat_orientation.weight = -0.9
        self.rewards.feet_slide.weight = -0.9

        if self.__class__.__name__ == "Go2WalkSlopeEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2WalkSlopeEnvCfg_PLAY(Go2WalkSlopeEnvCfg):
    """Visualization config for walk+slope."""

    def __post_init__(self):
        super().__post_init__()
