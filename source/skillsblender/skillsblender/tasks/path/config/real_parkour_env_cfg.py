"""Real (Phase-2) mixed-terrain parkour environment configuration.

This config is intended for fine-tuning after virtual training. It restores
penalties and emphasizes survival while exposing the policy to mixed terrain.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg


# ==============================================================================
# Mixed terrain for real (Phase-2) adaptation
# ==============================================================================

REAL_PARKOUR_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.5,
    num_rows=12,
    num_cols=18,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Flat base to stabilize locomotion
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        # Gaps for jump adaptation
        "narrow_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.30, 0.70),
            platform_width=2.0,
        ),
        "wide_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.10,
            gap_width_range=(0.70, 1.20),
            platform_width=2.5,
        ),
        # Stairs up/down
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.22),
            step_width=0.30,
            platform_width=2.5,
            border_width=0.25,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.22),
            step_width=0.30,
            platform_width=2.5,
            border_width=0.25,
            holes=False,
        ),
        # Slopes for climb adaptation
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.10, 0.40),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class MyRealParkourSceneCfg(MySceneCfg):
    """Scene configuration with mixed terrain for real adaptation."""

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = REAL_PARKOUR_TERRAIN_CFG


@configclass
class Go2RealParkourEnvCfg(PathEnvCfg):
    """Go2 Phase-2 real parkour environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Scene + robot
        self.scene: MyRealParkourSceneCfg = MyRealParkourSceneCfg(num_envs=4096, env_spacing=3.0)
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Encourage survival and stable locomotion (restore penalties)
        self.rewards.is_terminated.weight = -300.0
        self.rewards.joint_deviation.weight = -0.2

        # Tracking (moderate to allow adaptation)
        self.rewards.track_xy.weight = 6.0
        self.rewards.track_yaw.weight = 3.0
        self.rewards.track_velocity_along_path_exp.weight = 3.5
        self.rewards.stalling_penalty.weight = -1.0

        # Base stability penalties
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_lin_vel_z.weight = -0.5
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.base_acc.weight = -2.5e-4

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-4
        self.rewards.joint_vel_l2.weight = -1.0e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01

        # Contact penalties
        self.rewards.undesired_contacts.weight = -2.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_hip",
            ".*_thigh",
            ".*_calf",
        ]

        # Multi-skill sequence over mixed terrain
        self.commands.path_tracking.path_generator_cfg.skill_sequence = [
            "walk",
            "jump",
            "stairs_up",
            "stairs_down",
            "climb",
            "platform",
        ]

        # Termination thresholds (survival)
        self.terminations.bad_orientation.params["limit_angle"] = 0.9
        self.terminations.path_deviation.params["max_threshold"] = 6.0

        if self.__class__.__name__ == "Go2RealParkourEnvCfg":
            self.disable_zero_weight_rewards()
