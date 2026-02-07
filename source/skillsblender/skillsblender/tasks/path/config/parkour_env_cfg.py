"""Parkour (multi-skill fusion) environment configurations.

This module provides two training approaches for multi-skill parkour:
1. Direct Training: Train a unified policy from scratch on mixed terrain
2. Distillation Training: Use pre-trained skill experts for knowledge distillation

Terrain Types:
- Flat (warm-up/recovery zones)
- Gaps (jumping)
- Stairs (ascending/descending)
- Slopes (climbing)
- Rough terrain (stability)
- Boxes/obstacles (agility)
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, MySceneCfg
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Parkour Terrain Configurations
# ==============================================================================

# Level 1: Easy terrain for initial training
PARKOUR_TERRAIN_EASY_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.5),  # Easy difficulty
    use_cache=False,
    sub_terrains={
        # 30% flat terrain (more recovery zones)
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),

        # 15% small gaps
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.2, 0.5),  # Smaller gaps
            platform_width=2.5,
        ),

        # 20% easy stairs
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.10),  # Lower steps
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # 15% gentle slopes
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.05, 0.20),  # Gentler slopes
            platform_width=2.5,
            border_width=0.25
        ),

        # 20% light rough terrain
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.05),  # Less noise
            noise_step=0.01,
            border_width=0.25
        ),
    },
)

# Level 2: Medium terrain for progressive training
PARKOUR_TERRAIN_MEDIUM_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.3, 0.7),
    use_cache=False,
    sub_terrains={
        # 20% flat terrain
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),

        # 20% medium gaps
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.35, 0.70),
            platform_width=2.0,
        ),

        # 20% medium stairs
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.06, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # 15% medium slopes
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.10, 0.30),
            platform_width=2.0,
            border_width=0.25
        ),

        # 10% rough terrain
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.08),
            noise_step=0.02,
            border_width=0.25
        ),

        # 15% boxes
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.40,
            grid_height_range=(0.05, 0.15),
            platform_width=2.0
        ),
    },
)

# Level 3: Hard terrain for advanced training
PARKOUR_TERRAIN_HARD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.5, 1.0),
    use_cache=False,
    sub_terrains={
        # 10% flat terrain (minimal recovery)
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),

        # 20% large gaps
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.5, 0.9),
            platform_width=1.8,
        ),

        # 20% challenging stairs
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.08, 0.20),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),

        # 15% steep slopes
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.15, 0.40),
            platform_width=2.0,
            border_width=0.25
        ),

        # 15% rough terrain
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.03, 0.12),
            noise_step=0.02,
            border_width=0.25
        ),

        # 20% tall boxes
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.08, 0.25),
            platform_width=2.0
        ),
    },
)


# ==============================================================================
# Parkour Scene Configuration
# ==============================================================================

@configclass
class ParkourSceneCfg(MySceneCfg):
    """Parkour scene with mixed terrain and obstacles."""

    # Low roof for crouch training
    crouch_roof = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CrouchRoof",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(6.5, 0.0, 0.36),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.8, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        # Default to medium terrain
        self.terrain.terrain_generator = PARKOUR_TERRAIN_MEDIUM_CFG


# ==============================================================================
# Curriculum Configuration for Parkour
# ==============================================================================

@configclass
class ParkourCurriculumCfg:
    """Progressive curriculum for parkour training."""

    # Stairs difficulty progression
    stairs_difficulty = CurrTerm(
        func=mdp.curriculum_stairs_difficulty,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 60.0,
            "initial_step_height": 0.06,
            "final_step_height": 0.18,
            "step_height_step": 0.01,
            "initial_stairs_len": 1.5,
            "final_stairs_len": 2.5,
            "stairs_len_step": 0.1,
        },
    )

    # Stairs height by tracking error (new curriculum)
    stairs_tracking = CurrTerm(
        func=mdp.curriculum_stairs_height_by_tracking,
        params={
            "command_name": "path_tracking",
            "error_threshold": 0.25,
            "success_rate_threshold": 0.65,
            "initial_step_height": 0.06,
            "final_step_height": 0.20,
            "step_height_increment": 0.015,
            "window_size": 100,
        },
    )


# ==============================================================================
# Direct Training Configuration (From Scratch)
# ==============================================================================

@configclass
class ParkourDirectEnvCfg(PathEnvCfg):
    """Direct multi-skill parkour training from scratch.

    This configuration trains a unified policy to handle all terrain types
    without using pre-trained expert policies. Uses curriculum learning
    to progressively increase difficulty.

    Training Strategy:
    1. Start with easy terrain (more flat areas, smaller obstacles)
    2. Gradually increase terrain difficulty via curriculum
    3. Use skill-conditioned rewards for different terrain phases
    4. Apply transition blending for smooth skill switching
    """

    curriculum: ParkourCurriculumCfg = ParkourCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Scene configuration
        self.scene: ParkourSceneCfg = ParkourSceneCfg(num_envs=4096, env_spacing=2.5)
        # Start with easy terrain
        self.scene.terrain.terrain_generator = PARKOUR_TERRAIN_EASY_CFG

        # Path command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 8.0
        self.commands.path_tracking.ranges.z_clip = 0.5  # Independent Z scaling
        self.commands.path_tracking.ranges.transition_window_s = 0.3  # Enable transition blending
        self.commands.path_tracking.sampling.yaw_type = "fixed"
        self.commands.path_tracking.sampling.start_heading = (-0.3, 0.3)
        self.commands.path_tracking.path_generator_cfg.skill_sequence = [
            "walk",
            "jump",
            "stairs_up",
            "stairs_down",
            "climb",
            "crouch",
        ]

        # === Core Task Rewards ===
        self.rewards.track_xy.weight = 8.0
        self.rewards.track_xy.params["std"] = 0.4
        self.rewards.track_yaw.weight = 4.0
        self.rewards.track_yaw.params["std"] = 0.4
        self.rewards.track_velocity_along_path_exp.weight = 5.0
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.9

        # === Skill-Conditioned Z Tracking ===
        self.rewards.track_z_walk.weight = 1.5
        self.rewards.track_z_walk.params["std"] = 0.25
        self.rewards.track_z_jump.weight = 4.0
        self.rewards.track_z_jump.params["std"] = 0.12
        self.rewards.track_z_stairs.weight = 3.0
        self.rewards.track_z_stairs.params["std"] = 0.08

        # === Preparation Reward ===
        self.rewards.preparation_reward.weight = 2.0
        self.rewards.preparation_reward.params["scan_dist"] = 1.2
        self.rewards.preparation_reward.params["z_threshold"] = 0.08

        # === Base Stability ===
        self.rewards.flat_orientation.weight = -1.5
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.34
        self.rewards.base_lin_vel_z.weight = -0.3
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.base_acc.weight = -2e-4

        # === Joint Penalties ===
        self.rewards.joint_torques_l2.weight = -1.5e-4
        self.rewards.joint_vel_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0
        self.rewards.joint_deviation.weight = -0.2

        # === Action Penalties ===
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.applied_torque_limits.weight = -0.1

        # === Feet Rewards ===
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_stumble.weight = -1.5
        self.rewards.feet_stumble_terrain.weight = -3.0  # Enhanced terrain penalty

        # === Contact Penalties ===
        self.rewards.undesired_contacts.weight = -2.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_hip", ".*_thigh", ".*_calf"
        ]

        # === Termination Penalty ===
        self.rewards.is_terminated.weight = -350.0

        # === Crouch Phase ===
        self.rewards.crouch_base_height_phase.weight = -3.0
        self.rewards.crouch_base_height_phase.params["target_height"] = 0.24

        # === Events ===
        self.events.reset_base.params["pose_range"]["x"] = (-0.3, 0.3)
        self.events.reset_base.params["pose_range"]["y"] = (-0.3, 0.3)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)

        # === Termination ===
        self.terminations.bad_orientation.params["limit_angle"] = 1.0  # ~57 degrees

        # Episode settings
        self.episode_length_s = 12.0


@configclass
class ParkourDirectEnvCfg_PLAY(ParkourDirectEnvCfg):
    """Play configuration for direct parkour training."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 30.0
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0
        self.sim.render_interval = 2
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = PARKOUR_TERRAIN_HARD_CFG
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum = None
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.origin_type = "env"
        self.commands.path_tracking.debug_vis = True
        self.observations.policy.enable_corruption = False


# ==============================================================================
# Distillation Training Configuration
# ==============================================================================

@configclass
class ParkourDistillEnvCfg(PathEnvCfg):
    """Multi-skill parkour training with policy distillation.

    This configuration uses pre-trained expert policies for each skill
    and distills their knowledge into a unified student policy.

    Training Strategy:
    1. Load pre-trained expert policies (walk, jump, stairs, climb, crouch)
    2. Use skill phase detection to select appropriate expert
    3. Add distillation loss to match expert actions
    4. Gradually reduce distillation weight as student improves

    Expert Checkpoints (configure in training script):
    - walk_expert: logs/rsl_rl/go2-path-walk/*/model_*.pt
    - jump_expert: logs/rsl_rl/go2-path-jump/*/model_*.pt
    - stairs_expert: logs/rsl_rl/go2-path-stairs/*/model_*.pt
    - climb_expert: logs/rsl_rl/go2-path-climb/*/model_*.pt
    - crouch_expert: logs/rsl_rl/go2-path-crouch/*/model_*.pt
    """

    def __post_init__(self):
        super().__post_init__()

        # Scene configuration - use harder terrain since we have expert guidance
        self.scene: ParkourSceneCfg = ParkourSceneCfg(num_envs=4096, env_spacing=2.5)
        self.scene.terrain.terrain_generator = PARKOUR_TERRAIN_MEDIUM_CFG

        # Path command configuration
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.ranges.default_path_len = 8.0
        self.commands.path_tracking.ranges.z_clip = 0.5
        self.commands.path_tracking.ranges.transition_window_s = 0.2
        self.commands.path_tracking.sampling.yaw_type = "fixed"
        self.commands.path_tracking.path_generator_cfg.skill_sequence = [
            "walk",
            "jump",
            "stairs_up",
            "stairs_down",
            "climb",
            "crouch",
        ]

        # === Task Rewards (lower weights, distillation provides main signal) ===
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.5
        self.rewards.track_velocity_along_path_exp.weight = 3.0

        # === Skill-Conditioned Z Tracking ===
        self.rewards.track_z_walk.weight = 1.0
        self.rewards.track_z_jump.weight = 2.5
        self.rewards.track_z_stairs.weight = 2.0

        # === Preparation Reward ===
        self.rewards.preparation_reward.weight = 1.5

        # === Base Stability ===
        self.rewards.flat_orientation.weight = -1.0
        self.rewards.base_height_l2.weight = -0.8
        self.rewards.base_lin_vel_z.weight = -0.2
        self.rewards.base_ang_vel_xy.weight = -0.03

        # === Joint Penalties ===
        self.rewards.joint_torques_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2e-7
        self.rewards.joint_pos_limits.weight = -8.0
        self.rewards.joint_vel_limits.weight = -0.8

        # === Action Penalties ===
        self.rewards.action_rate_l2.weight = -0.008

        # === Feet Rewards ===
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.feet_slide.weight = -1.5
        self.rewards.feet_stumble_terrain.weight = -2.0

        # === Contact Penalties ===
        self.rewards.undesired_contacts.weight = -2.0

        # === Termination ===
        self.rewards.is_terminated.weight = -300.0
        self.terminations.bad_orientation.params["limit_angle"] = 1.1

        # Episode settings
        self.episode_length_s = 12.0


@configclass
class ParkourDistillEnvCfg_PLAY(ParkourDistillEnvCfg):
    """Play configuration for distillation parkour training."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 30.0
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0
        self.sim.render_interval = 2
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = PARKOUR_TERRAIN_HARD_CFG
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.origin_type = "env"
        self.commands.path_tracking.debug_vis = True
        self.observations.policy.enable_corruption = False
