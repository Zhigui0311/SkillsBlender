"""Unitree Go2 parkour (multi-skill fusion) configurations.

Two training approaches:
1. Direct Training (go2_parkour_direct_cfg): Train from scratch with curriculum
2. Distillation Training (go2_parkour_distill_cfg): Use expert policies for guidance
"""

from skillsblender.tasks.path.config.parkour_env_cfg import (
    ParkourDirectEnvCfg,
    ParkourDirectEnvCfg_PLAY,
    ParkourDistillEnvCfg,
    ParkourDistillEnvCfg_PLAY,
    PARKOUR_TERRAIN_EASY_CFG,
    PARKOUR_TERRAIN_MEDIUM_CFG,
    PARKOUR_TERRAIN_HARD_CFG,
)
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp


# ==============================================================================
# Go2 Direct Training Configuration (From Scratch)
# ==============================================================================

@configclass
class Go2ParkourDirectEnvCfg(ParkourDirectEnvCfg):
    """Unitree Go2 parkour direct training configuration.

    Train a unified multi-skill policy from scratch using curriculum learning.

    Training Command:
        python scripts/rsl_rl/train.py --task go2-parkour-direct-v0 --num_envs 4096

    Key Features:
    - Progressive terrain difficulty (easy -> medium -> hard)
    - Skill-conditioned Z tracking rewards
    - Transition blending for smooth skill switching
    - Curriculum learning for stairs height
    """

    def __post_init__(self):
        super().__post_init__()

        # Set Go2 robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Observation scaling
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Go2-specific reward tuning
        self.rewards.is_terminated.weight = -350.0
        self.rewards.joint_deviation.weight = -0.25

        # Joint symmetry for stable gait
        self.rewards.joint_mirror.weight = -0.3
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Torque limits
        self.rewards.applied_torque_limits.weight = -0.15

        if self.__class__.__name__ == "Go2ParkourDirectEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2ParkourDirectEnvCfg_PLAY(Go2ParkourDirectEnvCfg):
    """Go2 parkour direct training PLAY configuration."""

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
# Go2 Distillation Training Configuration
# ==============================================================================

@configclass
class Go2ParkourDistillEnvCfg(ParkourDistillEnvCfg):
    """Unitree Go2 parkour distillation training configuration.

    Train a unified policy using knowledge distillation from expert policies.

    Training Command:
        python scripts/rsl_rl/train_distill.py --task go2-parkour-distill-v0 \\
            --walk_expert logs/rsl_rl/go2-path-walk/*/model_*.pt \\
            --jump_expert logs/rsl_rl/go2-path-jump/*/model_*.pt \\
            --stairs_expert logs/rsl_rl/go2-path-stairs/*/model_*.pt \\
            --climb_expert logs/rsl_rl/go2-path-climb/*/model_*.pt \\
            --crouch_expert logs/rsl_rl/go2-path-crouch/*/model_*.pt

    Key Features:
    - Uses pre-trained expert policies for each skill
    - Skill phase detection selects appropriate expert
    - Distillation loss matches expert actions
    - Faster convergence than direct training
    """

    def __post_init__(self):
        super().__post_init__()

        # Set Go2 robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Observation scaling
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Go2-specific reward tuning
        self.rewards.is_terminated.weight = -300.0
        self.rewards.joint_deviation.weight = -0.2

        # Joint symmetry
        self.rewards.joint_mirror.weight = -0.25
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        if self.__class__.__name__ == "Go2ParkourDistillEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2ParkourDistillEnvCfg_PLAY(Go2ParkourDistillEnvCfg):
    """Go2 parkour distillation training PLAY configuration."""

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


# ==============================================================================
# Go2 Parkour with CNN Path Encoder
# ==============================================================================

@configclass
class Go2ParkourCNNEnvCfg(Go2ParkourDirectEnvCfg):
    """Go2 parkour with CNN path encoder for better spatial understanding.

    Uses 1D-CNN to encode path slice observations, capturing spatial patterns
    in the upcoming terrain for better anticipation and planning.

    Training Command:
        python scripts/rsl_rl/train.py --task go2-parkour-cnn-v0 --num_envs 4096
    """

    def __post_init__(self):
        super().__post_init__()

        # Use medium terrain (CNN helps with harder terrain)
        self.scene.terrain.terrain_generator = PARKOUR_TERRAIN_MEDIUM_CFG

        # Slightly longer lookahead for CNN
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 32

        if self.__class__.__name__ == "Go2ParkourCNNEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2ParkourCNNEnvCfg_PLAY(Go2ParkourCNNEnvCfg):
    """Go2 parkour CNN PLAY configuration."""

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
