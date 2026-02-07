from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import skillsblender.tasks.path.mdp as mdp
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg, RewardsCfg, GO2_JOINT_NAMES
from skillsblender.tasks.path.mdp.commands.virtual_jump_path_command import (
    VirtualJumpPathCommand,
    VirtualJumpPathCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
class VirtualJumpRewards(RewardsCfg):
    """Reward shaping for virtual training: strong tracking, minimal penalties."""

    # Survival is still penalized to avoid trivial falling.
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Task tracking (strong).
    track_xy = RewTerm(
        func=mdp.track_path_pos_xy_exp,
        weight=8.0,
        params={"std": 0.5, "command_name": "path_tracking"},
    )
    track_yaw = RewTerm(
        func=mdp.track_path_heading_exp,
        weight=4.0,
        params={"std": 0.5, "command_name": "path_tracking"},
    )
    track_velocity_along_path_exp = RewTerm(
        func=mdp.track_velocity_along_path_exp,
        weight=4.0,
        params={"std": 0.6, "command_name": "path_tracking", "desired_speed": 1.0},
    )
    # Virtual Z tracking (hallucinated path).
    track_z = RewTerm(
        func=mdp.track_path_pos_z_exp,
        weight=12.0,
        params={"std": 0.10, "command_name": "path_tracking"},
    )

    # Remove penalties for fast skill acquisition (can be re-enabled in Phase 2).
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={"target_height": 0.34, "asset_cfg": SceneEntityCfg("robot")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0, params={})
    undesired_contacts_hip = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Head_upper", "Head_lower", "RL_hip", "RR_hip"],
            ),
            "threshold": 1.0,
        },
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES), "soft_ratio": 1.0},
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]]},
    )
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)


@configclass
class VirtualJumpCurriculumCfg:
    virtual_difficulty = CurrTerm(
        func=mdp.curriculum_virtual_skill_difficulty,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 55.0,
            "prob_step": 0.002,
            "max_prob": 0.08,
            "height_step": 0.02,
            "max_height": 0.70,
            "length_step": 0.10,
            "max_length": 1.8,
        },
    )


@configclass
class Go2VirtualJumpEnvCfg(PathEnvCfg):
    """Go2 virtual-jump environment on forced flat terrain.

    This is terrain-decoupled instruction hallucination training:
    the jump profile exists in command path slices, not in real terrain geometry.
    """
    curriculum: VirtualJumpCurriculumCfg = VirtualJumpCurriculumCfg()
    rewards: VirtualJumpRewards = VirtualJumpRewards()

    def __post_init__(self):
        super().__post_init__()

        # Required robot binding (explicitly set to avoid missing asset wiring).
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Bind virtual jump command.
        self.commands.path_tracking = VirtualJumpPathCommandCfg(
            class_type=VirtualJumpPathCommand,
            asset_name="robot",
            # Shorter resample window + higher trigger rate for faster skill acquisition.
            resampling_time_range=(1.0, 3.0),
            ranges=mdp.commands.PathCommandCfg.Ranges(
                num_waypoints=80,
                num_lookahead_waypoints=24,
                waypoint_reach_threshold=0.8,
                default_path_len=5.0,
            ),
            sampling=mdp.commands.PathCommandCfg.Sampling(
                path_type="linear",
                height_change=False,
                end_to_start_pos=(4.0, 6.0, 0.0),
                yaw_type="along_path",
                start_heading=(-3.1415926, 3.1415926),
                end_heading=(0.0, 0.0),
                sample_goal_distance=True,
            ),
            yaw_mode="interp",
            virtual_prob=0.50,
            jump_height_range=(0.3, 0.5),
            gap_width_range=(0.8, 1.2),
            cool_down=0.3,
            skill_name="jump",
            profile_type="parabola_up",
            debug_vis=True,
        )

        # Force absolute flat terrain for safe virtual-only training.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains = {
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
            }
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)

        # Minimal gait regularization for jump stability (can be relaxed later).
        self.rewards.joint_mirror.weight = -0.2
        self.rewards.joint_vel_l2.weight = -1.0e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.005

        # Keep base tracking objectives active.
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity_along_path_exp.weight = 3.0
        self.rewards.stalling_penalty.weight = -1.0

        # Terminations tuned for virtual profile tracking.
        self.terminations.path_deviation.params["min_threshold"] = 0.8
        self.terminations.path_deviation.params["max_threshold"] = 6.0
        self.terminations.bad_orientation.params["limit_angle"] = 0.9

        if self.__class__.__name__ == "Go2VirtualJumpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2VirtualJumpEnvCfg_PLAY(Go2VirtualJumpEnvCfg):
    """Play configuration for virtual jump debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0
        self.scene.terrain.max_init_terrain_level = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum = None
        self.commands.path_tracking.debug_vis = True
        self.observations.policy.enable_corruption = False
