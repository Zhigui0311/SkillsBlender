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
from skillsblender.tasks.path.mdp.commands.virtual_crouch_path_command import (
    VirtualCrouchPathCommand,
    VirtualCrouchPathCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
class VirtualCrouchRewards(RewardsCfg):
    """Reward shaping for virtual crouch: strong tracking, minimal penalties."""

    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

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
        params={"std": 0.6, "command_name": "path_tracking", "desired_speed": 0.8},
    )
    track_z = RewTerm(
        func=mdp.track_path_pos_z_exp,
        weight=12.0,
        params={"std": 0.10, "command_name": "path_tracking"},
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={"target_height": 0.24, "asset_cfg": SceneEntityCfg("robot")},
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
class VirtualCrouchCurriculumCfg:
    virtual_difficulty = CurrTerm(
        func=mdp.curriculum_virtual_skill_difficulty,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 55.0,
            "prob_step": 0.002,
            "max_prob": 0.08,
            "height_step": 0.015,
            "max_height": 0.40,
            "length_step": 0.15,
            "max_length": 2.6,
        },
    )


@configclass
class Go2VirtualCrouchEnvCfg(PathEnvCfg):
    """Virtual crouch training on flat terrain only."""
    curriculum: VirtualCrouchCurriculumCfg = VirtualCrouchCurriculumCfg()
    rewards: VirtuaCrouchlRewards = VirtualCrouchRewards()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.commands.path_tracking = VirtualCrouchPathCommandCfg(
            class_type=VirtualCrouchPathCommand,
            asset_name="robot",
            resampling_time_range=(1.0, 3.0),
            virtual_prob=0.50,
            crouch_depth_range=(0.22, 0.34),
            crouch_length_range=(1.2, 2.0),
            cool_down=0.3,
            debug_vis=True,
        )

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains = {
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
            }
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)

        # Mild gait regularization to prevent joint jitter during crouch.
        self.rewards.joint_mirror.weight = -0.3
        self.rewards.joint_vel_l2.weight = -1.0e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity_along_path_exp.weight = 2.5
        self.rewards.stalling_penalty.weight = -1.0

        self.terminations.path_deviation.params["min_threshold"] = 0.8
        self.terminations.path_deviation.params["max_threshold"] = 6.0
        self.terminations.bad_orientation.params["limit_angle"] = 1.0

        if self.__class__.__name__ == "Go2VirtualCrouchEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2VirtualCrouchEnvCfg_PLAY(Go2VirtualCrouchEnvCfg):
    """Play configuration for virtual crouch debugging."""

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
