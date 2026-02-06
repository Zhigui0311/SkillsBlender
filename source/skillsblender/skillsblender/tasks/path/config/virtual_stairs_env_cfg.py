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
from skillsblender.tasks.path.mdp.commands.virtual_stairs_path_command import (
    VirtualStairsPathCommand,
    VirtualStairsPathCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_path_pos_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    command_name: str = "path_tracking",
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    err = command.metrics.get("error_pos_z", torch.zeros(env.num_envs, device=env.device))
    return torch.exp(-torch.square(err) / (std**2))


@configclass
class VirtualRewards(RewardsCfg):
    """Reward shaping for virtual stairs: strong tracking, minimal penalties."""

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
        params={"std": 0.6, "command_name": "path_tracking", "desired_speed": 0.9},
    )
    track_z = RewTerm(
        func=track_path_pos_z_exp,
        weight=8.0,
        params={"std": 0.10, "command_name": "path_tracking"},
    )

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
class VirtualStairsCurriculumCfg:
    virtual_difficulty = CurrTerm(
        func=mdp.curriculum_virtual_skill_difficulty,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 55.0,
            "prob_step": 0.002,
            "max_prob": 0.08,
            "height_step": 0.01,
            "max_height": 0.30,
            "length_step": 0.10,
            "max_length": 2.8,
        },
    )
    virtual_steps = CurrTerm(
        func=mdp.curriculum_virtual_stairs_steps,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 60.0,
            "step_increase": 1,
            "max_steps": 10,
        },
    )


@configclass
class Go2VirtualStairsEnvCfg(PathEnvCfg):
    """Virtual stairs training on flat terrain only."""
    curriculum: VirtualStairsCurriculumCfg = VirtualStairsCurriculumCfg()
    rewards: VirtualRewards = VirtualRewards()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.commands.path_tracking = VirtualStairsPathCommandCfg(
            class_type=VirtualStairsPathCommand,
            asset_name="robot",
            resampling_time_range=(1.0, 3.0),
            virtual_prob=0.50,
            stairs_up_prob=0.5,
            step_height=0.15,
            step_width=0.30,
            stairs_length_range=(1.2, 2.4),
            max_up_height=0.40,
            max_down_depth=0.25,
            down_pitch_deg=-10.0,
            cool_down=0.3,
            debug_vis=True,
        )

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains = {
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
            }
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)

        # Feet-clearance shaping: strong positive weight to encourage high stepping.
        self.rewards.feet_clearance = RewTerm(
            func=mdp.feet_height_body,
            weight=6.0,
            params={
                "command_name": "path_tracking",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_height": 0.15,
                "dis_threshold": 0.30,
                "heading_threshold": 0.60,
                "jump_only": False,
            },
        )

        self.terminations.path_deviation.params["min_threshold"] = 0.8
        self.terminations.path_deviation.params["max_threshold"] = 6.0
        self.terminations.bad_orientation.params["limit_angle"] = 0.95

        if self.__class__.__name__ == "Go2VirtualStairsEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2VirtualStairsDownEnvCfg(Go2VirtualStairsEnvCfg):
    """Virtual stairs-down training on flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        # Force down-stairs by setting up-probability to zero.
        self.commands.path_tracking.stairs_up_prob = 0.0


@configclass
class Go2VirtualStairsEnvCfg_PLAY(Go2VirtualStairsEnvCfg):
    """Play configuration for virtual stairs debugging."""

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
