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
from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg
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

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.commands.path_tracking = VirtualStairsPathCommandCfg(
            class_type=VirtualStairsPathCommand,
            asset_name="robot",
            resampling_time_range=(3.0, 12.0),
            stairs_prob=0.02,
            step_height_range=(0.12, 0.20),
            step_width_range=(0.25, 0.40),
            num_steps_range=(3, 6),
            direction="up",
            cool_down=0.8,
            debug_vis=True,
        )

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains = {
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
            }
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)

        self.rewards.track_z = RewTerm(
            func=track_path_pos_z_exp,
            weight=10.0,
            params={"std": 0.10, "command_name": "path_tracking"},
        )
        # Feet-clearance shaping: negative weight on low swing-foot penalty => encourage lift.
        self.rewards.feet_clearance = RewTerm(
            func=mdp.feet_height_body,
            weight=-4.0,
            params={
                "command_name": "path_tracking",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_height": -0.10,
                "dis_threshold": 0.30,
                "heading_threshold": 0.60,
                "jump_only": False,
            },
        )

        self.rewards.base_lin_vel_z.weight = 0.0
        self.rewards.joint_torques_l2.weight = -2.0e-5
        self.rewards.track_xy.weight = 5.0
        self.rewards.track_yaw.weight = 2.0
        self.rewards.track_velocity_along_path_exp.weight = 2.8
        self.rewards.stalling_penalty.weight = -1.0

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
        self.commands.path_tracking.direction = "down"
        self.commands.path_tracking.skill_name = "stairs_down"
        self.commands.path_tracking.profile_type = "stairs_down"


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
