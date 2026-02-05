from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import skillsblender.tasks.path.mdp as mdp
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg
from skillsblender.tasks.path.mdp.commands.virtual_jump_path_command import (
    VirtualJumpPathCommand,
    VirtualJumpPathCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_path_pos_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    command_name: str = "path_tracking",
) -> torch.Tensor:
    """Track virtual Z target from command waypoints with an exponential kernel."""
    command = env.command_manager.get_term(command_name)
    err = command.metrics.get("error_pos_z", torch.zeros(env.num_envs, device=env.device))
    return torch.exp(-torch.square(err) / (std**2))


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

    def __post_init__(self):
        super().__post_init__()

        # Required robot binding (explicitly set to avoid missing asset wiring).
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Bind virtual jump command.
        self.commands.path_tracking = VirtualJumpPathCommandCfg(
            class_type=VirtualJumpPathCommand,
            asset_name="robot",
            resampling_time_range=(3.0, 12.0),
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
            jump_prob=0.02,
            jump_height_range=(0.3, 0.5),
            jump_length_range=(0.8, 1.2),
            cool_down=0.8,
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

        # Reward shaping: strong Z-following, allow vertical impulse, reduce torque penalty.
        self.rewards.track_z = RewTerm(
            func=track_path_pos_z_exp,
            weight=10.0,
            params={"std": 0.10, "command_name": "path_tracking"},
        )
        self.rewards.base_lin_vel_z.weight = 0.0
        self.rewards.joint_torques_l2.weight = -2.0e-5

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
