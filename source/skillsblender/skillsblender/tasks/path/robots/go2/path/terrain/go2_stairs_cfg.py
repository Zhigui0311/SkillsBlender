"""Unitree Go2 stairs skill configuration."""

from skillsblender.tasks.path.config.stairs_env_cfg import StairsPathEnvCfg
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp


_STAIRS_INIT_STEP_H = 0.08
_STAIRS_FINAL_STEP_H = 0.14
_STAIRS_INIT_LEN = 1.6
_STAIRS_FINAL_LEN = 2.2


@configclass
class StairsCurriculumCfg:
    """Stairs-specific curriculum learning configuration."""

    stairs_difficulty = CurrTerm(
        func=mdp.curriculum_stairs_difficulty,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 65.0,
            "initial_step_height": _STAIRS_INIT_STEP_H,
            "final_step_height": _STAIRS_FINAL_STEP_H,
            "step_height_step": 0.005,
            "initial_stairs_len": _STAIRS_INIT_LEN,
            "final_stairs_len": _STAIRS_FINAL_LEN,
            "stairs_len_step": 0.05,
        },
    )


@configclass
class Go2StairsEnvCfg(StairsPathEnvCfg):
    """Unitree Go2 stairs skill training configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Set Go2 robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # General
        self.rewards.is_terminated.weight = -300.0
        self.rewards.joint_deviation.weight = -0.25

        # Obs scaling
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Task rewards
        self.rewards.track_xy.weight = 7.0
        self.rewards.track_yaw.weight = 3.0
        self.rewards.track_velocity_along_path_exp.weight = 3.0
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.7

        # Base stability
        self.rewards.flat_orientation.weight = -1.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_lin_vel_z.weight = -0.5
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.base_acc.weight = -2.5e-4

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-4
        self.rewards.joint_vel_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0

        # Feet rewards: high clearance for stairs
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.stairs_feet_air_time_phase.weight = 3.0
        self.rewards.stairs_feet_air_time_phase.params["threshold"] = 0.3
        self.rewards.feet_slide.weight = -3.0

        # Contact penalties
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_hip", ".*_thigh", ".*_calf"
        ]

        # Command parameters
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk", "stairs_up", "stairs_down", "walk"]

        if self.__class__.__name__ == "Go2StairsEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2StairsCurEnvCfg(Go2StairsEnvCfg):
    """Unitree Go2 stairs training configuration with curriculum."""

    curriculum: StairsCurriculumCfg = StairsCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.path_tracking.stairs_params.step_height = _STAIRS_INIT_STEP_H
        self.commands.path_tracking.stairs_params.stairs_len = _STAIRS_INIT_LEN

        if self.__class__.__name__ == "Go2StairsCurEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2StairsEnvCfg_PLAY(Go2StairsEnvCfg):
    """Unitree Go2 stairs PLAY configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0
        self.sim.render_interval = 2
        self.scene.terrain.max_init_terrain_level = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum = None
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.origin_type = "env"
        self.commands.path_tracking.debug_vis = True
        self.observations.policy.enable_corruption = False


@configclass
class Go2StairsCurEnvCfg_PLAY(Go2StairsCurEnvCfg):
    """Unitree Go2 stairs curriculum PLAY configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None
