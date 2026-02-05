"""Unitree Go2 climb skill configuration."""

from skillsblender.tasks.path.config.climb_env_cfg import ClimbPathEnvCfg
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp


_CLIMB_INIT_H = 0.56
_CLIMB_FINAL_H = 0.78
_CLIMB_INIT_LEN = 1.30
_CLIMB_FINAL_LEN = 0.90


@configclass
class ClimbCurriculumCfg:
    """Climb-specific curriculum learning configuration."""

    climb_difficulty = CurrTerm(
        func=mdp.curriculum_climb_difficulty,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 60.0,
            "initial_climb_height": _CLIMB_INIT_H,
            "final_climb_height": _CLIMB_FINAL_H,
            "climb_height_step": 0.02,
            "initial_climb_len": _CLIMB_INIT_LEN,
            "final_climb_len": _CLIMB_FINAL_LEN,
            "climb_len_step": 0.03,
            "step_asset_name": "climb_step",
            "top_asset_name": "climb_top",
            "step_half_height": 0.28,
            "top_half_thickness": 0.05,
        },
    )


@configclass
class Go2ClimbEnvCfg(ClimbPathEnvCfg):
    """Unitree Go2 climb skill training configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Set Go2 robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # General
        self.rewards.is_terminated.weight = -250.0
        self.rewards.joint_deviation.weight = -0.2

        # Obs scaling
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Task rewards
        self.rewards.track_xy.weight = 6.0
        self.rewards.track_yaw.weight = 2.5
        self.rewards.track_velocity_along_path_exp.weight = 1.0
        self.rewards.track_velocity_along_path_exp.params["desired_speed"] = 0.65
        self.rewards.climb_track_velocity_phase.weight = 5.0
        self.rewards.climb_track_velocity_phase.params["desired_speed"] = 0.65

        # Base stability on slopes
        self.rewards.flat_orientation.weight = -1.5
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

        # Prevent slipping
        self.rewards.feet_slide.weight = -3.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*_foot"]

        # Contact penalties
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_hip", ".*_thigh", ".*_calf"
        ]

        # Command parameters
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk", "climb", "walk"]

        if self.__class__.__name__ == "Go2ClimbEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2ClimbCurEnvCfg(Go2ClimbEnvCfg):
    """Unitree Go2 climb training configuration with curriculum."""

    curriculum: ClimbCurriculumCfg = ClimbCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.path_tracking.climb_params.climb_height = _CLIMB_INIT_H
        self.commands.path_tracking.climb_params.climb_len = _CLIMB_INIT_LEN

        if hasattr(self.scene, "climb_step"):
            self.scene.climb_step.init_state.pos = (1.8, 0.0, _CLIMB_INIT_H - 0.28)
        if hasattr(self.scene, "climb_top"):
            self.scene.climb_top.init_state.pos = (2.9, 0.0, _CLIMB_INIT_H + 0.05)

        if self.__class__.__name__ == "Go2ClimbCurEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2ClimbEnvCfg_PLAY(Go2ClimbEnvCfg):
    """Unitree Go2 climb PLAY configuration."""

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
class Go2ClimbCurEnvCfg_PLAY(Go2ClimbCurEnvCfg):
    """Unitree Go2 climb curriculum PLAY configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None
