"""Unitree Go2 crouch skill configuration."""

from skillsblender.tasks.path.config.crouch_env_cfg import CrouchPathEnvCfg
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp

_CROUCH_ROOF_INIT_H = 0.42
_CROUCH_ROOF_FINAL_H = 0.36
_CROUCH_ROOF_THICKNESS = 0.10
_CROUCH_CLEARANCE = 0.04

def _crouch_target_height(roof_h: float) -> float:
    return roof_h - _CROUCH_ROOF_THICKNESS * 0.5 - _CROUCH_CLEARANCE


@configclass
class CrouchCurriculumCfg:
    """Crouch-specific curriculum learning configuration."""

    roof_height = CurrTerm(
        func=mdp.curriculum_crouch_roof_height,
        params={
            "asset_name": "crouch_roof",
            "initial_height": _CROUCH_ROOF_INIT_H,
            "final_height": _CROUCH_ROOF_FINAL_H,
            "step_size": 0.01,
            "reward_threshold": 60.0,
            "roof_thickness": _CROUCH_ROOF_THICKNESS,
            "clearance": _CROUCH_CLEARANCE,
        },
    )


@configclass
class Go2CrouchEnvCfg(CrouchPathEnvCfg):
    """Unitree Go2 crouch skill training configuration."""

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
        self.rewards.track_velocity_along_path_exp.weight = 3.0

        # Low height reward (encourage crouching)
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.24
        self.rewards.crouch_base_height_phase.weight = -3.5
        self.rewards.crouch_base_height_phase.params["target_height"] = 0.24

        # Stability in low stance
        self.rewards.flat_orientation.weight = -2.0
        self.rewards.base_lin_vel_z.weight = -0.5
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.base_acc.weight = -2.5e-4

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-4
        self.rewards.joint_vel_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0

        # Avoid hitting obstacles with body
        self.rewards.undesired_contacts.weight = -8.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", "Head_upper", "Head_lower", ".*_hip", ".*_thigh"
        ]

        # Command parameters
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24

        if self.__class__.__name__ == "Go2CrouchEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2CrouchCurEnvCfg(Go2CrouchEnvCfg):
    """Unitree Go2 crouch training configuration with curriculum."""

    curriculum: CrouchCurriculumCfg = CrouchCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Set easier initial roof height for curriculum start
        if hasattr(self.scene, "crouch_roof"):
            self.scene.crouch_roof.init_state.pos = (1.15, 0.0, _CROUCH_ROOF_INIT_H)
        # Tail roof removed - using single 0.4m obstacle
        # if hasattr(self.scene, "crouch_roof_tail"):
        #     self.scene.crouch_roof_tail.init_state.pos = (3.0, 0.0, _CROUCH_ROOF_INIT_H)

        target_h = _crouch_target_height(_CROUCH_ROOF_INIT_H)
        self.commands.path_tracking.crouch_params.base_height_ref = target_h
        if self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["target_height"] = target_h
        if self.rewards.crouch_base_height_phase is not None:
            self.rewards.crouch_base_height_phase.params["target_height"] = target_h


@configclass
class Go2CrouchEnvCfg_PLAY(Go2CrouchEnvCfg):
    """Unitree Go2 crouch PLAY configuration."""

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
        # Keep camera locked to the robot while showing the obstacle corridor.
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "base"
        self.viewer.eye = (-2.0, 1.0, 1.1)
        self.viewer.lookat = (1.5, 0.0, 0.25)
        self.commands.path_tracking.debug_vis = True
        self.observations.policy.enable_corruption = False


@configclass
class Go2CrouchCurEnvCfg_PLAY(Go2CrouchCurEnvCfg):
    """Crouch curriculum PLAY configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None
        # Match crouch play camera behavior.
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "base"
        self.viewer.eye = (-2.0, 1.0, 1.1)
        self.viewer.lookat = (1.5, 0.0, 0.25)
