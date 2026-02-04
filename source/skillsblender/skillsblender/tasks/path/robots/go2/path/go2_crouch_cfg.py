"""Unitree Go2 crouch skill configuration."""

from skillsblender.tasks.path.config.crouch_env_cfg import CrouchPathEnvCfg
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp


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
        self.rewards.base_height_l2.weight = -3.0
        self.rewards.base_height_l2.params["target_height"] = 0.24

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
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", ".*_hip", ".*_thigh"
        ]

        # Command parameters
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["crouch"]

        if self.__class__.__name__ == "Go2CrouchEnvCfg":
            self.disable_zero_weight_rewards()


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
        self.viewer.origin_type = "env"
        self.commands.path_tracking.debug_vis = True
        self.observations.policy.enable_corruption = False
