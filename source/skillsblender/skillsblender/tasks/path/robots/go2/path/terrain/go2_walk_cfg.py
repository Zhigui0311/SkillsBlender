"""Unitree Go2 walk skill configuration."""

from skillsblender.tasks.path.config.walk_env_cfg import WalkPathEnvCfg
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp


@configclass
class Go2WalkEnvCfg(WalkPathEnvCfg):
    """Unitree Go2 walk skill training configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Set Go2 robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # General penalties
        self.rewards.is_terminated.weight = -200.0
        self.rewards.joint_deviation.weight = -0.2

        # Observation scaling
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Task rewards
        self.rewards.track_xy.weight = 6.0
        self.rewards.track_yaw.weight = 3.0
        self.rewards.track_velocity_along_path_exp.weight = 4.0

        # Base stability
        self.rewards.flat_orientation.weight = -0.5
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_lin_vel_z.weight = -0.5
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.base_acc.weight = -2.5e-4

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1e-4
        self.rewards.joint_vel_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0

        # Joint symmetry
        self.rewards.joint_mirror.weight = -0.7
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.applied_torque_limits.weight = -0.1
        self.rewards.action_rate_l2.weight = -0.01

        # Feet rewards: encourage trotting gait
        self.rewards.feet_air_time.weight = 0.8
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.air_time_variance.weight = -1.2
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_contact_balance.weight = -1.5

        # Gait reward
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot")
        )

        # Contact penalties
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_hip", ".*_thigh", ".*_calf"
        ]
        self.rewards.undesired_contacts.params["threshold"] = 1.0

        # Command parameters
        self.commands.path_tracking.ranges.num_waypoints = 80
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 24
        self.commands.path_tracking.path_generator_cfg.skill_sequence = ["walk"]
        self.commands.path_tracking.sampling.yaw_type = "decoupled"
        self.commands.path_tracking.sampling.start_heading = (-3.1415926, 3.1415926)
        self.commands.path_tracking.sampling.end_to_start_pos = (3.0, 6.0, 0.0)
        self.commands.path_tracking.sampling.sample_goal_distance = True

        self.events.reset_base.params["pose_range"]["x"] = (-1.2, 1.2)
        self.events.reset_base.params["pose_range"]["y"] = (-1.2, 1.2)

        # Disable zero-weight rewards
        if self.__class__.__name__ == "Go2WalkEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2WalkEnvCfg_PLAY(Go2WalkEnvCfg):
    """Unitree Go2 walk PLAY configuration for visualization."""

    def __post_init__(self):
        super().__post_init__()

        # Simulation parameters
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0

        # Scene configuration
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0
        self.sim.render_interval = 2

        # Disable terrain difficulty progression
        self.scene.terrain.max_init_terrain_level = None

        # Disable disturbances
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Disable curriculum
        self.curriculum = None

        # Physics material
        self.sim.physics_material = self.scene.terrain.physics_material

        # Viewer settings
        self.viewer.origin_type = "env"

        # Enable path visualization
        self.commands.path_tracking.debug_vis = True

        # Disable observation noise
        self.observations.policy.enable_corruption = False
