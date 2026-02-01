from skillsblender.tasks.path.config.jump_env_cfg import JumpPathEnvCfg
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Go2JumpEnvCfg(JumpPathEnvCfg):
    """Unitree Go2 in gap terrain path following task configuration."""

    def __post_init__(self):

        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # General
        self.rewards.is_terminated.weight = -400.0
        self.rewards.joint_deviation.weight = -0.25
        
        # obs
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        #task
        self.rewards.jump_height_tracking.weight = 4.0  # 3.0 → 4.0
        self.rewards.jump_landing_stability.weight = 2.5  # 2.0 → 2.5
        self.rewards.jump_forward_velocity.weight = 1.5  # 2.0 → 1.5
        
        # emphasize tracking the planned trajectory
        self.rewards.track_xy.weight = 8.0
        self.rewards.track_yaw.weight = 4.0
        self.rewards.track_velocity.weight = 5.0
        
        # # Base 
        # self.rewards.base_height.weight = -10.0  #！1 这个奖励还没有写
        self.rewards.base_height_l2.weight = 0
        self.rewards.flat_orientation.weight = -0.5
        self.rewards.base_lin_vel_z.weight = -0.7
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.base_acc.weight = -5e-4

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2e-4
        self.rewards.joint_vel_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0
        self.rewards.joint_mirror.weight = -0.5
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.applied_torque_limits.weight = -0.2
        self.rewards.action_rate_l2.weight = -2e-5
        # Feet rewards: encourage all feet to lift during jump
        self.rewards.air_time_variance.weight = -2.0
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["dis_threshold"] = 0.3
        
        # Contact sensor
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", ".*_thigh", ".*_calf"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        
        # 命令参数调整
        self.commands.path_tracking.ranges.num_waypoints = 100
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 5
        self.commands.path_tracking.jump_params.jump_height = 0.45
        # Others
        # self.rewards.air_time_variance.weight = -4.0
        # self.rewards.feet_acc.weight = -2e-6
        # self.rewards.feet_acc.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_slide.weight = -2.0
        # self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_gait.weight = 2.0 # Increased to strongly encourage trotting and prevent tripod gait
        # self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        # self.rewards.feet_height.weight = -5.0
        # self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_height.params["target_height"] = -0.22
        # self.rewards.feet_height.params["dis_threshold"] = 0.25
        # self.rewards.feet_height.params["heading_threshold"] = 0.5
        # self.rewards.feet_air_time.weight = 1.0
        # self.rewards.feet_air_time.params["threshold"] = 0.5
        # self.rewards.feet_air_time.params["dis_threshold"] = 0.25
        # self.rewards.feet_air_time.params["heading_threshold"] = 0.5
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_stumble.weight = -2.0
        self.rewards.stalling_penalty.weight = -1.0

        
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2JumpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2JumpEnvCfg_PLAY(Go2JumpEnvCfg):
    """Unitree Go2 in gap terrain path following PLAY configuration for visualization."""

    def __post_init__(self):
        super().__post_init__()

        # 仿真参数（这些会在父类__post_init__中设置，但这里再设置一次确保正确）
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz control
        self.episode_length_s = 15.0  # 跳跃任务需要更长时间

        # 场景配置（少环境，便于观察）
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0  

        self.sim.render_interval = 2

        # 禁用地形难度递增（方便测试特定难度）
        self.scene.terrain.max_init_terrain_level = None

        # 禁用干扰事件
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 禁用课程学习
        self.curriculum = None

        # 物理材质
        self.sim.physics_material = self.scene.terrain.physics_material

        # 视角设置
        self.viewer.origin_type = "env"

        # 启用路径可视化（观察跳跃轨迹）
        self.commands.path_tracking.debug_vis = True

        # 禁用观测噪声（便于调试）
        self.observations.policy.enable_corruption = False
