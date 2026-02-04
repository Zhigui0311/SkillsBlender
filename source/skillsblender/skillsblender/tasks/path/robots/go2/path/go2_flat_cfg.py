from skillsblender.tasks.path.config.path_env_cfg import PathEnvCfg #, PathPlayEnvCfg
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Go2PathEnvCfg(PathEnvCfg):
    "Unitree Go2 in flat terrain path following task configuration."

    def __post_init__(self):

        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.actions.joint_pos_action.scale = 0.2
  
        # joint的关节尺度 关节限位
        
        #版本1
        self.commands.path_tracking.inpoints.end_to_start_pos = (4.0, 5.0, 0.0)
        self.commands.path_tracking.ranges.default_path_len = 5.0
        self.commands.path_tracking.ranges.num_waypoints = 100
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 5
        self.commands.path_tracking.inpoints.yaw_type = 'decoupled'


@configclass
class Go2PathEnvCfg_PLAY(Go2PathEnvCfg):
    "Unitree Go2 in flat terrain path following play task configuration."
    
    def __post_init__(self):
        super().__post_init__()
        
        self.sim.dt = 0.005 # 200Hz Simulation frequency
        self.decimation = 4 # 50Hz control frequency
        self.episode_length_s = 10.0 
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        
        self.sim.render_interval = 2  
        
        self.scene.terrain.max_init_terrain_level = None
        self.events.base_external_force_torque = None
        self.curriculum = None

        self.sim.physics_material = self.scene.terrain.physics_material
        # self.viewer.asset_name = ""
        self.viewer.origin_type = "env" 
        # self.viewer.origin_type = "world"
        # self.viewer.eye = (10.0, 10.0, 80.0)
        # self.viewer.lookat = (0.0, 0.0, 0.0)
        self.commands.path_tracking.debug_vis = True
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.observations.policy.enable_corruption = False

from isaaclab.managers import RewardTermCfg as RewTerm
from skillsblender.tasks.path.config.path_env_cfg import RewardsCfg


@configclass
class Go2VelRewardCfg(RewardsCfg):
    track_velocity_along_path_exp = RewTerm(
        func=mdp.track_velocity_along_path_exp,
        weight=1.0,  #根据训练效果调整，先给 1.0 - 2.0
        params={"std": 0.5, "command_name": "path_tracking"}
    )

#带速度的
@configclass
class Go2PathFlatCfg(Go2PathEnvCfg):
    foot_link_name = ".*_foot"
    def __post_init__(self):
        super().__post_init__()
        self.rewards: RewardsCfg = Go2VelRewardCfg()
        self.rewards.track_xy.weight = 8.0
        self.rewards.track_yaw.weight = 0.8
        self.rewards.track_velocity_along_path_exp.weight = 4.0
        # self.rewards.hip_angle_penalty
        
        self.commands.path_tracking.resampling_time_range = (8.0, 12.0)

 # ------------------------------Rewards------------------------------
 # 哪个奖励的维度不正确？
 
        # General
        self.rewards.is_terminated.weight = -400.0
        self.rewards.joint_deviation.weight = -0.25
        
        # Base
        # self.rewards.base_height.weight = -10.0
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

        # Contact sensor
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", ".*_thigh", ".*_calf"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0

        # Others
        self.rewards.air_time_variance.weight = -4.0
        self.rewards.feet_acc.weight = -2e-6
        self.rewards.feet_acc.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 2.0 # Increased to strongly encourage trotting and prevent tripod gait
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.feet_height.weight = -5.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = -0.22
        self.rewards.feet_height.params["dis_threshold"] = 0.25
        self.rewards.feet_height.params["heading_threshold"] = 0.5
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["dis_threshold"] = 0.25
        self.rewards.feet_air_time.params["heading_threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -2.0
        self.rewards.stalling_penalty.weight = -1.0

        if self.__class__.__name__ == "Go2PathFlatCfg":
            self.disable_zero_weight_rewards()


        
        
@configclass
class Go2PathFlatEnvCfg_PLAY(Go2PathFlatCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005 # 200Hz Simulation frequency
        self.decimation = 4 # 50Hz control frequency
        self.episode_length_s = 10.0 
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        
        self.sim.render_interval = 2  
        
        self.scene.terrain.max_init_terrain_level = None
        self.events.base_external_force_torque = None
        self.curriculum = None

        self.sim.physics_material = self.scene.terrain.physics_material
        # self.viewer.asset_name = ""
        self.viewer.origin_type = "env" 
        # self.viewer.origin_type = "world"
        # self.viewer.eye = (10.0, 10.0, 80.0)
        # self.viewer.lookat = (0.0, 0.0, 0.0)
        self.commands.path_tracking.debug_vis = True
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.observations.policy.enable_corruption = False
