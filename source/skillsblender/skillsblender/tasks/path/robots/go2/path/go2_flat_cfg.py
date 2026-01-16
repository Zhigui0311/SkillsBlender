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

        self.actions.joint_pos_actoion.scale = 0.2
        self.rewards.torques.weight = -0.0005
        self.rewards.action_rate.weight = -0.05
        self.rewards.joint_dev.weight = -0.2
        # joint的关节尺度 关节限位
        
        #版本1
        self.commands.path_tracking.inpoints.end_to_start_pos = (4.0, 5.0, 0.0)
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
    
@configclass
class Go2PathFlatCfg(Go2PathEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards: RewardsCfg = Go2VelRewardCfg()
        self.rewards.track_xy.weight = 8.0
        self.rewards.track_yaw.weight = 0.8
        self.rewards.track_velocity_along_path_exp.weight = 2.0
        self.rewards.action_rate.weight = -0.005
        self.commands.path_tracking.resampling_time_range = (8.0, 12.0)


        
        
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