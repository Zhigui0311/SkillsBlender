from skillsblender.tasks.path.config.jump_env_cfg import JumpPathEnvCfg 
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG





@configclass
class Go2JumpEnvCfg(JumpPathEnvCfg):
    "Unitree Go2 in gap terrain path following task configuration."

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
  
    
        self.commands.path_tracking.ranges.num_waypoints = 100
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 5
