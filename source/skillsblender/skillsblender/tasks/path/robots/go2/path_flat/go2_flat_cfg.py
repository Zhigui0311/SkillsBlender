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

        # joint的关节尺度 关节限位



@configclass
class Go2PathPlayEnvCfg(Go2PathEnvCfg):
    "Unitree Go2 in flat terrain path following play task configuration."
    
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 10
        self.scene.env_spacing = 10.0

        self.scene.terrain.max_init_terrain_level = None

