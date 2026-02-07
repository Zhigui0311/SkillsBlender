"""Unitree Go2 climb (slope) skill configuration."""

from isaaclab.utils import configclass

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG
from skillsblender.tasks.path.config.climb_env_cfg import ClimbPathEnvCfg, ClimbPathEnvCfg_PLAY


@configclass
class Go2ClimbEnvCfg(ClimbPathEnvCfg):
    """Unitree Go2 climb skill training configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if self.__class__.__name__ == "Go2ClimbEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Go2ClimbEnvCfg_PLAY(ClimbPathEnvCfg_PLAY):
    """Unitree Go2 climb PLAY configuration."""

    def __post_init__(self):
        super().__post_init__()
