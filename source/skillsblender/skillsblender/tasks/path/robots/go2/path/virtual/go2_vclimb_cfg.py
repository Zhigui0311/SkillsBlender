"""Go2 virtual climb wrappers for quick parameter edits."""

from isaaclab.utils import configclass

from skillsblender.tasks.path.config.virtual_climb_env_cfg import (
    Go2VirtualClimbEnvCfg,
    Go2VirtualClimbEnvCfg_PLAY,
)


@configclass
class Go2VClimbEnvCfg(Go2VirtualClimbEnvCfg):
    """Go2 virtual climb training configuration."""

    pass


@configclass
class Go2VClimbEnvCfg_PLAY(Go2VirtualClimbEnvCfg_PLAY):
    """Go2 virtual climb PLAY configuration."""

    pass
