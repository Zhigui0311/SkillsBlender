"""Go2 virtual stairs wrappers for quick parameter edits."""

from isaaclab.utils import configclass

from skillsblender.tasks.path.config.virtual_stairs_env_cfg import (
    Go2VirtualStairsEnvCfg,
    Go2VirtualStairsDownEnvCfg,
    Go2VirtualStairsEnvCfg_PLAY,
)


@configclass
class Go2VStairsEnvCfg(Go2VirtualStairsEnvCfg):
    """Go2 virtual stairs training configuration."""

    pass


@configclass
class Go2VStairsDownEnvCfg(Go2VirtualStairsDownEnvCfg):
    """Go2 virtual stairs-down training configuration."""

    pass


@configclass
class Go2VStairsEnvCfg_PLAY(Go2VirtualStairsEnvCfg_PLAY):
    """Go2 virtual stairs PLAY configuration."""

    pass
