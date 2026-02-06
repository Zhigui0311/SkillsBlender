"""Go2 virtual jump wrappers for quick parameter edits."""

from isaaclab.utils import configclass

from skillsblender.tasks.path.config.virtual_jump_env_cfg import (
    Go2VirtualJumpEnvCfg,
    Go2VirtualJumpEnvCfg_PLAY,
)


@configclass
class Go2VJumpEnvCfg(Go2VirtualJumpEnvCfg):
    """Go2 virtual jump training configuration."""

    pass


@configclass
class Go2VJumpEnvCfg_PLAY(Go2VirtualJumpEnvCfg_PLAY):
    """Go2 virtual jump PLAY configuration."""

    pass
