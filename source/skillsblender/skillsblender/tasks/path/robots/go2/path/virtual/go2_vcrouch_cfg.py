"""Go2 virtual crouch wrappers for quick parameter edits."""

from isaaclab.utils import configclass

from skillsblender.tasks.path.config.virtual_crouch_env_cfg import (
    Go2VirtualCrouchEnvCfg,
    Go2VirtualCrouchEnvCfg_PLAY,
)


@configclass
class Go2VCrouchEnvCfg(Go2VirtualCrouchEnvCfg):
    """Go2 virtual crouch training configuration."""

    pass


@configclass
class Go2VCrouchEnvCfg_PLAY(Go2VirtualCrouchEnvCfg_PLAY):
    """Go2 virtual crouch PLAY configuration."""

    pass
