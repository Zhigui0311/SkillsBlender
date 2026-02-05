"""Legacy flat task aliases.

This module keeps old `go2-path-flat-*` task entry-points working while using
`go2_walk_cfg.py` as the single walk-skill implementation.
"""

from skillsblender.tasks.path.robots.go2.path.go2_walk_cfg import (
    Go2WalkEnvCfg,
    Go2WalkEnvCfg_PLAY,
)


# Legacy class names kept for backward compatibility with existing task ids.
Go2PathEnvCfg = Go2WalkEnvCfg
Go2PathEnvCfg_PLAY = Go2WalkEnvCfg_PLAY
Go2PathFlatCfg = Go2WalkEnvCfg
Go2PathFlatEnvCfg_PLAY = Go2WalkEnvCfg_PLAY
