from .path_command_cfg import (
    PathCommandCfg,
    JumpPathCommandCfg,
    StairsPathCommandCfg,
    PlatformPathCommandCfg,
    ClimbPathCommandCfg,
    CrouchPathCommandCfg,
)
from .flat_path_command import FlatPathCommand
from .jump_path_command import JumpPathCommand
from .stairs_path_command import StairsPathCommand
from .platform_path_command import PlatformPathCommand
from .climb_path_command import ClimbPathCommand
from .crouch_path_command import CrouchPathCommand
from .planner_path_command import PlannerPathCommand
from .virtual_jump_path_command import VirtualJumpPathCommand, VirtualJumpPathCommandCfg
from .virtual_crouch_path_command import VirtualCrouchPathCommand, VirtualCrouchPathCommandCfg

# Backward-compatible alias
PathCommand = FlatPathCommand
