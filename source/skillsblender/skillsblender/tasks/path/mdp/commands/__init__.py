from .path_command_cfg import (
    PathCommandCfg,
    JumpPathCommandCfg,
    StairsPathCommandCfg,
    ClimbPathCommandCfg,
    CrouchPathCommandCfg,
)
from .flat_path_command import FlatPathCommand
from .jump_path_command import JumpPathCommand
from .stairs_path_command import StairsPathCommand
from .climb_path_command import ClimbPathCommand
from .crouch_path_command import CrouchPathCommand
from .planner_path_command import PlannerPathCommand
from .virtual_jump_path_command import VirtualJumpPathCommand, VirtualJumpPathCommandCfg
from .virtual_crouch_path_command import VirtualCrouchPathCommand, VirtualCrouchPathCommandCfg
from .virtual_climb_path_command import VirtualClimbPathCommand, VirtualClimbPathCommandCfg
from .virtual_stairs_path_command import VirtualStairsPathCommand, VirtualStairsPathCommandCfg

# Backward-compatible alias
PathCommand = FlatPathCommand
