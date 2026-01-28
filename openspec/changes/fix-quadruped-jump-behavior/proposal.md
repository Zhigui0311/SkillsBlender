## Why

The quadruped robot's jumping behavior is severely degraded - instead of jumping over gaps, the robot spins and lifts legs at the starting position without any takeoff tendency. The robot fails to reach the target on the opposite side of the gap, and the current endpoint positioning is too close to the gap edge, causing the robot to frequently fall into the gap. This prevents effective jump training and deployment.

## What Changes

- **Adjust endpoint positioning logic** to place the target further from the gap edge, providing a safer landing zone
- **Modify reward weights** to encourage forward commitment and penalize spinning/stalling behavior at the start position
- **Tune jump trajectory parameters** to ensure proper takeoff timing and arc generation
- **Adjust curriculum progression** to ensure the robot learns basic forward jumping before attempting complex gaps
- **Review and fix waypoint update logic** to prevent premature waypoint advancement that might confuse the policy

## Capabilities

### New Capabilities
- `jump-behavior-diagnostics`: Add diagnostic logging and visualization to identify why the robot is not initiating jumps properly

### Modified Capabilities
- `jump-trajectory-planning`: Modify the trajectory generation to ensure proper takeoff timing and endpoint positioning
- `jump-reward-shaping`: Adjust reward weights to strongly discourage spinning/stalling and encourage forward momentum
- `jump-curriculum-tuning`: Revise curriculum parameters to ensure progressive learning from simple to complex jumps

## Impact

**Affected Code:**
- `source/skillsblender/skillsblender/tasks/path/mdp/commands/jump_path_command.py` - Endpoint positioning and trajectory generation
- `source/skillsblender/skillsblender/tasks/path/mdp/rewards/jump_rewards.py` - Reward weight adjustments
- `source/skillsblender/skillsblender/tasks/path/mdp/curriculum.py` - Curriculum progression parameters
- `source/skillsblender/skillsblender/tasks/path/config/go2_jump_cur_cfg.py` - Configuration parameters

**Training Impact:**
- May require retraining from scratch or from an earlier checkpoint
- Curriculum progression thresholds may need adjustment
- Reward normalization may be affected by weight changes

**Deployment Impact:**
- Improved jump success rate and safety
- More reliable gap crossing behavior
- Reduced risk of falling into gaps
