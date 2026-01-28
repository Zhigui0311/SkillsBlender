## Why

Currently, jump trajectory parameters (gap size ranges, endpoint distance extensions, takeoff/landing margins) are hardcoded in the implementation, making it difficult to experiment with different jump configurations or adapt to different robot capabilities without modifying source code. Users need the flexibility to configure these parameters in cfg files to quickly iterate on jump behavior and support different training scenarios (e.g., conservative vs aggressive jumping, different gap sizes for different robot models).

## What Changes

- **Add configurable gap size ranges** to terrain configuration for easy adjustment of narrow/wide gap definitions
- **Add configurable endpoint distance parameters** to control how far beyond the gap the robot should land (separate settings for narrow vs wide gaps)
- **Add configurable jump margins** (takeoff and landing) to control when the parabolic arc starts and ends relative to gap boundaries
- **Add optional post-jump forward distance** to configure whether the robot should continue walking after landing or stop immediately
- **Expose all jump trajectory parameters** in JumpPathCommandCfg for centralized configuration
- **Update default configurations** to use the new configurable parameters instead of hardcoded values

## Capabilities

### New Capabilities
- `configurable-jump-trajectory`: Expose jump trajectory parameters (endpoint extensions, margins, post-jump distance) in configuration files
- `configurable-gap-definitions`: Allow gap size ranges (narrow/wide thresholds) to be defined in terrain configuration

### Modified Capabilities
None - this is purely adding configuration flexibility without changing existing behavior requirements.

## Impact

**Affected Code:**
- `source/skillsblender/skillsblender/tasks/path/mdp/commands/jump_path_command.py` - Replace hardcoded values with cfg parameters
- `source/skillsblender/skillsblender/tasks/path/mdp/commands/path_command_cfg.py` - Add new configuration parameters to JumpPathCommandCfg
- `source/skillsblender/skillsblender/tasks/path/config/jump_env_cfg.py` - Update default configuration values
- `source/skillsblender/skillsblender/tasks/path/robots/go2/path/go2_jump_cur_cfg.py` - Set robot-specific jump parameters

**Configuration Impact:**
- Existing configurations will continue to work with default values
- Users can now override jump parameters in their custom cfg files
- Training experiments can easily test different jump parameter combinations

**Training Impact:**
- No impact on existing trained policies (behavior unchanged with default parameters)
- Enables faster iteration on jump behavior tuning
- Supports creating specialized configurations for different scenarios (e.g., cautious jumping, aggressive jumping)

**Benefits:**
- Eliminates need to modify source code for parameter tuning
- Enables A/B testing of different jump configurations
- Supports robot-specific jump parameter profiles
- Facilitates research on optimal jump parameters
