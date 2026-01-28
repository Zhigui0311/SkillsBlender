## Context

Currently, jump trajectory parameters are hardcoded in `jump_path_command.py`:
- Endpoint extensions: `gap_end + 1.0` (recently changed to 1.5m/2.0m based on gap width)
- Takeoff margins: 0.4m for narrow, 0.5m for wide (recently updated)
- Landing margin: 0.5m (recently updated)
- Gap width threshold: 0.55m (hardcoded in recent changes)

These hardcoded values make it difficult to:
- Experiment with different jump strategies without code changes
- Create robot-specific jump profiles (e.g., conservative vs aggressive)
- Adapt to different gap size distributions in terrain
- A/B test jump parameter combinations

**Constraints:**
- Must maintain backward compatibility with existing configurations
- Should not require changes to existing trained policies
- Must integrate cleanly with Isaac Lab's configuration system
- Should follow existing configuration patterns in the codebase

**Stakeholders:**
- Researchers experimenting with jump parameters
- Users deploying on different robot platforms
- Training pipeline (should not break existing workflows)

## Goals / Non-Goals

**Goals:**
- Expose all jump trajectory parameters in configuration files
- Allow per-robot and per-scenario parameter customization
- Maintain backward compatibility with existing configurations
- Provide sensible defaults that match current behavior
- Enable easy parameter tuning without code modification

**Non-Goals:**
- Changing the fundamental jump trajectory algorithm
- Adding new trajectory planning features (e.g., velocity-dependent margins)
- Modifying terrain generation beyond gap width ranges
- Implementing parameter validation beyond basic sanity checks
- Creating a GUI or interactive parameter tuning tool

## Decisions

### Decision 1: Add parameters to JumpPathCommandCfg.JumpParams
**Choice:** Extend the existing `JumpParams` dataclass with new configuration fields.

**Rationale:** The `JumpParams` class already exists and is the natural place for jump-related parameters. This keeps all jump configuration in one place and follows the existing pattern.

**Alternatives considered:**
- Create a separate `JumpTrajectoryConfig` class: More modular but adds complexity and breaks existing patterns
- Add parameters directly to `JumpPathCommandCfg`: Too flat, mixes trajectory params with other command params
- Use environment variables: Not type-safe, harder to document, doesn't integrate with cfg system

**Implementation:**
```python
@configclass
class JumpParams:
    # Existing parameters
    jump_height: float = 0.35
    gap_threshold: float = -0.4
    scan_dist: float = 6.0
    scan_step: float = 0.1

    # New parameters
    gap_width_threshold: float = 0.55  # Threshold between narrow/wide gaps
    narrow_gap_endpoint_extension: float = 1.5  # Distance beyond gap end for narrow gaps
    wide_gap_endpoint_extension: float = 2.0  # Distance beyond gap end for wide gaps
    narrow_gap_takeoff_margin: float = 0.4  # Distance before gap to start arc (narrow)
    wide_gap_takeoff_margin: float = 0.5  # Distance before gap to start arc (wide)
    landing_margin: float = 0.5  # Distance after gap to end arc
    post_jump_distance: float = 0.0  # Optional additional distance after landing
```

### Decision 2: Keep gap width ranges in terrain configuration
**Choice:** Gap width ranges remain in `TerrainGeneratorCfg.sub_terrains` configuration.

**Rationale:** Gap widths are a property of terrain generation, not jump trajectory planning. Keeping them in terrain config maintains separation of concerns.

**Alternatives considered:**
- Move gap ranges to JumpParams: Violates separation of concerns, terrain generation shouldn't depend on jump command config
- Duplicate in both places: Leads to inconsistency and confusion
- Create a shared config: Over-engineering for this use case

**Implementation:** No changes needed - gap ranges are already configurable in terrain config. Jump command will read them if needed for adaptive behavior.

### Decision 3: Use gap width to determine narrow vs wide classification
**Choice:** Compare detected gap width against `gap_width_threshold` to classify gaps.

**Rationale:** Simple, deterministic, and allows users to adjust the boundary between narrow and wide gaps based on their terrain configuration.

**Alternatives considered:**
- Use terrain config gap ranges: Requires accessing terrain config from jump command, adds coupling
- Fixed classification (e.g., < 0.5m = narrow): Not flexible, doesn't adapt to user's terrain setup
- Multiple thresholds: Over-complicated for current needs

**Implementation:**
```python
gap_width = gap_end_dist - gap_start_dist
is_narrow_gap = gap_width < self.cfg.jump_params.gap_width_threshold
```

### Decision 4: Make post-jump distance optional (default 0)
**Choice:** Add `post_jump_distance` parameter with default value of 0 (no extension).

**Rationale:** Most users want the robot to land and stabilize immediately. Advanced users can enable post-jump forward movement if needed for their scenario.

**Alternatives considered:**
- Always extend trajectory: Wastes time in most scenarios, may cause instability
- Make it a boolean flag: Less flexible, can't control how far to extend
- Tie to gap width: Too implicit, users should explicitly control this behavior

**Implementation:**
```python
if self.cfg.jump_params.post_jump_distance > 0:
    total_len += self.cfg.jump_params.post_jump_distance
```

### Decision 5: Provide defaults that match current behavior
**Choice:** Set default parameter values to match the recently updated hardcoded values.

**Rationale:** Ensures backward compatibility and prevents breaking existing experiments or trained policies.

**Default values:**
- `gap_width_threshold`: 0.55m (current hardcoded value)
- `narrow_gap_endpoint_extension`: 1.5m (current behavior)
- `wide_gap_endpoint_extension`: 2.0m (current behavior)
- `narrow_gap_takeoff_margin`: 0.4m (current behavior)
- `wide_gap_takeoff_margin`: 0.5m (current behavior)
- `landing_margin`: 0.5m (current behavior)
- `post_jump_distance`: 0.0m (no extension, current behavior)

### Decision 6: No runtime parameter validation beyond type checking
**Choice:** Rely on Python type hints and basic sanity checks (e.g., positive values) rather than complex validation.

**Rationale:** Users are researchers who understand the parameters. Over-validation adds complexity and may prevent legitimate experimental configurations.

**Basic checks to include:**
- Margins and extensions should be >= 0
- Gap width threshold should be > 0
- No checks on "reasonable" ranges (e.g., don't prevent 10m endpoint extension if user wants to test it)

## Risks / Trade-offs

### Risk 1: Users may set incompatible parameter combinations
**Example:** Very large takeoff margin + small gap width could cause arc to start before previous gap ends.

**Mitigation:** Document parameter interactions in configuration comments. Add warnings (not errors) for suspicious combinations. Trust users to understand their experimental setup.

### Risk 2: Default values may not be optimal for all robots
**Example:** Smaller robots might need smaller margins, larger robots might need larger extensions.

**Mitigation:** This is intentional - users should tune parameters for their robot. Provide robot-specific configuration examples (e.g., `go2_jump_cfg.py` with GO2-tuned values).

### Risk 3: Too many parameters may overwhelm users
**Trade-off:** Flexibility vs simplicity. More parameters = more tuning burden.

**Mitigation:** Provide good defaults that work for most cases. Document which parameters have the biggest impact. Consider creating preset profiles (conservative, balanced, aggressive) in future if needed.

### Risk 4: Changing defaults in future may break reproducibility
**Example:** If we later decide 1.5m endpoint extension is too short and change default to 2.0m, old experiments won't reproduce exactly.

**Mitigation:** Document default values in release notes. Encourage users to explicitly set parameters in their config files rather than relying on defaults for published experiments.

### Risk 5: Gap width threshold may not align with terrain config
**Example:** User sets narrow gaps to 0.3-0.6m and wide gaps to 0.7-1.0m, but threshold is 0.55m. Some "narrow" terrain gaps will be classified as "wide" by jump command.

**Mitigation:** Document that threshold should be set based on desired behavior, not necessarily terrain config boundaries. Add example configurations showing how to align them.

## Migration Plan

**Phase 1: Add configuration parameters (no behavior change)**
1. Add new fields to `JumpParams` dataclass with defaults matching current hardcoded values
2. Update `jump_path_command.py` to read from `self.cfg.jump_params` instead of hardcoded values
3. Verify that default configuration produces identical behavior to current implementation

**Phase 2: Update default configurations**
1. Add comments to `jump_env_cfg.py` documenting the new parameters
2. Update `go2_jump_cur_cfg.py` to explicitly set parameters (even if same as defaults) for clarity
3. Create example configurations showing different parameter combinations

**Phase 3: Documentation**
1. Document parameter meanings and interactions
2. Provide guidelines for tuning parameters
3. Add examples of conservative vs aggressive jump profiles

**Phase 4: Testing**
1. Test with default parameters (should match current behavior)
2. Test with extreme parameter values (very small/large margins)
3. Test with post-jump distance enabled
4. Verify backward compatibility with existing config files

**Rollback Strategy:**
- If issues are found, parameters can be reverted to hardcoded values by setting defaults
- No trained policies are affected (behavior unchanged with defaults)
- No breaking changes to configuration API

## Open Questions

1. **Should we add parameter validation warnings?** E.g., warn if `narrow_gap_endpoint_extension < landing_margin` (endpoint would be before arc ends). Decision: Start without warnings, add if users report confusion.

2. **Should gap_width_threshold be automatically derived from terrain config?** E.g., set to midpoint between narrow and wide gap ranges. Decision: No, keep them independent. Users may want different classification than terrain generation.

3. **Should we support per-gap-width parameter interpolation?** E.g., linearly interpolate endpoint extension based on exact gap width rather than binary narrow/wide classification. Decision: Not in this change. Can be added later if needed without breaking compatibility.

4. **Should post_jump_distance affect waypoint generation?** E.g., add extra waypoints in the post-jump segment. Decision: No, keep it simple. Just extend the endpoint, existing waypoint generation will handle it.

5. **Should we add a "gap detection sensitivity" parameter?** E.g., adjust `gap_threshold` based on terrain roughness. Decision: Out of scope. `gap_threshold` is already configurable in existing `JumpParams`.
