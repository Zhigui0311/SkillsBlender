## 1. Add Configuration Parameters to JumpParams

- [x] 1.1 Add `gap_width_threshold` field to JumpParams dataclass in `path_command_cfg.py` with default value 0.55
- [x] 1.2 Add `narrow_gap_endpoint_extension` field to JumpParams with default value 1.5
- [x] 1.3 Add `wide_gap_endpoint_extension` field to JumpParams with default value 2.0
- [x] 1.4 Add `narrow_gap_takeoff_margin` field to JumpParams with default value 0.4
- [x] 1.5 Add `wide_gap_takeoff_margin` field to JumpParams with default value 0.5
- [x] 1.6 Add `landing_margin` field to JumpParams with default value 0.5
- [x] 1.7 Add `post_jump_distance` field to JumpParams with default value 0.0
- [x] 1.8 Add docstring comments explaining each new parameter

## 2. Update Jump Trajectory Planning Logic

- [x] 2.1 Replace hardcoded `0.55` gap width threshold with `self.cfg.jump_params.gap_width_threshold` in `jump_path_command.py`
- [x] 2.2 Replace hardcoded endpoint extensions (1.5m/2.0m) with `narrow_gap_endpoint_extension` and `wide_gap_endpoint_extension` from config
- [x] 2.3 Replace hardcoded takeoff margins (0.4m/0.5m) with `narrow_gap_takeoff_margin` and `wide_gap_takeoff_margin` from config
- [x] 2.4 Replace hardcoded landing margin (0.5m) with `self.cfg.jump_params.landing_margin` from config
- [x] 2.5 Add logic to apply `post_jump_distance` to trajectory endpoint when > 0

## 3. Update Default Configurations

- [ ] 3.1 Add comments to `jump_env_cfg.py` documenting the new JumpParams fields
- [ ] 3.2 Verify that default CommandsCfg uses the new default values (no explicit override needed)
- [ ] 3.3 Update `go2_jump_cfg.py` to explicitly set jump parameters with comments explaining GO2-specific tuning
- [ ] 3.4 Update `go2_jump_cur_cfg.py` to explicitly set jump parameters for curriculum training

## 4. Add Parameter Validation

- [ ] 4.1 Add validation in JumpPathCommand.__init__ to check that margins and extensions are >= 0
- [ ] 4.2 Add validation that gap_width_threshold > 0
- [ ] 4.3 Add warning log if narrow_gap_endpoint_extension < landing_margin (potential configuration issue)
- [ ] 4.4 Add warning log if takeoff margins are very large (> 1.0m) as this may indicate misconfiguration

## 5. Update Existing Hardcoded References

- [ ] 5.1 Search for any remaining hardcoded references to 0.4, 0.5, 1.5, 2.0 in jump_path_command.py and replace with config parameters
- [ ] 5.2 Remove or update comments that reference old hardcoded values
- [ ] 5.3 Verify that both height_scanner and raycast branches use config parameters consistently

## 6. Testing and Verification

- [ ] 6.1 Test with default configuration to verify behavior matches previous implementation
- [ ] 6.2 Test with custom narrow_gap_endpoint_extension (e.g., 1.0m) and verify endpoint positioning
- [ ] 6.3 Test with custom wide_gap_endpoint_extension (e.g., 2.5m) and verify endpoint positioning
- [ ] 6.4 Test with post_jump_distance = 1.0 and verify trajectory extends beyond landing point
- [ ] 6.5 Test with modified gap_width_threshold (e.g., 0.4m) and verify gap classification changes
- [ ] 6.6 Test with very small margins (0.1m) and verify no crashes or invalid trajectories
- [ ] 6.7 Test with very large margins (1.0m) and verify behavior is reasonable
- [ ] 6.8 Verify backward compatibility by running with an old config file that doesn't specify new parameters

## 7. Documentation

- [ ] 7.1 Add inline comments in path_command_cfg.py explaining parameter meanings and typical ranges
- [ ] 7.2 Add example configuration snippet showing conservative jump profile (small margins, large extensions)
- [ ] 7.3 Add example configuration snippet showing aggressive jump profile (large margins, small extensions)
- [ ] 7.4 Document parameter interactions (e.g., how gap_width_threshold affects which margins are used)
