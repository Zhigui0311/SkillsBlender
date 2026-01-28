## 1. Update Jump Trajectory Planning

- [x] 1.1 Modify endpoint positioning logic in `jump_path_command.py` to use gap-width-dependent extensions (1.5m for narrow, 2.0m for wide)
- [x] 1.2 Update takeoff margin calculation to use 0.4m for narrow gaps and 0.5m for wide gaps
- [x] 1.3 Update landing margin to 0.5m for all gaps
- [x] 1.4 Implement adaptive waypoint threshold (0.5m during jump phase, 0.8m elsewhere)
- [x] 1.5 Add gap width detection logic to determine narrow vs wide gap classification

## 2. Add Diagnostic Logging

- [ ] 2.1 Add spinning detection logging in `jump_path_command.py` (triggers when robot stays within 0.5m of start for >2s)
- [ ] 2.2 Add forward velocity warning logging (triggers when velocity <0.3 m/s for >1s after jump command)
- [ ] 2.3 Add waypoint skipping detection and logging
- [ ] 2.4 Add waypoint stuck detection and logging (same waypoint for >3s)
- [ ] 2.5 Implement rate limiting for diagnostic logs (max 1 log per second per environment)

## 3. Update Reward Weights and Add New Penalties

- [ ] 3.1 Modify `jump_forward_velocity` reward in `jump_rewards.py` to use dynamic weights (3.5 near gap, 4.0 in air)
- [ ] 3.2 Add distance-to-gap calculation to enable proximity-based reward scaling
- [x] 3.3 Implement spinning penalty in `jump_rewards.py` (-3.0 when angular velocity >1.0 rad/s and forward velocity <0.3 m/s)
- [x] 3.4 Update stalling penalty weight to -5.0 in `go2_jump_cur_cfg.py`
- [x] 3.5 Add approach momentum reward (+2.0 when within 3.0m of gap and velocity >1.0 m/s)
- [x] 3.6 Add consistency reward (+1.5 when maintaining velocity >0.8 m/s for 2s)
- [ ] 3.7 Implement dynamic height tracking reward (2.0 before gap, 5.0 during jump)
- [ ] 3.8 Add joint penalty reduction during jump phase (50% reduction for torque and velocity penalties)

## 4. Update Curriculum Parameters

- [x] 4.1 Modify `curriculum_jump_gap_width` in `curriculum.py` to use gentler progression (0.2-0.3m initial, 0.3-0.5m intermediate, 0.5-0.8m advanced)
- [x] 4.2 Update gap width progression threshold to 70.0 (from 50.0)
- [x] 4.3 Update height requirement progression threshold to 80.0 (from 60.0)
- [x] 4.4 Update terrain mix progression threshold to 60.0 (from 40.0)
- [x] 4.5 Modify `curriculum_jump_terrain_mix` to use extended flat terrain (50% flat at level 0-1, 30% at level 2-3, 10% at level 4+)
- [x] 4.6 Update `curriculum_jump_height_requirement` to start at 0.20m with 0.03m increments (max 0.40m)
- [x] 4.7 Update `curriculum_jump_speed_requirement` to start at 0.8 m/s with 0.08 m/s increments, threshold 75.0

## 5. Add Visualization Support

- [ ] 5.1 Add trajectory visualization rendering in play mode (parabolic arc, waypoints, gap boundaries)
- [ ] 5.2 Add endpoint position marker visualization
- [ ] 5.3 Add path deviation highlighting (when deviation >0.5m)
- [ ] 5.4 Add real-time reward component display in visualization mode

## 6. Update Configuration Files

- [ ] 6.1 Update reward weights in `go2_jump_cur_cfg.py` to reflect new dynamic weights
- [ ] 6.2 Add new reward function entries for spinning penalty, approach momentum, and consistency rewards
- [ ] 6.3 Update curriculum configuration in `go2_jump_cur_cfg.py` with new thresholds
- [ ] 6.4 Add diagnostic logging configuration flags (enable/disable, rate limit settings)

## 7. Testing and Validation

- [ ] 7.1 Create backup of current configuration files
- [ ] 7.2 Run small-scale test (32 environments, 1000 iterations) to verify no spinning behavior
- [ ] 7.3 Verify diagnostic logs are generated correctly for test scenarios
- [ ] 7.4 Validate endpoint positioning in visualization mode
- [ ] 7.5 Check that curriculum progression follows new thresholds
- [ ] 7.6 Monitor training convergence and reward trends
