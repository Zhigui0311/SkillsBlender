## Context

The quadruped robot's jump training has failed to produce functional jumping behavior. Instead of jumping over gaps, the robot exhibits spinning and leg-lifting behavior at the start position, never initiating forward momentum or takeoff. Analysis of the current implementation reveals several issues:

**Current State:**
- Jump trajectory planning in `jump_path_command.py` uses fixed margins (0.2m takeoff, 0.3m landing)
- Endpoint is placed at `gap_end + 1.0m`, which is too close to the gap edge
- Reward weights may not sufficiently encourage forward commitment
- Curriculum progression may be too aggressive, jumping to difficult gaps before basic jumping is learned
- Waypoint advancement logic uses the same threshold (0.8m) for all phases

**Constraints:**
- Must maintain compatibility with existing Isaac Lab simulation framework
- Cannot change the fundamental RL training loop or policy network architecture
- Must preserve existing reward function interfaces
- Training time should not increase significantly

**Stakeholders:**
- RL training pipeline (affected by reward weight changes)
- Visualization/debugging tools (need new diagnostic outputs)
- Deployment systems (benefit from improved jump reliability)

## Goals / Non-Goals

**Goals:**
- Fix the spinning/stalling behavior at the start position
- Ensure the robot initiates forward momentum and commits to jumps
- Provide safer landing zones by extending endpoint positioning
- Enable progressive learning through gentler curriculum progression
- Add diagnostic tools to identify future training issues

**Non-Goals:**
- Redesigning the entire reward system architecture
- Changing the policy network structure or hyperparameters
- Modifying the terrain generation system beyond parameter adjustments
- Implementing new termination conditions (existing ones are sufficient)
- Supporting jumps wider than 1.0m (out of scope for this change)

## Decisions

### Decision 1: Extend endpoint positioning beyond gap
**Choice:** Place endpoint at `gap_end + 1.5m` for narrow gaps and `gap_end + 2.0m` for wide gaps.

**Rationale:** The current `gap_end + 1.0m` positioning places the target too close to the gap edge. When the robot lands, it often overshoots slightly and falls back into the gap. Extending the endpoint provides a safety buffer and gives the robot a clear "safe zone" to aim for.

**Alternatives considered:**
- Fixed 2.5m extension: Too far for narrow gaps, wastes training time
- Adaptive based on robot velocity: Too complex, adds state dependency
- Keep current 1.0m: Insufficient, root cause of falling issue

### Decision 2: Increase takeoff margin based on gap width
**Choice:** Use 0.4m margin for narrow gaps (0.3-0.5m) and 0.5m margin for wide gaps (0.6-1.0m).

**Rationale:** The current 0.2m margin doesn't give the robot enough preparation time. By starting the parabolic arc earlier, the robot has more time to build upward momentum before reaching the gap edge. Wider gaps need more preparation time.

**Alternatives considered:**
- Fixed 0.5m for all gaps: Simpler but may cause premature jumping for narrow gaps
- Velocity-dependent margin: More accurate but adds complexity and state coupling
- Keep current 0.2m: Insufficient, contributes to late/failed takeoffs

### Decision 3: Increase forward velocity reward weight dynamically
**Choice:** Use weight 3.5 when within 2.0m of gap, weight 4.0 when in air.

**Rationale:** The current weight (2.5) is dominated by other rewards, allowing the robot to optimize for spinning/leg-lifting instead of forward progress. By increasing the weight near the gap and during flight, we strongly encourage commitment to the jump.

**Alternatives considered:**
- Fixed high weight (5.0) throughout: May cause rushing and poor control
- Exponential increase near gap: More complex, harder to tune
- Keep current 2.5: Insufficient, allows spinning behavior to dominate

### Decision 4: Add strong spinning penalty
**Choice:** Apply -3.0 penalty when angular velocity > 1.0 rad/s while forward velocity < 0.3 m/s.

**Rationale:** The spinning behavior is a failure mode that needs explicit discouragement. This penalty specifically targets the problematic behavior pattern without affecting normal turning during locomotion.

**Alternatives considered:**
- Penalize all angular velocity: Too restrictive, prevents normal turning
- Increase existing stalling penalty only: Doesn't specifically target spinning
- Add orientation tracking reward: Positive rewards are weaker than negative penalties for eliminating bad behaviors

### Decision 5: Gentler curriculum progression with higher thresholds
**Choice:** Start with 0.2-0.3m gaps (vs 0.3-0.5m), require 70.0 avg reward for gap width progression (vs 50.0).

**Rationale:** The current curriculum may be advancing too quickly, before the robot has mastered basic jumping. By starting easier and requiring higher performance before progression, we ensure solid fundamentals.

**Alternatives considered:**
- Keep current progression, add more levels: Doesn't address threshold issue
- Manual curriculum control: Requires human intervention, not scalable
- Adaptive per-environment progression only: Already exists, but global progression also needs adjustment

### Decision 6: Stricter waypoint threshold during jump phase
**Choice:** Use 0.5m threshold during jump segment (gap_start to gap_end), keep 0.8m elsewhere.

**Rationale:** The current 0.8m threshold may allow waypoint skipping during jumps, confusing the policy about where it should be. A stricter threshold during the critical jump phase ensures accurate tracking.

**Alternatives considered:**
- Fixed 0.5m throughout: Too strict for normal locomotion, may cause stuttering
- Velocity-dependent threshold: More complex, harder to debug
- Keep current 0.8m: May contribute to waypoint confusion during jumps

### Decision 7: Add diagnostic logging without performance impact
**Choice:** Log diagnostics only when anomalies are detected (spinning, stuck waypoints, reward imbalance).

**Rationale:** Continuous logging would slow down training significantly (4096 environments). Event-triggered logging provides debugging information without performance cost.

**Alternatives considered:**
- Continuous logging: Too slow, generates massive log files
- No logging, rely on tensorboard: Insufficient detail for debugging specific failure modes
- Separate diagnostic mode: Requires maintaining two code paths

## Risks / Trade-offs

### Risk 1: Reward weight changes may require retraining from scratch
**Mitigation:** Test with a small number of environments first. If policy diverges significantly, retrain from scratch. If it adapts, continue from current checkpoint.

### Risk 2: Extended endpoint may increase episode length
**Mitigation:** The 15-second episode timeout is sufficient for the extended distance. Monitor average episode length during training.

### Risk 3: Gentler curriculum may slow initial training progress
**Trade-off:** Accepted. Slower but more reliable learning is preferable to fast learning that produces non-functional behavior.

### Risk 4: Stricter waypoint threshold may cause stuttering near waypoints
**Mitigation:** Only apply during jump phase. Monitor for oscillation behavior in logs.

### Risk 5: New penalties may create local minima (robot learns to avoid gaps entirely)
**Mitigation:** Ensure forward velocity reward and path tracking rewards remain dominant. The spinning penalty only activates for extreme behavior.

### Risk 6: Diagnostic logging may still impact performance if triggered frequently
**Mitigation:** Use rate limiting (max 1 log per second per environment). If performance impact is observed, make logging optional via config flag.

## Migration Plan

**Phase 1: Code changes (no training impact)**
1. Update `jump_path_command.py` with new endpoint positioning and margins
2. Add diagnostic logging to `jump_path_command.py` and reward functions
3. Update reward weights in `go2_jump_cur_cfg.py`
4. Update curriculum parameters in `curriculum.py`
5. Add visualization for trajectory in play mode

**Phase 2: Testing with small-scale training**
1. Run training with 32 environments for 1000 iterations
2. Verify robot initiates forward movement (no spinning)
3. Check diagnostic logs for anomalies
4. Validate endpoint positioning in visualization mode

**Phase 3: Full-scale training**
1. If Phase 2 successful, launch full training (4096 environments)
2. Monitor curriculum progression rates
3. Compare jump success rate to baseline (if available)

**Phase 4: Validation**
1. Test trained policy in play mode with visualization
2. Verify robot successfully jumps narrow gaps (0.2-0.5m)
3. Verify robot successfully jumps wide gaps (0.6-0.8m)
4. Confirm safe landing beyond gap edge

**Rollback Strategy:**
- Keep backup of current config files
- If training fails to converge after 5000 iterations, revert changes
- If robot exhibits new failure modes, analyze diagnostic logs and adjust parameters

## Open Questions

1. **Should we add a "commitment point" mechanism?** Once the robot crosses a threshold distance to the gap, lock in the jump trajectory to prevent last-minute hesitation. This could help with commitment but adds complexity.

2. **Should landing margin be velocity-dependent?** Faster approaches need more landing space. Current fixed margin may be insufficient for high-speed jumps in later curriculum stages.

3. **Should we add a separate "approach phase" reward?** Currently we modify existing rewards near the gap. A dedicated approach reward might be clearer but adds another reward component to tune.

4. **What is the optimal balance between height tracking and forward velocity rewards?** Current design increases both, but they may conflict (jumping high vs jumping far). May need empirical tuning.

5. **Should curriculum progression be based on success rate instead of average reward?** Success rate is more interpretable but requires defining "success" criteria. Average reward is noisy but already implemented.
