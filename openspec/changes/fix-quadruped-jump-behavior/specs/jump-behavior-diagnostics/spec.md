## ADDED Requirements

### Requirement: Diagnostic logging for jump initiation
The system SHALL log diagnostic information when the robot fails to initiate a jump within expected timeframes.

#### Scenario: Robot spinning at start position
- **WHEN** the robot remains within 0.5m of the start position for more than 2 seconds
- **THEN** the system SHALL log the robot's state including base velocity, joint positions, current waypoint, and reward components

#### Scenario: No forward velocity detected
- **WHEN** the robot's forward velocity remains below 0.3 m/s for more than 1 second after jump command
- **THEN** the system SHALL log a warning with current action outputs and policy network activations

### Requirement: Visualization of jump trajectory
The system SHALL provide visual debugging tools to display the planned jump trajectory and robot's actual path.

#### Scenario: Trajectory visualization in play mode
- **WHEN** running in play/visualization mode
- **THEN** the system SHALL render the planned parabolic arc, waypoints, gap boundaries, and endpoint position

#### Scenario: Real-time path deviation display
- **WHEN** the robot deviates from the planned trajectory by more than 0.5m
- **THEN** the system SHALL highlight the deviation in the visualization with distance metrics

### Requirement: Reward component breakdown logging
The system SHALL log individual reward component values to identify which rewards are dominating the policy.

#### Scenario: Reward analysis at episode start
- **WHEN** an episode begins
- **THEN** the system SHALL log all reward weights and their initial values

#### Scenario: Abnormal reward detection
- **WHEN** any single reward component exceeds 80% of the total reward magnitude
- **THEN** the system SHALL log a warning indicating potential reward imbalance

### Requirement: Waypoint progression tracking
The system SHALL track and log waypoint advancement to detect premature or stuck waypoint updates.

#### Scenario: Waypoint skipping detection
- **WHEN** the robot advances to the next waypoint without being within 0.8m of the current waypoint
- **THEN** the system SHALL log a warning with the robot's position and waypoint positions

#### Scenario: Waypoint stuck detection
- **WHEN** the robot remains on the same waypoint for more than 3 seconds
- **THEN** the system SHALL log the robot's state and distance to the waypoint
