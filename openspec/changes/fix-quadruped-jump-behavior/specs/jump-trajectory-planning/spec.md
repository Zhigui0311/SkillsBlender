## ADDED Requirements

### Requirement: Extended endpoint positioning
The system SHALL position the endpoint at least 1.5m beyond the gap end to provide a safe landing zone.

#### Scenario: Endpoint placement for narrow gaps
- **WHEN** a gap of width 0.3-0.5m is detected
- **THEN** the endpoint SHALL be placed at gap_end + 1.5m minimum

#### Scenario: Endpoint placement for wide gaps
- **WHEN** a gap of width 0.6-1.0m is detected
- **THEN** the endpoint SHALL be placed at gap_end + 2.0m minimum

#### Scenario: No gap detected
- **WHEN** no gap is detected within the scan distance
- **THEN** the endpoint SHALL be placed at the default distance of 5.0m from start

### Requirement: Adjusted takeoff margin
The system SHALL initiate the parabolic arc earlier to ensure proper jump preparation.

#### Scenario: Takeoff margin for narrow gaps
- **WHEN** a narrow gap (0.3-0.5m) is detected
- **THEN** the parabolic arc SHALL begin at gap_start - 0.4m (increased from 0.2m)

#### Scenario: Takeoff margin for wide gaps
- **WHEN** a wide gap (0.6-1.0m) is detected
- **THEN** the parabolic arc SHALL begin at gap_start - 0.5m

### Requirement: Landing margin adjustment
The system SHALL extend the landing zone to ensure the robot has sufficient space to stabilize.

#### Scenario: Landing margin for all gaps
- **WHEN** any gap is detected
- **THEN** the parabolic arc SHALL end at gap_end + 0.5m (increased from 0.3m)

### Requirement: Waypoint update logic for in-air state
The system SHALL use XY-plane distance only when the robot is in the air to prevent waypoint skipping.

#### Scenario: In-air waypoint advancement
- **WHEN** the robot is in the air (no foot contact)
- **THEN** the system SHALL calculate waypoint distance using only XY coordinates, ignoring Z-axis

#### Scenario: On-ground waypoint advancement
- **WHEN** the robot has at least one foot in contact with the ground
- **THEN** the system SHALL calculate waypoint distance using full 3D coordinates

### Requirement: Adaptive waypoint threshold
The system SHALL use a stricter waypoint advancement threshold during the jump phase.

#### Scenario: Jump phase waypoint threshold
- **WHEN** the robot is within the jump segment (between gap_start and gap_end)
- **THEN** the waypoint advancement threshold SHALL be 0.5m (reduced from 0.8m)

#### Scenario: Normal phase waypoint threshold
- **WHEN** the robot is outside the jump segment
- **THEN** the waypoint advancement threshold SHALL remain at 0.8m
