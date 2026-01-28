## ADDED Requirements

### Requirement: Increased forward velocity reward weight
The system SHALL increase the reward weight for forward velocity to encourage commitment to the jump.

#### Scenario: Forward velocity reward during jump approach
- **WHEN** the robot is within 2.0m of the gap start
- **THEN** the jump_forward_velocity reward weight SHALL be 3.5 (increased from 2.5)

#### Scenario: Forward velocity reward during jump
- **WHEN** the robot is in the air during a jump
- **THEN** the jump_forward_velocity reward weight SHALL be 4.0

### Requirement: Enhanced stalling penalty
The system SHALL apply a stronger penalty for spinning and stalling behavior at the start position.

#### Scenario: Low forward velocity penalty
- **WHEN** the robot's forward velocity is below 0.2 m/s for more than 1 second
- **THEN** the system SHALL apply a stalling penalty of -5.0 (increased from -4.0)

#### Scenario: Spinning behavior penalty
- **WHEN** the robot's angular velocity magnitude exceeds 1.0 rad/s while forward velocity is below 0.3 m/s
- **THEN** the system SHALL apply an additional spinning penalty of -3.0

### Requirement: Progressive height tracking reward
The system SHALL adjust the height tracking reward weight based on distance to the gap.

#### Scenario: Height tracking before gap
- **WHEN** the robot is more than 1.0m before the gap start
- **THEN** the jump_height_tracking reward weight SHALL be 2.0 (reduced to avoid premature jumping)

#### Scenario: Height tracking during jump
- **WHEN** the robot is within the jump segment (gap_start to gap_end)
- **THEN** the jump_height_tracking reward weight SHALL be 5.0 (increased from 4.0)

### Requirement: Early commitment reward
The system SHALL reward the robot for maintaining forward momentum in the approach phase.

#### Scenario: Approach phase momentum reward
- **WHEN** the robot is within 3.0m of the gap start and maintains forward velocity above 1.0 m/s
- **THEN** the system SHALL apply an approach momentum reward of +2.0

#### Scenario: Consistent forward progress reward
- **WHEN** the robot maintains forward velocity above 0.8 m/s for 2 consecutive seconds
- **THEN** the system SHALL apply a consistency reward of +1.5

### Requirement: Reduced joint penalty during jump
The system SHALL reduce joint penalties during the jump phase to allow more dynamic movements.

#### Scenario: Joint torque penalty during jump
- **WHEN** the robot is in the air during a jump
- **THEN** the joint torque penalty weight SHALL be reduced by 50%

#### Scenario: Joint velocity penalty during jump
- **WHEN** the robot is in the air during a jump
- **THEN** the joint velocity penalty weight SHALL be reduced by 50%
