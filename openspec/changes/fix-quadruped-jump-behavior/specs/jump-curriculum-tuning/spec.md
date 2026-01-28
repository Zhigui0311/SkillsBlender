## ADDED Requirements

### Requirement: Gentler gap width progression
The system SHALL use a more gradual gap width progression to ensure the robot masters each difficulty level.

#### Scenario: Initial gap width range
- **WHEN** curriculum level is 0-1
- **THEN** gap width SHALL be 0.2-0.3m (reduced from 0.3-0.5m)

#### Scenario: Intermediate gap width range
- **WHEN** curriculum level is 2-3
- **THEN** gap width SHALL be 0.3-0.5m

#### Scenario: Advanced gap width range
- **WHEN** curriculum level is 4+
- **THEN** gap width SHALL be 0.5-0.8m (reduced from 0.6-1.0m)

### Requirement: Higher progression thresholds
The system SHALL require higher average rewards before advancing curriculum difficulty.

#### Scenario: Gap width progression threshold
- **WHEN** evaluating gap width curriculum advancement
- **THEN** the average reward threshold SHALL be 70.0 (increased from 50.0)

#### Scenario: Height requirement progression threshold
- **WHEN** evaluating height requirement curriculum advancement
- **THEN** the average reward threshold SHALL be 80.0 (increased from 60.0)

#### Scenario: Terrain mix progression threshold
- **WHEN** evaluating terrain mix curriculum advancement
- **THEN** the average reward threshold SHALL be 60.0 (increased from 40.0)

### Requirement: Extended flat terrain training
The system SHALL maintain a higher proportion of flat terrain in early training stages.

#### Scenario: Initial terrain mix
- **WHEN** curriculum level is 0-1
- **THEN** terrain mix SHALL be 50% flat, 40% narrow gaps, 10% wide gaps (increased flat from 20%)

#### Scenario: Intermediate terrain mix
- **WHEN** curriculum level is 2-3
- **THEN** terrain mix SHALL be 30% flat, 50% narrow gaps, 20% wide gaps

#### Scenario: Advanced terrain mix
- **WHEN** curriculum level is 4+
- **THEN** terrain mix SHALL be 10% flat, 60% narrow gaps, 30% wide gaps

### Requirement: Conservative jump height progression
The system SHALL use smaller increments for jump height requirements.

#### Scenario: Initial jump height
- **WHEN** curriculum level is 0-1
- **THEN** required jump height SHALL be 0.20m (reduced from 0.25m)

#### Scenario: Height increment size
- **WHEN** advancing to the next curriculum level
- **THEN** jump height SHALL increase by 0.03m (reduced from 0.05m)

#### Scenario: Maximum jump height
- **WHEN** curriculum reaches maximum level
- **THEN** required jump height SHALL not exceed 0.40m (reduced from 0.45m)

### Requirement: Slower speed requirement progression
The system SHALL delay speed requirement increases until basic jumping is mastered.

#### Scenario: Initial speed requirement
- **WHEN** curriculum level is 0-2
- **THEN** forward speed requirement SHALL be 0.8 m/s (reduced from 1.0 m/s)

#### Scenario: Speed progression threshold
- **WHEN** evaluating speed requirement curriculum advancement
- **THEN** the average reward threshold SHALL be 75.0 (increased from 65.0)

#### Scenario: Speed increment size
- **WHEN** advancing to the next curriculum level
- **THEN** forward speed requirement SHALL increase by 0.08 m/s (reduced from 0.1 m/s)
