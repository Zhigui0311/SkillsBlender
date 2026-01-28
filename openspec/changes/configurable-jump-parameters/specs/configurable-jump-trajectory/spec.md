## ADDED Requirements

### Requirement: Configurable endpoint distance extensions
The system SHALL allow users to configure how far beyond the gap end the trajectory endpoint should be placed, with separate settings for narrow and wide gaps.

#### Scenario: Narrow gap endpoint extension
- **WHEN** a narrow gap (width < 0.55m) is detected
- **THEN** the endpoint SHALL be placed at `gap_end + narrow_gap_endpoint_extension` meters

#### Scenario: Wide gap endpoint extension
- **WHEN** a wide gap (width >= 0.55m) is detected
- **THEN** the endpoint SHALL be placed at `gap_end + wide_gap_endpoint_extension` meters

#### Scenario: No gap detected
- **WHEN** no gap is detected within scan distance
- **THEN** the endpoint SHALL be placed at the default path length from start

### Requirement: Configurable gap width threshold
The system SHALL allow users to configure the threshold that distinguishes narrow gaps from wide gaps.

#### Scenario: Gap classification
- **WHEN** a gap is detected with width W
- **THEN** the gap SHALL be classified as narrow if W < gap_width_threshold, otherwise wide

### Requirement: Configurable takeoff margin
The system SHALL allow users to configure the distance before the gap where the parabolic arc begins, with separate settings for narrow and wide gaps.

#### Scenario: Narrow gap takeoff margin
- **WHEN** a narrow gap is detected
- **THEN** the parabolic arc SHALL begin at `gap_start - narrow_gap_takeoff_margin` meters

#### Scenario: Wide gap takeoff margin
- **WHEN** a wide gap is detected
- **THEN** the parabolic arc SHALL begin at `gap_start - wide_gap_takeoff_margin` meters

### Requirement: Configurable landing margin
The system SHALL allow users to configure the distance after the gap where the parabolic arc ends.

#### Scenario: Landing margin application
- **WHEN** any gap is detected
- **THEN** the parabolic arc SHALL end at `gap_end + landing_margin` meters

### Requirement: Optional post-jump forward distance
The system SHALL allow users to optionally configure an additional forward distance after landing to continue the trajectory.

#### Scenario: Post-jump distance enabled
- **WHEN** post_jump_distance is configured and > 0
- **THEN** the trajectory endpoint SHALL be extended by post_jump_distance beyond the landing point

#### Scenario: Post-jump distance disabled
- **WHEN** post_jump_distance is 0 or not configured
- **THEN** the trajectory SHALL end at the landing point without additional extension

### Requirement: Configuration parameter exposure
The system SHALL expose all jump trajectory parameters in JumpPathCommandCfg for centralized configuration.

#### Scenario: Parameters available in configuration
- **WHEN** a user creates a custom configuration file
- **THEN** all jump trajectory parameters (endpoint extensions, margins, thresholds, post-jump distance) SHALL be accessible as configuration attributes

#### Scenario: Default values provided
- **WHEN** a parameter is not explicitly set in user configuration
- **THEN** the system SHALL use sensible default values that maintain current behavior

### Requirement: Backward compatibility
The system SHALL maintain backward compatibility with existing configurations that do not specify the new parameters.

#### Scenario: Legacy configuration support
- **WHEN** an existing configuration file is used without the new parameters
- **THEN** the system SHALL use default values and produce the same behavior as before this change
