## ADDED Requirements

### Requirement: Configurable narrow gap width range
The system SHALL allow users to configure the width range for narrow gaps in terrain generation.

#### Scenario: Narrow gap range configuration
- **WHEN** terrain is generated with narrow gaps
- **THEN** gap widths SHALL be randomly sampled from the configured narrow_gap_width_range (min, max)

#### Scenario: Default narrow gap range
- **WHEN** narrow_gap_width_range is not explicitly configured
- **THEN** the system SHALL use a default range that maintains current behavior

### Requirement: Configurable wide gap width range
The system SHALL allow users to configure the width range for wide gaps in terrain generation.

#### Scenario: Wide gap range configuration
- **WHEN** terrain is generated with wide gaps
- **THEN** gap widths SHALL be randomly sampled from the configured wide_gap_width_range (min, max)

#### Scenario: Default wide gap range
- **WHEN** wide_gap_width_range is not explicitly configured
- **THEN** the system SHALL use a default range that maintains current behavior

### Requirement: Gap range validation
The system SHALL validate that gap width ranges are physically reasonable and do not overlap inappropriately.

#### Scenario: Minimum width validation
- **WHEN** a gap width range is configured
- **THEN** the minimum width SHALL be greater than 0 meters

#### Scenario: Range ordering validation
- **WHEN** a gap width range is configured
- **THEN** the maximum width SHALL be greater than or equal to the minimum width

#### Scenario: Narrow-wide separation
- **WHEN** both narrow and wide gap ranges are configured
- **THEN** the system SHOULD warn if the ranges overlap significantly (e.g., narrow max > wide min)

### Requirement: Terrain configuration integration
The system SHALL integrate gap width range parameters into the terrain generator configuration structure.

#### Scenario: Parameters in terrain config
- **WHEN** a user configures terrain generation
- **THEN** gap width ranges SHALL be accessible as part of the terrain sub-terrain configuration

#### Scenario: Per-terrain-type configuration
- **WHEN** multiple gap terrain types are defined (e.g., narrow_gaps, wide_gaps)
- **THEN** each terrain type SHALL have its own configurable gap_width_range parameter

### Requirement: Runtime gap size query
The system SHALL allow the jump command to query the configured gap size ranges to determine appropriate trajectory parameters.

#### Scenario: Gap classification based on configured ranges
- **WHEN** the jump command detects a gap
- **THEN** it SHALL classify the gap as narrow or wide based on the configured gap width ranges

#### Scenario: Adaptive trajectory parameters
- **WHEN** gap width ranges are modified in configuration
- **THEN** the jump trajectory planning SHALL automatically adapt to use appropriate parameters for the new gap definitions
