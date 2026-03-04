"""Math utilities and constants for numerical operations.

This module provides named constants for numerical tolerances and small
epsilon values used throughout the fyst_pointing library.

Constants
---------
ZENITH_COS_EPSILON : float
    Minimum value for cos(elevation) to prevent division by zero near zenith.
    At 89.9999 degrees elevation, cos(el) = 1.7e-6, so 1e-10 provides a safe
    floor while allowing operations very close to zenith.

SMALL_DISTANCE_EPSILON : float
    Epsilon for detecting near-zero distances or radii.
    Used when checking if position is effectively at center/origin.

SMALL_TIME_EPSILON : float
    Epsilon for detecting near-zero time intervals.
    Used to prevent division by zero in velocity calculations.
"""

ZENITH_COS_EPSILON: float = 1e-10

SMALL_DISTANCE_EPSILON: float = 1e-10

SMALL_TIME_EPSILON: float = 1e-10
