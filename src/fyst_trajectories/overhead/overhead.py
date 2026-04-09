"""Calibration state tracking and overhead injection.

Manages when calibrations are due based on cadence policies,
inspired by SO schedlib's State-tracking approach.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from astropy.time import Time

from .models import CalibrationPolicy, CalibrationSpec, CalibrationType, OverheadModel

if TYPE_CHECKING:
    from ..coordinates import Coordinates

__all__ = [
    "CalibrationState",
]


@dataclass(frozen=True)
class CalibrationState:
    """Tracks when each calibration type was last performed.

    Used by the scheduler to determine which calibrations are due
    at a given time based on the calibration policy cadences.

    Parameters
    ----------
    last_retune : Time or None
        Last KID probe tone reset time.
    last_pointing_cal : Time or None
        Last pointing correction time.
    last_focus : Time or None
        Last focus check time.
    last_skydip : Time or None
        Last sky dip time.
    last_planet_cal : Time or None
        Last planet calibration time.
    """

    last_retune: Time | None = None
    last_pointing_cal: Time | None = None
    last_focus: Time | None = None
    last_skydip: Time | None = None
    last_planet_cal: Time | None = None

    def needs_calibration(
        self,
        current_time: Time,
        policy: CalibrationPolicy,
        overhead: OverheadModel,
        coords: Coordinates | None = None,
    ) -> list[CalibrationSpec]:
        """Determine which calibrations are due.

        Checks each calibration type against its cadence. A cadence of 0
        means "every scan boundary" -- the retune is always needed in that
        case (the scheduler inserts it between scans).

        Returns calibrations in priority order: retune, pointing, focus,
        skydip, planet_cal.

        Parameters
        ----------
        current_time : Time
            Current scheduling time.
        policy : CalibrationPolicy
            Cadence configuration.
        overhead : OverheadModel
            Duration information for calibration operations.
        coords : Coordinates or None
            Coordinate transformer. When provided, planet_cal is only
            included if at least one planet target is above the horizon.

        Returns
        -------
        list of CalibrationSpec
            Calibrations that should be performed now.
        """
        needed = []

        checks = [
            (CalibrationType.RETUNE, self.last_retune, policy.retune_cadence),
            (CalibrationType.POINTING_CAL, self.last_pointing_cal, policy.pointing_cadence),
            (CalibrationType.FOCUS, self.last_focus, policy.focus_cadence),
            (CalibrationType.SKYDIP, self.last_skydip, policy.skydip_cadence),
            (CalibrationType.PLANET_CAL, self.last_planet_cal, policy.planet_cal_cadence),
        ]

        for cal_type, last_time, cadence in checks:
            if self._is_due(current_time, last_time, cadence):
                target = None
                if cal_type == CalibrationType.PLANET_CAL and policy.planet_targets:
                    target = self._find_visible_planet(
                        policy.planet_targets,
                        current_time,
                        coords,
                        min_elevation=policy.planet_min_elevation,
                    )
                    if target is None:
                        continue
                needed.append(
                    CalibrationSpec(
                        name=cal_type,
                        duration=overhead.get_calibration_duration(cal_type),
                        target=target,
                    )
                )

        return needed

    @staticmethod
    def _find_visible_planet(
        planet_targets: tuple[str, ...],
        current_time: Time,
        coords: Coordinates | None,
        min_elevation: float = 20.0,
    ) -> str | None:
        """Return the first visible planet target, or fall back to first target.

        Parameters
        ----------
        planet_targets : tuple of str
            Planet names to check.
        current_time : Time
            Current time.
        coords : Coordinates or None
            Coordinate transformer. If None, returns the first planet
            (no visibility check).
        min_elevation : float
            Minimum altitude in degrees for a planet to be considered
            visible. Default is 20.0 degrees.

        Returns
        -------
        str or None
            Name of a visible planet, or None if coords is provided
            and no planet is above the minimum elevation.
        """
        if coords is None:
            return planet_targets[0]

        for planet in planet_targets:
            _, alt = coords.get_body_altaz(planet, current_time)
            if alt > min_elevation:
                return planet

        return None

    def update(self, cal_type: CalibrationType | str, time: Time) -> CalibrationState:
        """Return a new state with the given calibration type updated.

        Parameters
        ----------
        cal_type : CalibrationType or str
            Calibration type name.
        time : Time
            Time the calibration was performed.

        Returns
        -------
        CalibrationState
            New state with the updated timestamp.
        """
        if isinstance(cal_type, str) and not isinstance(cal_type, CalibrationType):
            cal_type = CalibrationType(cal_type)
        field_map = {
            CalibrationType.RETUNE: "last_retune",
            CalibrationType.POINTING_CAL: "last_pointing_cal",
            CalibrationType.FOCUS: "last_focus",
            CalibrationType.SKYDIP: "last_skydip",
            CalibrationType.PLANET_CAL: "last_planet_cal",
        }
        if cal_type not in field_map:
            raise ValueError(f"Unknown calibration type: {cal_type}")

        return dataclasses.replace(self, **{field_map[cal_type]: time})

    @staticmethod
    def _is_due(current_time: Time, last_time: Time | None, cadence: float) -> bool:
        """Check whether a calibration is due.

        Parameters
        ----------
        current_time : Time
            Current time.
        last_time : Time or None
            Last time this calibration was performed (None = never).
        cadence : float
            Required interval in seconds. 0 means always due.

        Returns
        -------
        bool
            True if the calibration should be performed.
        """
        if last_time is None:
            return True
        if cadence == 0:
            return True
        elapsed = (current_time - last_time).sec
        return elapsed >= cadence
