"""Pure helpers used by scheduler phases.

Dependency-free (no scheduler state); consumed by
:mod:`.phases` and the :func:`generate_timeline` wrapper.
"""

import math

import numpy as np
from astropy.time import Time, TimeDelta

from ...coordinates import Coordinates
from ...site import Site
from ..constraints import Constraint, ElevationConstraint, SunAvoidanceConstraint
from ..models import ObservingPatch, OverheadModel
from ..utils import get_observable_windows

__all__ = [
    "_compute_az_range",
    "_compute_scan_duration",
    "_default_constraints",
    "_evaluate_patch",
    "_time_until_set",
]


def _default_constraints(site: Site) -> list[Constraint]:
    """Create default constraints from site configuration."""
    constraints: list[Constraint] = [
        ElevationConstraint(
            el_min=site.telescope_limits.elevation.min,
            el_max=site.telescope_limits.elevation.max,
        ),
    ]
    if site.sun_avoidance.enabled:
        constraints.append(SunAvoidanceConstraint(min_angle=site.sun_avoidance.exclusion_radius))
    return constraints


def _evaluate_patch(
    patch: ObservingPatch,
    time: Time,
    az: float,
    el: float,
    coords: Coordinates,
    constraints: list[Constraint],
) -> float:
    """Evaluate a patch against all constraints.

    Returns the product of all constraint scores. A zero from any
    constraint immediately returns 0.0 (short-circuit).
    """
    score = 1.0
    for constraint in constraints:
        s = constraint.score(patch, time, az, el, coords)
        if s == 0.0:
            return 0.0
        score *= s
    return score


def _time_until_set(
    ra: float,
    dec: float,
    start_time: Time,
    max_duration: float,
    coords: Coordinates,
    min_elevation: float,
    step_seconds: float = 300.0,
) -> float:
    """Compute how long a source stays above *min_elevation* from *start_time*.

    Samples elevation at ``step_seconds`` intervals up to *max_duration*
    using a single vectorised ``radec_to_altaz`` call, then linearly
    interpolates to find the crossing time.

    Returns *max_duration* if the source never drops below the limit
    (circumpolar or long transit).
    """
    n_steps = max(2, int(max_duration / step_seconds) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    _, el_arr = coords.radec_to_altaz(
        np.full(n_steps, ra),
        np.full(n_steps, dec),
        times,
    )

    below = np.where(el_arr < min_elevation)[0]
    if len(below) == 0:
        return max_duration

    idx = below[0]
    if idx == 0:
        return 0.0

    # Linear interpolation for sub-step precision at the crossing.
    el_prev = float(el_arr[idx - 1])
    el_curr = float(el_arr[idx])
    denom = el_prev - el_curr
    if abs(denom) < 1e-12:
        frac = 0.5
    else:
        frac = (el_prev - min_elevation) / denom
    return float(dt[idx - 1] + frac * (dt[idx] - dt[idx - 1]))


def _compute_scan_duration(
    patch: ObservingPatch,
    start_time: Time,
    end_time: Time,
    site: Site,
    coords: Coordinates,
    overhead: OverheadModel,
    center_el: float = 50.0,
) -> float:
    """Compute how long we can observe this patch.

    For constant-elevation scans, the duration is determined by how long
    the field crosses the elevation.  For pong/daisy, we start with the
    max scan duration (or remaining schedule time) and then clip to the
    observability window so the source never drops below the telescope
    elevation limit mid-scan.

    Parameters
    ----------
    center_el : float
        Computed elevation of the patch center at the current time.
        Used as fallback when patch.elevation is None.
    """
    remaining = (end_time - start_time).sec

    if patch.scan_type == "constant_el":
        el = patch.elevation if patch.elevation is not None else center_el
        windows = get_observable_windows(
            patch.ra_center,
            patch.dec_center,
            start_time,
            end_time,
            site,
            min_elevation=el - 1.0,  # Slightly below target elevation
            check_sun=site.sun_avoidance.enabled,
        )
        if not windows:
            return 0.0
        for w_start, w_end in windows:
            if w_end.unix > start_time.unix:
                window_dur = (w_end - max(w_start, start_time, key=lambda t: t.unix)).sec
                return min(window_dur, remaining)
        return 0.0
    else:
        max_dur = min(overhead.max_scan_duration, remaining)
        el_min = site.telescope_limits.elevation.min
        observable_dur = _time_until_set(
            patch.ra_center,
            patch.dec_center,
            start_time,
            max_dur,
            coords,
            el_min,
        )
        return min(max_dur, observable_dur)


def _compute_az_range(
    patch: ObservingPatch, center_az: float, center_el: float
) -> tuple[float, float]:
    """Compute azimuth range for a scan.

    Uses explicit overrides from scan_params if provided, otherwise
    estimates from the field width and elevation.

    Parameters
    ----------
    patch : ObservingPatch
        The observing patch.
    center_az : float
        Center azimuth in degrees.
    center_el : float
        Center elevation in degrees (used when patch.elevation is None).
    """
    params = patch.scan_params

    if "az_min" in params and "az_max" in params:
        return float(params["az_min"]), float(params["az_max"])

    elevation = patch.elevation if patch.elevation is not None else center_el
    el_rad = math.radians(elevation)
    cos_el = max(math.cos(el_rad), 0.1)
    half_throw = patch.width / (2.0 * cos_el)

    return center_az - half_throw, center_az + half_throw
