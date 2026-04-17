"""Field-geometry helpers used by the constant-elevation scan planner.

Public to the package-internal planner; not part of the
:mod:`fyst_trajectories.planning` public API.
"""

import math
import warnings

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..exceptions import PointingWarning
from ._types import FieldRegion


def _field_region_corners(
    ra_center: float,
    dec_center: float,
    width: float,
    height: float,
    angle_deg: float,
) -> list[tuple[float, float]]:
    """Compute RA/Dec corners of a rotated rectangular field region.

    Uses a flat-sky approximation to rotate corners around the field center.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the field center in degrees.
    dec_center : float
        Declination of the field center in degrees.
    width : float
        RA extent of the field in degrees (before rotation).
    height : float
        Dec extent of the field in degrees (before rotation).
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    list of (ra, dec) tuples
        The four corners of the rotated rectangle.
    """
    hw, hh = width / 2.0, height / 2.0
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cos_dec = math.cos(math.radians(dec_center))
    # The flat-sky rotation that follows is only a sensible approximation
    # while cos(dec) is non-trivial. ``cos(dec) < 0.01`` corresponds to
    # ``|dec| > 89.43°``; beyond that the cylindrical projection breaks
    # down well before the cos_dec=0 singularity. FYST's lat=-23° puts
    # such declinations below the elevation cut anyway, so the check is
    # a defensive boundary, not an operationally tight one.
    if abs(cos_dec) < 0.01:
        raise ValueError("FieldRegion too close to celestial pole (|dec| > 89.43°)")
    corners = []
    for dx, dy in corners_local:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        corners.append((ra_center + rx / cos_dec, dec_center + ry))
    return corners


def _find_elevation_crossing(
    el_array: np.ndarray,
    search_times: Time,
    target_el: float,
    rising: bool,
    step_seconds: float,
) -> Time | None:
    """Find the first rising or setting crossing of a target elevation.

    Parameters
    ----------
    el_array : ndarray
        Elevation values at each search time.
    search_times : Time
        Array of search times.
    target_el : float
        Target elevation in degrees.
    rising : bool
        If True, find the rising crossing; if False, the setting crossing.
    step_seconds : float
        Time step between search times in seconds.

    Returns
    -------
    Time or None
        Time of crossing, or None if no crossing found.
    """
    above = el_array >= target_el
    diff = np.diff(above.astype(int))
    if rising:
        crossings = np.where(diff == 1)[0]
    else:
        crossings = np.where(diff == -1)[0]
    if len(crossings) == 0:
        return None
    idx = crossings[0]
    denom = el_array[idx + 1] - el_array[idx]
    frac = 0.5 if abs(denom) < 1e-12 else (target_el - el_array[idx]) / denom
    return search_times[idx] + TimeDelta(frac * step_seconds * u.s)


def _compute_ce_duration(
    field: FieldRegion,
    angle: float,
    elevation: float,
    coords_obj: Coordinates,
    base_search_time: Time,
    rising: bool,
    max_search_hours: float = 12.0,
    step_seconds: float = 30.0,
) -> tuple[Time, Time, float]:
    """Compute when RA edges of a field cross the target elevation.

    Searches forward from ``base_search_time`` to find when the leading
    and trailing RA edges (at the field's central Dec) pass through the
    target elevation.

    Handles the RA = 0/360 wrap by detecting when the corner span exceeds
    180° and re-centering the values around ``field.ra_center`` so the
    leading and trailing RA edges are correctly identified for fields
    near RA = 0.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification.
    angle : float
        Field rotation angle in degrees.
    elevation : float
        Target elevation in degrees.
    coords_obj : Coordinates
        Coordinate transformer for the site.
    base_search_time : Time
        Start of the time search window.
    rising : bool
        If True, find the rising crossing; if False, the setting crossing.
    max_search_hours : float, optional
        Maximum time to search forward in hours. Default is 12.0.
    step_seconds : float, optional
        Time step for the search in seconds. Default is 30.0.

    Returns
    -------
    start_time : Time
        When the first RA edge crosses the target elevation.
    end_time : Time
        When the last RA edge crosses the target elevation.
    duration_seconds : float
        Duration in seconds.

    Raises
    ------
    ValueError
        If elevation crossings cannot be found in the search window.
    """
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )
    ra_vals = [c[0] % 360.0 for c in corners]

    # Detect RA wrap-around: if the naive span exceeds 180 degrees, the
    # field straddles the RA=0/360 boundary.
    naive_span = max(ra_vals) - min(ra_vals)
    if naive_span > 180.0:
        # Shift values that are below the center into [center-180, center+180]
        ra_center_mod = field.ra_center % 360.0
        ra_vals = [((r - ra_center_mod + 180.0) % 360.0 - 180.0 + ra_center_mod) for r in ra_vals]
    ra_min = min(ra_vals)
    ra_max = max(ra_vals)

    dt_sec = np.arange(0, max_search_hours * 3600, step_seconds)
    search_times = base_search_time + TimeDelta(dt_sec * u.s)

    _, el_min_arr = coords_obj.radec_to_altaz(
        np.full(len(search_times), ra_min),
        np.full(len(search_times), field.dec_center),
        search_times,
    )
    _, el_max_arr = coords_obj.radec_to_altaz(
        np.full(len(search_times), ra_max),
        np.full(len(search_times), field.dec_center),
        search_times,
    )

    t_start = _find_elevation_crossing(el_min_arr, search_times, elevation, rising, step_seconds)
    t_end = _find_elevation_crossing(el_max_arr, search_times, elevation, rising, step_seconds)

    if t_start is None or t_end is None:
        raise ValueError(
            f"Could not find elevation crossing for field edges at el={elevation} "
            f"(rising={rising}) within {max_search_hours} hours of {base_search_time.iso}"
        )

    if t_start > t_end:
        t_start, t_end = t_end, t_start

    duration_seconds = (t_end - t_start).to_value(u.s)

    if duration_seconds > max_search_hours * 3600 * 0.5:
        warnings.warn(
            f"Computed observation duration {duration_seconds / 3600:.1f}h is unusually long. "
            f"Check field geometry and search parameters.",
            PointingWarning,
            stacklevel=2,
        )

    return t_start, t_end, duration_seconds


def _compute_ce_az_range(
    field: FieldRegion,
    angle: float,
    coords_obj: Coordinates,
    obs_start: Time,
    obs_end: Time,
    padding: float,
) -> tuple[float, float]:
    """Compute azimuth range needed to cover a field at given elevation.

    Evaluates the azimuth of all four rotated corners and the field center
    at three times (start, midpoint, end) and returns the encompassing range
    with padding. Using three times captures the temporal variation in
    azimuth coverage as the field transits.

    Handles the azimuth = 0/360 discontinuity for sources transiting through
    north (plausible at FYST's −23° latitude for sources with dec ≳ +20°):
    when the naive max−min span exceeds 180°, the samples are unwrapped
    around the median azimuth so the returned range is contiguous.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification.
    angle : float
        Field rotation angle in degrees.
    coords_obj : Coordinates
        Coordinate transformer for the site.
    obs_start : Time
        Start time of the observation.
    obs_end : Time
        End time of the observation.
    padding : float
        Extra padding in degrees on each side.

    Returns
    -------
    az_min, az_max : float
        Azimuth range in degrees. May lie outside ``[0, 360)`` when the
        field straddles north (e.g. ``(-5.0, 12.0)`` rather than
        ``(355.0, 12.0)``); callers and consumers handle the unwrapped
        representation directly.
    """
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )

    obs_mid = obs_start + (obs_end - obs_start) / 2.0
    eval_times = [obs_start, obs_mid, obs_end]

    all_azimuths: list[float] = []
    points = list(corners) + [(field.ra_center, field.dec_center)]
    for t in eval_times:
        for ra_c, dec_c in points:
            az_c, _ = coords_obj.radec_to_altaz(ra_c, dec_c, t)
            all_azimuths.append(az_c)

    az_arr = np.asarray(all_azimuths)
    # Unwrap if the samples straddle the 0/360 discontinuity. Re-centre
    # around the median so the result is a single contiguous interval.
    if az_arr.max() - az_arr.min() > 180.0:
        median = float(np.median(az_arr))
        az_arr = ((az_arr - median + 180.0) % 360.0) - 180.0 + median

    return float(az_arr.min()) - padding, float(az_arr.max()) + padding
