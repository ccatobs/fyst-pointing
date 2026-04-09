"""Trajectory utility functions.

Free functions for validating, exporting, formatting, and plotting
Trajectory objects. These are the primary API. The corresponding
methods on Trajectory delegate here.
"""

import dataclasses
import sys
import warnings
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from .exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingWarning,
)
from .site import Site
from .trajectory import SCAN_FLAG_RETUNE, SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND

if TYPE_CHECKING:
    from .coordinates import Coordinates
    from .trajectory import Trajectory


def validate_trajectory_bounds(
    site: Site,
    az: np.ndarray,
    el: np.ndarray,
) -> None:
    """Validate that all trajectory points are within telescope limits.

    Parameters
    ----------
    site : Site
        Telescope site configuration containing telescope_limits.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.

    Raises
    ------
    AzimuthBoundsError
        If any point exceeds telescope azimuth limits.
    ElevationBoundsError
        If any point exceeds telescope elevation limits.

    Examples
    --------
    >>> from fyst_trajectories import get_fyst_site
    >>> site = get_fyst_site()
    >>> az = np.array([100, 150, 200])
    >>> el = np.array([45, 50, 55])
    >>> validate_trajectory_bounds(site, az, el)  # Passes if within limits
    """
    limits = site.telescope_limits

    az_min, az_max = float(az.min()), float(az.max())
    if az_min < limits.azimuth.min or az_max > limits.azimuth.max:
        raise AzimuthBoundsError(
            actual_min=az_min,
            actual_max=az_max,
            limit_min=limits.azimuth.min,
            limit_max=limits.azimuth.max,
        )

    el_min, el_max = float(el.min()), float(el.max())
    if el_min < limits.elevation.min or el_max > limits.elevation.max:
        raise ElevationBoundsError(
            actual_min=el_min,
            actual_max=el_max,
            limit_min=limits.elevation.min,
            limit_max=limits.elevation.max,
        )


def validate_trajectory_dynamics(
    site: Site,
    az: np.ndarray,
    el: np.ndarray,
    times: np.ndarray,
) -> None:
    """Check that trajectory velocities and accelerations are within limits.

    Computes numerical derivatives of position to estimate velocity and
    acceleration, then warns if they exceed the telescope's configured
    limits.

    Issues warnings rather than raising exceptions because exceeding a
    dynamics limit does not make a trajectory unexecutable. The
    telescope will simply track slower than requested at those points.

    Parameters
    ----------
    site : Site
        Telescope site configuration containing telescope_limits with
        max_velocity and max_acceleration for each axis.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.
    times : np.ndarray
        Timestamps in seconds.

    Warns
    -----
    PointingWarning
        If velocity or acceleration exceeds configured limits, or if
        the trajectory has too few points for meaningful validation.
    """
    if len(times) < 2:
        warnings.warn(
            "Trajectory has fewer than 2 points, skipping dynamics validation.",
            PointingWarning,
            stacklevel=2,
        )
        return

    if len(times) < 4:
        warnings.warn(
            f"Trajectory has only {len(times)} points. Acceleration estimates "
            "require at least 4 points; skipping acceleration validation.",
            PointingWarning,
            stacklevel=2,
        )
        return

    limits = site.telescope_limits

    az_unwrapped = np.unwrap(az, period=360.0)
    az_vel = np.gradient(az_unwrapped, times)
    el_vel = np.gradient(el, times)

    max_az_vel = np.abs(az_vel).max()
    max_el_vel = np.abs(el_vel).max()

    if max_az_vel > limits.azimuth.max_velocity:
        warnings.warn(
            f"Trajectory azimuth velocity ({max_az_vel:.2f} deg/s) exceeds "
            f"limit ({limits.azimuth.max_velocity:.2f} deg/s).",
            PointingWarning,
            stacklevel=2,
        )

    if max_el_vel > limits.elevation.max_velocity:
        warnings.warn(
            f"Trajectory elevation velocity ({max_el_vel:.2f} deg/s) exceeds "
            f"limit ({limits.elevation.max_velocity:.2f} deg/s).",
            PointingWarning,
            stacklevel=2,
        )

    # Advisory: check if cos(el) scaling makes coordinate velocity misleading.
    # At high elevation, small on-sky motions require large az coordinate rates.
    cos_el = np.cos(np.radians(el))
    min_cos_el = cos_el.min()
    if min_cos_el > 0 and max_az_vel > 0:
        on_sky_az_vel = np.abs(az_vel) * cos_el
        max_on_sky = on_sky_az_vel.max()
        # Warn when coordinate velocity exceeds on-sky by >2x (el > ~60 deg)
        if max_az_vel > 2.0 * max_on_sky:
            warnings.warn(
                f"High elevation reduces on-sky azimuth speed to "
                f"{max_on_sky:.2f} deg/s (coordinate: {max_az_vel:.2f} deg/s, "
                f"min cos(el)={min_cos_el:.3f}). Verify scan design is appropriate.",
                PointingWarning,
                stacklevel=2,
            )

    az_accel = np.gradient(az_vel, times)
    el_accel = np.gradient(el_vel, times)

    max_az_accel = np.abs(az_accel).max()
    max_el_accel = np.abs(el_accel).max()

    if max_az_accel > limits.azimuth.max_acceleration:
        warnings.warn(
            f"Trajectory azimuth acceleration ({max_az_accel:.2f} deg/s^2) exceeds "
            f"limit ({limits.azimuth.max_acceleration:.2f} deg/s^2).",
            PointingWarning,
            stacklevel=2,
        )

    if max_el_accel > limits.elevation.max_acceleration:
        warnings.warn(
            f"Trajectory elevation acceleration ({max_el_accel:.2f} deg/s^2) exceeds "
            f"limit ({limits.elevation.max_acceleration:.2f} deg/s^2).",
            PointingWarning,
            stacklevel=2,
        )


def validate_trajectory(
    trajectory: "Trajectory",
    site: Site,
    check_sun: bool = True,
) -> None:
    """Validate trajectory against telescope limits.

    Checks position bounds (raises on violation),
    velocity/acceleration limits (warns on violation), and optionally
    sun avoidance constraints (warns on violation).

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to validate.
    site : Site
        Telescope site with axis limits.
    check_sun : bool, optional
        Whether to check sun avoidance constraints. Default True.
        Sun checking requires ``trajectory.start_time`` to be set;
        if it is None the sun check is skipped silently.

    Raises
    ------
    AzimuthBoundsError
        If azimuth positions are outside telescope movement range.
    ElevationBoundsError
        If elevation positions are outside telescope movement range.

    Warns
    -----
    PointingWarning
        If any velocities or accelerations exceed limits, or if any
        trajectory point is within the sun exclusion or warning radius.
    """
    validate_trajectory_bounds(site, trajectory.az, trajectory.el)
    validate_trajectory_dynamics(site, trajectory.az, trajectory.el, trajectory.times)
    if check_sun and trajectory.start_time is not None:
        abs_times = get_absolute_times(trajectory)
        validate_sun_avoidance(site, trajectory.az, trajectory.el, abs_times)


def get_absolute_times(trajectory: "Trajectory") -> Time:
    """Get absolute timestamps for the trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory with a start_time set.

    Returns
    -------
    Time
        Astropy Time array with absolute timestamps.

    Raises
    ------
    ValueError
        If start_time is not set.
    """
    if trajectory.start_time is None:
        raise ValueError("start_time not set; cannot compute absolute times")
    return trajectory.start_time + TimeDelta(trajectory.times * u.s)


def validate_sun_avoidance(
    site: Site,
    az: np.ndarray,
    el: np.ndarray,
    times: Time | np.ndarray,
    coords: "Coordinates | None" = None,
) -> None:
    """Check sun avoidance constraints, emitting warnings for violations.

    .. warning::

       This check is **advisory only**.  It emits Python warnings but
       never blocks trajectory generation or raises exceptions.  Telescope
       control systems **must** enforce their own hard sun-avoidance limits
       independently of this function.

    The sun moves slowly (~0.25 deg/min), so this function subsamples
    the trajectory to check approximately every 60 seconds rather than
    every point.  For each subsampled time the sun position is computed
    once and compared against all trajectory points in that time window.

    This function never blocks trajectory generation. Violations are
    reported as warnings so that downstream consumers can decide how
    to handle them.

    Parameters
    ----------
    site : Site
        Site configuration with sun avoidance settings.
    az : np.ndarray
        Azimuth array in degrees.
    el : np.ndarray
        Elevation array in degrees.
    times : Time or np.ndarray
        Absolute times for each trajectory point.
    coords : Coordinates, optional
        Pre-constructed Coordinates instance. Created internally if
        not provided.

    Warns
    -----
    PointingWarning
        If any trajectory point is within the exclusion radius
        ("EXCLUSION ZONE") or the warning radius ("WARNING ZONE")
        of the Sun.
    """
    if not site.sun_avoidance.enabled:
        return

    from .coordinates import Coordinates  # pylint: disable=import-outside-toplevel

    if coords is None:
        coords = Coordinates(site, atmosphere=None)

    n_points = len(az)
    if n_points == 0:
        return

    if isinstance(times, Time):
        total_seconds = (times[-1] - times[0]).to_value(u.s)
    else:
        total_seconds = float(times[-1] - times[0])

    subsample_interval = 60.0  # seconds
    if total_seconds <= 0:
        step = n_points
    else:
        step = max(1, int(subsample_interval * n_points / total_seconds))

    sample_indices = np.arange(0, n_points, step)
    if sample_indices[-1] != n_points - 1:
        sample_indices = np.append(sample_indices, n_points - 1)

    sample_times = times[sample_indices]
    sample_az = az[sample_indices]
    sample_el = el[sample_indices]

    sun_az, sun_alt = coords.get_sun_altaz(sample_times)
    sun_az = np.atleast_1d(sun_az)
    sun_alt = np.atleast_1d(sun_alt)

    az1 = np.deg2rad(sample_az)
    el1 = np.deg2rad(sample_el)
    az2 = np.deg2rad(sun_az)
    el2 = np.deg2rad(sun_alt)

    daz = az1 - az2
    cos_el2 = np.cos(el2)
    sin_el2 = np.sin(el2)
    cos_el1 = np.cos(el1)
    sin_el1 = np.sin(el1)

    num = np.sqrt(
        (cos_el2 * np.sin(daz)) ** 2 + (cos_el1 * sin_el2 - sin_el1 * cos_el2 * np.cos(daz)) ** 2
    )
    den = sin_el1 * sin_el2 + cos_el1 * cos_el2 * np.cos(daz)
    separations = np.rad2deg(np.arctan2(num, den))

    min_idx = int(np.argmin(separations))
    min_sep = float(separations[min_idx])

    exclusion = site.sun_avoidance.exclusion_radius
    warning = site.sun_avoidance.warning_radius

    if isinstance(sample_times, Time):
        closest_time_str = sample_times[min_idx].iso
    else:
        closest_time_str = str(sample_times[min_idx])

    if min_sep < exclusion:
        warnings.warn(
            f"EXCLUSION ZONE: Trajectory passes {min_sep:.1f}\u00b0 from the Sun "
            f"(exclusion radius: {exclusion}\u00b0) at {closest_time_str}. "
            f"The telescope hardware may refuse this trajectory.",
            PointingWarning,
            stacklevel=2,
        )
    elif min_sep < warning:
        warnings.warn(
            f"WARNING ZONE: Trajectory passes {min_sep:.1f}\u00b0 from the Sun "
            f"(warning radius: {warning}\u00b0) at {closest_time_str}.",
            PointingWarning,
            stacklevel=2,
        )


def to_arrays(
    trajectory: "Trajectory",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export trajectory as simple arrays for ACU upload.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to export.

    Returns
    -------
    times : np.ndarray
        Timestamps in seconds from start.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.
    """
    return trajectory.times.copy(), trajectory.az.copy(), trajectory.el.copy()


def to_path_format(trajectory: "Trajectory") -> list[list[float]]:
    """Convert trajectory to list format for /path endpoint.

    Converts the trajectory arrays into the format expected by the OCS
    /path endpoint: a list of [time, az, el, az_vel, el_vel] points.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to convert.

    Returns
    -------
    list
        List of [time, az, el, az_vel, el_vel] points.

    Examples
    --------
    >>> points = to_path_format(trajectory)
    >>> data = {"start_time": trajectory.start_time.unix, "points": points}
    """
    return np.column_stack(
        [
            trajectory.times,
            trajectory.az,
            trajectory.el,
            trajectory.az_vel,
            trajectory.el_vel,
        ]
    ).tolist()


def plot_trajectory(trajectory: "Trajectory", show: bool) -> Any:
    """Plot trajectory az/el vs time and sky track.

    Creates a 3-panel figure showing azimuth vs time, elevation vs time,
    and azimuth vs elevation (sky track).

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to plot.
    show : bool
        Whether to call plt.show() after creating the figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_trajectory(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(trajectory.times, trajectory.az)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Azimuth (deg)")
    axes[0].set_title("Az vs Time")

    axes[1].plot(trajectory.times, trajectory.el)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Elevation (deg)")
    axes[1].set_title("El vs Time")

    axes[2].plot(trajectory.az, trajectory.el)
    axes[2].set_xlabel("Azimuth (deg)")
    axes[2].set_ylabel("Elevation (deg)")
    axes[2].set_title("Sky Track")
    axes[2].set_aspect("equal")

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def inject_retune(
    trajectory: "Trajectory",
    retune_interval: float = 300.0,
    retune_duration: float = 5.0,
    prefer_turnarounds: bool = False,
    turnaround_window: float = 5.0,
    module_index: int = 0,
    n_modules: int = 1,
) -> "Trajectory":
    """Inject retune flags into a trajectory at regular intervals.

    Walks forward through the trajectory timeline and places retune events
    every ``retune_interval`` seconds. If ``prefer_turnarounds`` is True and
    a turnaround region exists within ``turnaround_window`` seconds of the
    due time, the retune is snapped to start at the turnaround (zero
    additional dead time). Otherwise the retune is placed at the time-based
    position.

    The default is ``prefer_turnarounds=False`` (time-based placement),
    which produces uniform coverage. Set to True to snap retunes to nearby
    turnarounds, which saves ~0.04% science time but concentrates gaps at
    turnaround positions, creating persistent coverage non-uniformity.

    Only samples with ``SCAN_FLAG_SCIENCE`` are overwritten with
    ``SCAN_FLAG_RETUNE``; turnaround flags are never modified.

    **Per-module staggered retune** (UNCONFIRMED -- needs FYST team
    verification): Prime-Cam has 7 independent readout modules. If modules
    can retune independently, setting ``n_modules > 1`` offsets the first
    retune by ``module_index * retune_interval / n_modules``, so only one
    module is retuning at any given time. This reduces effective overhead
    from ~16% to ~2.4% for 7 modules. Set ``n_modules=1`` (the default)
    to disable staggering and retune all modules simultaneously.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory with scan_flag array.
    retune_interval : float
        Seconds between retune events (from last retune or start).
        Default 300s (5 min) is preliminary, based on NIKA2 heritage
        where KIDs retune at sub-scan boundaries. Optimal value depends
        on Prime-Cam detector stability under varying atmospheric load;
        validate with commissioning data.
    retune_duration : float
        Duration in seconds of each retune event.
    prefer_turnarounds : bool
        If True, snap retunes to nearby turnarounds when possible.
        Default is False (time-based placement for uniform coverage).
    turnaround_window : float
        Maximum seconds from due time to search for a turnaround start.
    module_index : int
        Index of this module (0-based) for staggered retune scheduling.
        Default is 0. Only meaningful when ``n_modules > 1``.
    n_modules : int
        Total number of independent modules. Default is 1 (no staggering,
        all modules retune simultaneously -- current behavior). Set to 7
        for Prime-Cam staggered retune.

    Returns
    -------
    Trajectory
        New trajectory with retune samples flagged.

    Raises
    ------
    ValueError
        If ``module_index`` is negative or >= ``n_modules``, or if
        ``n_modules`` is less than 1.
    """
    if retune_interval <= 0:
        raise ValueError(f"retune_interval must be positive, got {retune_interval}")
    if retune_duration <= 0:
        raise ValueError(f"retune_duration must be positive, got {retune_duration}")
    if n_modules < 1:
        raise ValueError(f"n_modules must be >= 1, got {n_modules}")
    if module_index < 0 or module_index >= n_modules:
        raise ValueError(f"module_index must be in [0, {n_modules}), got {module_index}")

    # N-6 defensive guard: turnaround detection relies on real velocities
    # (see inject_retune body below, which classifies turnarounds via
    # ``SCAN_FLAG_TURNAROUND`` samples derived from the trajectory's
    # velocity profile). A trajectory with identically zero az/el
    # velocities -- which the primecam wrapper currently supplies -- has no
    # detectable turnarounds, so snapping would silently collapse to
    # time-based placement anyway. Warn and fall back explicitly so the
    # caller is not misled.
    if prefer_turnarounds:
        if (
            trajectory.az_vel is not None
            and trajectory.el_vel is not None
            and np.all(trajectory.az_vel == 0.0)
            and np.all(trajectory.el_vel == 0.0)
        ):
            warnings.warn(
                "inject_retune called with prefer_turnarounds=True but the "
                "trajectory has all-zero velocities; turnaround detection "
                "requires real velocities. Falling back to time-based retune "
                "placement.",
                PointingWarning,
                stacklevel=2,
            )
            prefer_turnarounds = False

    times = trajectory.times

    if trajectory.scan_flag is None:
        scan_flag = np.full(len(times), SCAN_FLAG_SCIENCE, dtype=int)
    else:
        scan_flag = trajectory.scan_flag.copy()

    duration = float(times[-1] - times[0])
    if duration < retune_interval:
        return dataclasses.replace(trajectory, scan_flag=scan_flag)

    turnaround_starts: list[float] = []
    if prefer_turnarounds:
        is_turnaround = scan_flag == SCAN_FLAG_TURNAROUND
        for i in range(len(is_turnaround)):
            if is_turnaround[i] and (i == 0 or not is_turnaround[i - 1]):
                turnaround_starts.append(float(times[i]))

    # For staggered retune, offset the first retune time by a fraction
    # of the retune interval so different modules retune at different times.
    stagger_offset = module_index * retune_interval / n_modules
    last_retune_time = float(times[0]) + stagger_offset

    while True:
        due_time = last_retune_time + retune_interval
        if due_time > float(times[-1]):
            break

        retune_start = due_time
        if prefer_turnarounds and turnaround_starts:
            best_ta = None
            best_dist = turnaround_window + 1.0
            for ta_start in turnaround_starts:
                dist = abs(ta_start - due_time)
                if dist <= turnaround_window and dist < best_dist:
                    best_ta = ta_start
                    best_dist = dist
            if best_ta is not None:
                retune_start = best_ta

        retune_end = retune_start + retune_duration

        mask = (times >= retune_start) & (times < retune_end) & (scan_flag == SCAN_FLAG_SCIENCE)
        scan_flag[mask] = SCAN_FLAG_RETUNE

        # Use max(retune_end, due_time) to prevent backward drift when
        # prefer_turnarounds snaps to a turnaround before the due time.
        last_retune_time = max(retune_end, due_time)

    return dataclasses.replace(trajectory, scan_flag=scan_flag)


def _format_trajectory(
    trajectory: "Trajectory",
    head: int | None = 5,
    tail: int | None = 5,
) -> str:
    """Format trajectory as a table string."""
    lines = [repr(trajectory), ""]
    n = trajectory.n_points
    head_n = min(head or 0, n)
    tail_n = min(tail or 0, n)

    if (head_n + tail_n) >= n:
        indices: list[int | None] = list(range(n))
    else:
        indices = list(range(head_n))
        if head_n > 0 and tail_n > 0:
            indices.append(None)
        indices.extend(range(n - tail_n, n))

    has_abs = trajectory.start_time is not None
    abs_times = get_absolute_times(trajectory) if has_abs else None

    if has_abs:
        hdr = f"{'t (s)':>8}  {'UTC':^23}  {'az':>10}  {'el':>10}  {'az_vel':>10}  {'el_vel':>10}"
    else:
        hdr = f"{'t (s)':>8}  {'az':>10}  {'el':>10}  {'az_vel':>10}  {'el_vel':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for i in indices:
        if i is None:
            lines.append(
                "..."
                if not has_abs
                else f"{'...':>8}  {'':^23}  {'...':>10}  {'...':>10}  {'...':>10}  {'...':>10}"
            )
        else:
            row = f"{trajectory.times[i]:8.2f}  "
            if has_abs:
                row += f"{abs_times[i].iso[:23]:^23}  "
            row += f"{trajectory.az[i]:10.4f}  {trajectory.el[i]:10.4f}  "
            row += f"{trajectory.az_vel[i]:10.4f}  {trajectory.el_vel[i]:10.4f}"
            lines.append(row)

    return "\n".join(lines)


def print_trajectory(
    trajectory: "Trajectory",
    head: int | None = 5,
    tail: int | None = 5,
    file: TextIO | None = None,
) -> None:
    """Print a formatted table of trajectory points.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to print.
    head : int or None, optional
        Number of points from the beginning. Default is 5.
    tail : int or None, optional
        Number of points from the end. Default is 5.
    file : TextIO or None, optional
        Output stream. Default is sys.stdout.

    Examples
    --------
    >>> from fyst_trajectories import print_trajectory
    >>> print_trajectory(trajectory)

    Print only the first 10 points:

    >>> print_trajectory(trajectory, head=10, tail=None)
    """
    print(_format_trajectory(trajectory, head=head, tail=tail), file=file or sys.stdout)
