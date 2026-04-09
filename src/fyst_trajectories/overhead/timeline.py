"""Observation timeline generator.

Sequences science scans with calibration breaks and overhead,
inspired by TOAST's ``toast_ground_schedule.py`` and SO schedlib.

At each time step the generator selects the highest-scoring observable
patch, schedules a science scan on it, and inserts calibration operations
whose cadence has elapsed.
"""

import logging
import math

from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..site import Site
from .constraints import Constraint, ElevationConstraint, SunAvoidanceConstraint
from .io import _nasmyth_rotation
from .models import (
    BlockType,
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)
from .overhead import CalibrationState
from .utils import estimate_slew_time, get_observable_windows

__all__ = [
    "generate_timeline",
]

logger = logging.getLogger(__name__)


def generate_timeline(
    patches: list[ObservingPatch],
    site: Site,
    start_time: Time | str,
    end_time: Time | str,
    overhead_model: OverheadModel | None = None,
    calibration_policy: CalibrationPolicy | None = None,
    constraints: list[Constraint] | None = None,
    time_step: float = 300.0,
) -> ObservingTimeline:
    """Generate an observing timeline.

    At each time step, evaluates all patches, selects the highest-scoring
    one, schedules a science scan, and advances. Calibration operations
    are injected between scans when cadence thresholds are exceeded.

    Parameters
    ----------
    patches : list of ObservingPatch
        Sky regions to observe.
    site : Site
        Observatory site configuration.
    start_time : Time or str
        Timeline start time (UTC). Strings are auto-parsed.
    end_time : Time or str
        Timeline end time (UTC).
    overhead_model : OverheadModel or None
        Overhead timing parameters. Uses defaults if None.
    calibration_policy : CalibrationPolicy or None
        Calibration cadence policy. Uses defaults if None.
    constraints : list of Constraint or None
        Scheduling constraints. If None, uses default elevation + sun
        avoidance constraints from the site configuration.
    time_step : float
        Retry interval in seconds when no target is available.

    Returns
    -------
    ObservingTimeline
        Complete observing timeline with science, calibration,
        slew, and idle blocks.
    """
    if isinstance(start_time, str):
        start_time = Time(start_time, scale="utc")
    if isinstance(end_time, str):
        end_time = Time(end_time, scale="utc")

    if overhead_model is None:
        overhead_model = OverheadModel()
    if calibration_policy is None:
        calibration_policy = CalibrationPolicy()
    if constraints is None:
        constraints = _default_constraints(site)

    coords = Coordinates(site)
    cal_state = CalibrationState()
    blocks: list[TimelineBlock] = []
    scan_counter = 0
    current_time = start_time
    current_az = 180.0  # Assume starting near south
    # Initial elevation estimate for first slew time computation;
    # replaced by per-patch values once observations begin.
    current_el = 50.0

    while current_time.unix < end_time.unix:
        remaining = (end_time - current_time).sec
        if remaining < overhead_model.min_scan_duration:
            break

        needed_cals = cal_state.needs_calibration(
            current_time, calibration_policy, overhead_model, coords=coords
        )
        for cal_spec in needed_cals:
            if current_time.unix >= end_time.unix:
                break
            cal_duration = min(cal_spec.duration, (end_time - current_time).sec)
            cal_block = TimelineBlock(
                t_start=current_time,
                t_stop=current_time + TimeDelta(cal_duration, format="sec"),
                block_type=BlockType.CALIBRATION,
                patch_name=str(cal_spec.name),
                az_min=current_az,
                az_max=current_az,
                elevation=current_el,
                scan_index=scan_counter,
                scan_type=str(cal_spec.name),
                boresight_angle=_nasmyth_rotation(current_az, current_el, site),
                metadata={"cal_type": str(cal_spec.name), "target": cal_spec.target},
            )
            blocks.append(cal_block)
            cal_state = cal_state.update(cal_spec.name, current_time)
            current_time = current_time + TimeDelta(cal_duration, format="sec")

        if current_time.unix >= end_time.unix:
            break

        best_patch = None
        best_score = 0.0
        best_az = 0.0
        best_el = 0.0

        for patch in patches:
            az, el = coords.radec_to_altaz(patch.ra_center, patch.dec_center, current_time)
            check_el = patch.elevation if patch.elevation is not None else el
            score = _evaluate_patch(patch, current_time, az, check_el, coords, constraints)
            score *= patch.weight / patch.priority

            if score > best_score:
                best_score = score
                best_patch = patch
                best_az = az
                best_el = el

        if best_patch is None or best_score == 0.0:
            advance = min(time_step, (end_time - current_time).sec)
            idle_block = TimelineBlock(
                t_start=current_time,
                t_stop=current_time + TimeDelta(advance, format="sec"),
                block_type=BlockType.IDLE,
                patch_name="no_target",
                az_min=current_az,
                az_max=current_az,
                elevation=current_el,
                scan_index=scan_counter,
                scan_type="idle",
                boresight_angle=_nasmyth_rotation(current_az, current_el, site),
            )
            blocks.append(idle_block)
            current_time = current_time + TimeDelta(advance, format="sec")
            continue

        slew_time = estimate_slew_time(current_az, current_el, best_az, best_el, site)
        slew_time += overhead_model.settle_time

        if slew_time > 1.0:
            slew_end = current_time + TimeDelta(slew_time, format="sec")
            if slew_end.unix >= end_time.unix:
                break
            slew_block = TimelineBlock(
                t_start=current_time,
                t_stop=slew_end,
                block_type=BlockType.SLEW,
                patch_name=f"slew_to_{best_patch.name}",
                az_min=current_az,
                az_max=best_az,
                elevation=best_el,
                scan_index=scan_counter,
                scan_type="slew",
                boresight_angle=_nasmyth_rotation(0.5 * (current_az + best_az), best_el, site),
            )
            blocks.append(slew_block)
            current_time = slew_end

        scan_duration = _compute_scan_duration(
            best_patch, current_time, end_time, site, coords, overhead_model, best_el
        )

        if scan_duration < overhead_model.min_scan_duration:
            advance = min(time_step, (end_time - current_time).sec)
            current_time = current_time + TimeDelta(advance, format="sec")
            continue

        n_subscans = max(1, math.ceil(scan_duration / overhead_model.max_scan_duration))
        subscan_duration = scan_duration / n_subscans

        ha = coords.get_hour_angle(best_patch.ra_center, current_time)
        rising = ha < 0.0

        az_min, az_max = _compute_az_range(best_patch, best_az, best_el)

        for sub_idx in range(n_subscans):
            if current_time.unix >= end_time.unix:
                break

            actual_duration = min(subscan_duration, (end_time - current_time).sec)
            if actual_duration < overhead_model.min_scan_duration:
                break

            sci_el = best_el if best_patch.elevation is None else best_patch.elevation
            science_block = TimelineBlock(
                t_start=current_time,
                t_stop=current_time + TimeDelta(actual_duration, format="sec"),
                block_type=BlockType.SCIENCE,
                patch_name=best_patch.name,
                az_min=az_min,
                az_max=az_max,
                elevation=sci_el,
                scan_index=scan_counter,
                subscan_index=sub_idx,
                rising=rising,
                scan_type=best_patch.scan_type,
                boresight_angle=_nasmyth_rotation(0.5 * (az_min + az_max), sci_el, site),
                metadata={
                    "velocity": best_patch.velocity,
                    "scan_params": best_patch.scan_params,
                    "ra_center": best_patch.ra_center,
                    "dec_center": best_patch.dec_center,
                    "width": best_patch.width,
                    "height": best_patch.height,
                },
            )
            blocks.append(science_block)
            current_time = current_time + TimeDelta(actual_duration, format="sec")

            if sub_idx < n_subscans - 1:
                retune_needed = cal_state.needs_calibration(
                    current_time, calibration_policy, overhead_model, coords=coords
                )
                retune_specs = [c for c in retune_needed if c.name == "retune"]
                if retune_specs:
                    retune_dur = retune_specs[0].duration
                    retune_block = TimelineBlock(
                        t_start=current_time,
                        t_stop=current_time + TimeDelta(retune_dur, format="sec"),
                        block_type=BlockType.CALIBRATION,
                        patch_name="retune",
                        az_min=az_min,
                        az_max=az_max,
                        elevation=best_el,
                        scan_index=scan_counter,
                        scan_type="retune",
                        boresight_angle=_nasmyth_rotation(0.5 * (az_min + az_max), best_el, site),
                    )
                    blocks.append(retune_block)
                    cal_state = cal_state.update("retune", current_time)
                    current_time = current_time + TimeDelta(retune_dur, format="sec")

        current_az = az_max
        current_el = best_el if best_patch.elevation is None else best_patch.elevation
        scan_counter += 1

    return ObservingTimeline(
        blocks=blocks,
        site=site,
        start_time=start_time,
        end_time=end_time,
        overhead_model=overhead_model,
        calibration_policy=calibration_policy,
        metadata={
            "n_patches": len(patches),
            "time_step": time_step,
        },
    )


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
    the field crosses the elevation. For pong/daisy, we use the max scan
    duration or remaining schedule time, whichever is shorter.

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
        return min(overhead.max_scan_duration, remaining)


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
