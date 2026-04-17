"""Public :func:`generate_timeline` entry point."""

from astropy.time import Time

from ..site import Site
from .constraints import Constraint
from .models import (
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
)
from .scheduler import Scheduler, SchedulerContext

__all__ = [
    "generate_timeline",
]


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

    ctx = SchedulerContext.build(
        patches=patches,
        site=site,
        start_time=start_time,
        end_time=end_time,
        overhead_model=overhead_model,
        calibration_policy=calibration_policy,
        constraints=constraints,
        time_step=time_step,
    )
    return Scheduler(ctx).run()
