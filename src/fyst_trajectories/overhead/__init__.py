"""Overhead modeling and observing timeline generation for FYST/Prime-Cam.

This subpackage provides overhead budget modeling, calibration cadence tracking,
and timeline generation for the FYST telescope. It builds on fyst-trajectories
for coordinate transforms, site configuration, and trajectory generation.

The key entry point is :func:`generate_timeline`, which takes a list of
observing patches and produces a complete timeline with calibration injection.

Examples
--------
Generate a one-night timeline:

>>> from fyst_trajectories import get_fyst_site
>>> from fyst_trajectories.overhead import (
...     ObservingPatch,
...     generate_timeline,
...     write_timeline,
...     compute_budget,
... )
>>> site = get_fyst_site()
>>> patches = [
...     ObservingPatch(
...         name="Deep56",
...         ra_center=24.0,
...         dec_center=-32.0,
...         width=40.0,
...         height=10.0,
...         scan_type="constant_el",
...         velocity=1.0,
...         elevation=50.0,
...     ),
... ]
>>> timeline = generate_timeline(
...     patches=patches,
...     site=site,
...     start_time="2026-06-15T00:00:00",
...     end_time="2026-06-15T12:00:00",
... )
>>> stats = compute_budget(timeline)
>>> print(f"Efficiency: {stats['efficiency']:.1%}")
>>> write_timeline(timeline, "timeline.ecsv")
"""

from .constraints import (
    Constraint,
    ElevationConstraint,
    MinDurationConstraint,
    MoonAvoidanceConstraint,
    SunAvoidanceConstraint,
)
from .io import read_timeline, write_timeline
from .models import (
    BlockType,
    CalibrationPolicy,
    CalibrationSpec,
    CalibrationType,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)
from .overhead import CalibrationState
from .simulation import (
    accumulate_hitmaps,
    compute_budget,
    schedule_to_trajectories,
)
from .timeline import generate_timeline
from .utils import (
    estimate_slew_time,
    get_max_elevation,
    get_observable_windows,
    get_transit_time,
)

__all__ = [
    "BlockType",
    "CalibrationPolicy",
    "CalibrationState",
    "CalibrationSpec",
    "CalibrationType",
    "Constraint",
    "ElevationConstraint",
    "MinDurationConstraint",
    "MoonAvoidanceConstraint",
    "ObservingPatch",
    "ObservingTimeline",
    "OverheadModel",
    "SunAvoidanceConstraint",
    "TimelineBlock",
    "accumulate_hitmaps",
    "compute_budget",
    "estimate_slew_time",
    "generate_timeline",
    "get_max_elevation",
    "get_observable_windows",
    "get_transit_time",
    "read_timeline",
    "schedule_to_trajectories",
    "write_timeline",
]
