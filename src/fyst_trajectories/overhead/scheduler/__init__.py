"""Internal scheduler subpackage for ``overhead.timeline``.

Public API: :func:`fyst_trajectories.overhead.generate_timeline` stays
the sole entry point for downstream consumers. The classes in this
subpackage (:class:`Scheduler`, phase classes, state dataclasses) are
exposed for advanced users who want to extend scheduling behavior
(priority-weighted scheduling, multi-night stitching, lookahead).
"""

from .helpers import (
    _compute_az_range,
    _compute_scan_duration,
    _default_constraints,
    _evaluate_patch,
    _time_until_set,
)
from .phases import (
    CalibrationPhase,
    PatchSelectionPhase,
    Phase,
    PhaseResult,
    ScienceScanPhase,
    SlewPhase,
)
from .scheduler import Scheduler
from .state import SchedulerContext, SchedulerState

__all__ = [
    "CalibrationPhase",
    "PatchSelectionPhase",
    "Phase",
    "PhaseResult",
    "ScienceScanPhase",
    "Scheduler",
    "SchedulerContext",
    "SchedulerState",
    "SlewPhase",
    "_compute_az_range",
    "_compute_scan_duration",
    "_default_constraints",
    "_evaluate_patch",
    "_time_until_set",
]
