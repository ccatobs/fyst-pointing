"""Dataclasses and typed schemas for planning.

Contains:

* :class:`FieldRegion` and :class:`ScanBlock` — the public data containers
  returned by the planner functions (``plan_pong_scan``,
  ``plan_constant_el_scan``, ``plan_daisy_scan``).
* :class:`PongComputedParams`, :class:`ConstantElComputedParams`,
  :class:`DaisyComputedParams` — schemas that describe the shape of
  :attr:`ScanBlock.computed_params` returned by each planner.

The dataclasses and schemas are re-exported from
:mod:`fyst_trajectories.planning` and :mod:`fyst_trajectories`.
"""

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from ..exceptions import PointingWarning
from ..patterns.configs import ScanConfig
from ..trajectory import Trajectory


class PongComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_pong_scan`.

    Attributes
    ----------
    period : float
        Pattern period in seconds for one full Pong cycle.
    x_numvert : int
        Number of vertices along the x-axis of the Lissajous lattice.
    y_numvert : int
        Number of vertices along the y-axis of the Lissajous lattice.
    n_cycles : int
        Number of full pattern cycles in the planned observation.
    """

    period: float
    x_numvert: int
    y_numvert: int
    n_cycles: int


class ConstantElComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_constant_el_scan`.

    Attributes
    ----------
    az_start : float
        Lower azimuth bound of the scan in degrees.
    az_stop : float
        Upper azimuth bound of the scan in degrees.
    az_throw : float
        Total azimuth throw (``az_stop - az_start``) in degrees.
    n_scans : int
        Number of azimuth sweeps (legs) in the scan.
    start_time_iso : str
        ISO-format UTC start time of the observation.
    end_time_iso : str
        ISO-format UTC end time of the observation.
    duration : float
        Total observation duration in seconds.
    """

    az_start: float
    az_stop: float
    az_throw: float
    n_scans: int
    start_time_iso: str
    end_time_iso: str
    duration: float


class DaisyComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_daisy_scan`.

    Attributes
    ----------
    duration : float
        Observation duration in seconds.
    """

    duration: float


# Umbrella alias used by :attr:`ScanBlock.computed_params`. The concrete
# dict shape depends on which ``plan_*`` function produced the block.
ComputedParams = PongComputedParams | ConstantElComputedParams | DaisyComputedParams


@dataclass(frozen=True)
class FieldRegion:
    """Astronomer's specification of a rectangular field on the sky.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the field center in degrees.
    dec_center : float
        Declination of the field center in degrees.
    width : float
        Angular width of the field in degrees (cross-scan direction).
        This is the physical angular extent, not the RA span. The
        cos(dec) projection is applied internally when computing
        RA boundaries. Must be positive.
    height : float
        Angular height of the field in degrees (Dec extent). Must be
        positive.

    Raises
    ------
    ValueError
        If width or height is not positive.

    Examples
    --------
    >>> field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    """

    ra_center: float
    dec_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    @property
    def dec_min(self) -> float:
        """Minimum declination of the field in degrees."""
        return self.dec_center - self.height / 2.0

    @property
    def dec_max(self) -> float:
        """Maximum declination of the field in degrees."""
        return self.dec_center + self.height / 2.0


@dataclass(frozen=True)
class ScanBlock:
    """Complete observation specification produced by a planning function.

    Contains the generated trajectory, the pattern configuration used, and
    computed parameters that help the astronomer understand the observation.

    Parameters
    ----------
    trajectory : Trajectory
        The generated trajectory ready for telescope upload. Treat this
        as read-only after planning; downstream code should not mutate
        its arrays or metadata.
    config : ScanConfig
        The pattern configuration used to generate the trajectory.
    duration : float
        Observation duration in seconds.
    computed_params : ComputedParams
        A dict of computed parameters whose shape depends on the planner
        that produced the block: :class:`PongComputedParams`,
        :class:`ConstantElComputedParams`, or :class:`DaisyComputedParams`.
    summary : str
        Human-readable summary of the planned observation.

    Examples
    --------
    >>> block = plan_pong_scan(...)
    >>> print(block.summary)
    >>> print(f"Duration: {block.duration:.1f}s")
    >>> print(f"Points: {block.trajectory.n_points}")
    """

    trajectory: Trajectory
    config: ScanConfig
    duration: float
    # Runtime is a plain ``dict``; the TypedDict union is advisory for
    # static checkers. mypy can't match ``dict`` to any union member.
    computed_params: ComputedParams = dataclasses.field(default_factory=dict)  # type: ignore[assignment]
    summary: str = ""


# Expected keys per scan type, derived from each TypedDict's
# ``__required_keys__`` so the table cannot drift from the declared
# schemas (each TypedDict is ``total=True`` with no ``NotRequired``).
_SCAN_TYPE_TO_KEYS: dict[str, frozenset[str]] = {
    "pong": PongComputedParams.__required_keys__,
    "constant_el": ConstantElComputedParams.__required_keys__,
    "daisy": DaisyComputedParams.__required_keys__,
}


def validate_computed_params(params: Mapping[str, object], scan_type: str) -> None:
    """Validate the shape of a ``computed_params`` dict at runtime.

    Checks that the dict contains the keys expected for the given
    scan type. Missing required keys raise :class:`KeyError`;
    unexpected extra keys emit a
    :class:`~fyst_trajectories.exceptions.PointingWarning`.

    Parameters
    ----------
    params : mapping of str to object
        The candidate computed_params dict.
    scan_type : str
        One of ``"pong"``, ``"constant_el"``, or ``"daisy"``.

    Raises
    ------
    KeyError
        If ``scan_type`` is unknown or ``params`` is missing any key
        required by that scan type.
    """
    if scan_type not in _SCAN_TYPE_TO_KEYS:
        raise KeyError(
            f"Unknown scan_type {scan_type!r}; expected one of {sorted(_SCAN_TYPE_TO_KEYS)}"
        )
    expected = _SCAN_TYPE_TO_KEYS[scan_type]
    actual = set(params)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise KeyError(f"{scan_type} computed_params missing required keys: {sorted(missing)}")
    if extra:
        warnings.warn(
            f"{scan_type} computed_params has unexpected keys: {sorted(extra)}",
            PointingWarning,
            stacklevel=2,
        )
