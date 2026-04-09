"""Data model for observation scheduling.

Observation patches, calibration specifications, timeline blocks, overhead
models, calibration policies, and complete timelines.

Inspired by SO schedlib's Block hierarchy but simplified for FYST/Prime-Cam.
"""

import dataclasses
import enum
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astropy.time import Time

if TYPE_CHECKING:
    from ..planning import FieldRegion
    from ..site import Site

__all__ = [
    "BlockType",
    "CalibrationPolicy",
    "CalibrationSpec",
    "CalibrationType",
    "ObservingPatch",
    "ObservingTimeline",
    "OverheadModel",
    "TimelineBlock",
]


class BlockType(str, enum.Enum):
    """Type identifier for a :class:`TimelineBlock`.

    Members compare equal to their string values, so either the enum
    member (``BlockType.SCIENCE``) or the plain string (``"science"``)
    can be used interchangeably.
    """

    SCIENCE = "science"
    CALIBRATION = "calibration"
    SLEW = "slew"
    IDLE = "idle"

    def __str__(self) -> str:
        return self.value


class CalibrationType(str, enum.Enum):
    """Calibration operation types.

    Members compare equal to their string values, so either the enum
    member (``CalibrationType.RETUNE``) or the plain string
    (``"retune"``) can be used interchangeably.
    """

    RETUNE = "retune"
    POINTING_CAL = "pointing_cal"
    FOCUS = "focus"
    SKYDIP = "skydip"
    PLANET_CAL = "planet_cal"
    BEAM_MAP = "beam_map"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ObservingPatch:
    """A sky region to observe.

    Equivalent to a TOAST 'patch' or SO schedlib 'SourceBlock'.

    Parameters
    ----------
    name : str
        Unique identifier for this patch.
    ra_center : float
        Right Ascension of field center in degrees.
    dec_center : float
        Declination of field center in degrees.
    width : float
        Angular width of the field in degrees.
    height : float
        Angular height of the field in degrees.
    scan_type : str
        Scan pattern type: ``"constant_el"``, ``"pong"``, or ``"daisy"``.
    velocity : float
        Scan velocity in deg/s.
    priority : float
        Scheduling priority. Lower values = higher priority.
    weight : float
        Scheduling weight for patch equalization.
    elevation : float or None
        Fixed elevation for CE scans (degrees). None for auto-compute.
    scan_params : dict
        Additional scan-type-specific parameters (e.g., spacing, num_terms,
        az_min/az_max overrides).
    """

    name: str
    ra_center: float
    dec_center: float
    width: float
    height: float
    scan_type: str
    velocity: float
    priority: float = 1.0
    weight: float = 1.0
    elevation: float | None = None
    scan_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.scan_type not in ("constant_el", "pong", "daisy"):
            raise ValueError(
                f"scan_type must be 'constant_el', 'pong', or 'daisy', got '{self.scan_type}'"
            )
        if self.velocity <= 0:
            raise ValueError(f"velocity must be positive, got {self.velocity}")
        if self.priority <= 0:
            raise ValueError(f"priority must be positive, got {self.priority}")
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")

    @classmethod
    def from_field_region(
        cls,
        field: "FieldRegion",
        name: str,
        scan_type: str,
        velocity: float,
        **kwargs: Any,
    ) -> "ObservingPatch":
        """Create an ObservingPatch from an existing FieldRegion.

        Parameters
        ----------
        field : FieldRegion
            Field region from ``fyst_trajectories.planning``.
        name : str
            Unique patch identifier.
        scan_type : str
            Scan pattern type.
        velocity : float
            Scan velocity in deg/s.
        **kwargs
            Additional keyword arguments passed to ``ObservingPatch``.

        Returns
        -------
        ObservingPatch
        """
        return cls(
            name=name,
            ra_center=field.ra_center,
            dec_center=field.dec_center,
            width=field.width,
            height=field.height,
            scan_type=scan_type,
            velocity=velocity,
            **kwargs,
        )

    @property
    def dec_min(self) -> float:
        """Minimum declination of the field in degrees."""
        return self.dec_center - self.height / 2.0

    @property
    def dec_max(self) -> float:
        """Maximum declination of the field in degrees."""
        return self.dec_center + self.height / 2.0


@dataclass(frozen=True)
class CalibrationSpec:
    """Specification for a single calibration operation.

    Parameters
    ----------
    name : str or CalibrationType
        Calibration type. Accepts string values for backward
        compatibility; they are coerced to ``CalibrationType``.
    duration : float
        Expected duration in seconds.
    target : str or None
        Planet name for planet calibrations, None for in-place operations.
    elevation : float or None
        Required elevation (e.g., for skydips), or None.
    """

    name: CalibrationType | str
    duration: float
    target: str | None = None
    elevation: float | None = None

    def __post_init__(self) -> None:
        # Coerce string names to CalibrationType for backward compat.
        if isinstance(self.name, str) and not isinstance(self.name, CalibrationType):
            try:
                object.__setattr__(self, "name", CalibrationType(self.name))
            except ValueError:
                raise ValueError(
                    f"Unknown calibration type '{self.name}', expected one of "
                    f"{[e.value for e in CalibrationType]}"
                ) from None
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")


@dataclass(frozen=True)
class TimelineBlock:
    """A single time-bounded entry in a timeline.

    Represents either a science observation, calibration operation,
    telescope slew, or idle period.

    Parameters
    ----------
    t_start : Time
        UTC start time.
    t_stop : Time
        UTC stop time.
    block_type : BlockType or str
        Block kind. Accepts string values for backward compatibility;
        they are coerced to :class:`BlockType`.
    patch_name : str
        Patch name (science) or calibration type name.
    az_min : float
        Minimum azimuth in degrees.
    az_max : float
        Maximum azimuth in degrees.
    elevation : float
        Elevation in degrees.
    scan_index : int
        Parent scan counter.
    subscan_index : int
        Sub-scan index within a split observation (0 if unsplit).
    rising : bool
        Whether this is a rising-side observation.
    scan_type : str
        Scan pattern or calibration type identifier.
    boresight_angle : float
        Nasmyth/boresight rotation angle in degrees (``nasmyth_sign *
        elevation + parallactic_angle``). ``0.0`` means unset — I/O
        routines will recompute it from az/el as needed.
    metadata : dict
        Additional metadata (scan_params, ra_center, dec_center, width,
        height, velocity, cal_type, etc.). For science blocks, the
        geometry keys must be populated for the simulation bridge to
        reconstruct trajectories after an ECSV round-trip.
    """

    t_start: Time
    t_stop: Time
    block_type: BlockType | str
    patch_name: str
    az_min: float
    az_max: float
    elevation: float
    scan_index: int
    subscan_index: int = 0
    rising: bool = True
    scan_type: str = ""
    boresight_angle: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.block_type, BlockType):
            try:
                object.__setattr__(self, "block_type", BlockType(self.block_type))
            except ValueError:
                valid = [bt.value for bt in BlockType]
                raise ValueError(
                    f"block_type must be one of {valid}, got {self.block_type!r}"
                ) from None
        if self.t_stop.unix < self.t_start.unix:
            raise ValueError(f"t_stop ({self.t_stop.iso}) must be >= t_start ({self.t_start.iso})")

    @property
    def duration(self) -> float:
        """Block duration in seconds."""
        return (self.t_stop - self.t_start).sec


@dataclass(frozen=True)
class OverheadModel:
    """Timing parameters for non-science activities.

    Default values based on NIKA2 experience and FYST telescope specs.
    KID retune is ~3-5 seconds (probe tone reset). Pointing correction
    and focus checks follow NIKA2/TolTEC cadences.

    Parameters
    ----------
    retune_duration : float
        KID probe tone reset duration in seconds.
    pointing_cal_duration : float
        Pointing correction scan duration in seconds.
    focus_duration : float
        Focus check duration in seconds.
    skydip_duration : float
        Sky dip / elevation nod duration in seconds.
    planet_cal_duration : float
        Planet calibration scan duration in seconds.
    settle_time : float
        Post-slew settling time in seconds.
    min_scan_duration : float
        Minimum useful science scan duration in seconds.
    max_scan_duration : float
        Maximum scan duration before forced split in seconds.
    """

    retune_duration: float = 5.0
    pointing_cal_duration: float = 180.0
    focus_duration: float = 300.0
    skydip_duration: float = 300.0
    planet_cal_duration: float = 600.0
    settle_time: float = 5.0
    min_scan_duration: float = 60.0
    max_scan_duration: float = 3600.0

    def __post_init__(self) -> None:
        for fld in dataclasses.fields(self):
            val = getattr(self, fld.name)
            if val < 0:
                raise ValueError(f"{fld.name} must be non-negative, got {val}")
        if self.min_scan_duration >= self.max_scan_duration:
            raise ValueError(
                f"min_scan_duration ({self.min_scan_duration}) must be less than "
                f"max_scan_duration ({self.max_scan_duration})"
            )

    def get_calibration_duration(self, cal_type: CalibrationType | str) -> float:
        """Get duration for a calibration type.

        Parameters
        ----------
        cal_type : CalibrationType or str
            Calibration type name.

        Returns
        -------
        float
            Duration in seconds.
        """
        if isinstance(cal_type, str) and not isinstance(cal_type, CalibrationType):
            cal_type = CalibrationType(cal_type)
        duration_map = {
            CalibrationType.RETUNE: self.retune_duration,
            CalibrationType.POINTING_CAL: self.pointing_cal_duration,
            CalibrationType.FOCUS: self.focus_duration,
            CalibrationType.SKYDIP: self.skydip_duration,
            CalibrationType.PLANET_CAL: self.planet_cal_duration,
            CalibrationType.BEAM_MAP: self.planet_cal_duration,
        }
        if cal_type not in duration_map:
            raise ValueError(f"Unknown calibration type: {cal_type}")
        return duration_map[cal_type]


@dataclass(frozen=True)
class CalibrationPolicy:
    """Cadences for calibration operations.

    A cadence of 0 means "perform after every science scan". Cadences
    are in seconds.

    Parameters
    ----------
    retune_cadence : float
        Seconds between KID retunes. 0 = every scan boundary.
    pointing_cadence : float
        Seconds between pointing corrections.
    focus_cadence : float
        Seconds between focus checks.
    skydip_cadence : float
        Seconds between sky dips.
    planet_cal_cadence : float
        Seconds between planet calibrations.
    planet_targets : tuple of str
        Planet names to use for calibration (e.g., ``["jupiter", "saturn"]``).
    planet_min_elevation : float
        Minimum altitude in degrees for a planet to be considered visible
        for calibration. Default is 20.0 degrees.
    """

    retune_cadence: float = 0.0
    pointing_cadence: float = 1800.0
    focus_cadence: float = 7200.0
    skydip_cadence: float = 10800.0
    planet_cal_cadence: float = 43200.0
    planet_targets: tuple[str, ...] = ("jupiter", "saturn", "mars", "uranus", "neptune")
    planet_min_elevation: float = 20.0

    def __post_init__(self) -> None:
        for fld in (
            "retune_cadence",
            "pointing_cadence",
            "focus_cadence",
            "skydip_cadence",
            "planet_cal_cadence",
        ):
            val = getattr(self, fld)
            if val < 0:
                raise ValueError(f"{fld} must be non-negative, got {val}")


# ObservingTimeline is intentionally non-frozen: the scheduler builds it
# incrementally by appending to ``blocks``.  Once returned to the caller it
# should be treated as read-only (analogous to ``Trajectory`` in trajectory.py).
@dataclass
class ObservingTimeline:
    """A complete observation timeline.

    Contains an ordered sequence of timeline blocks (science, calibration,
    slew, idle) along with the configuration used to generate them.

    Parameters
    ----------
    blocks : list of TimelineBlock
        Time-ordered sequence of timeline entries.
    site : Site
        Observatory site configuration.
    start_time : Time
        Timeline start time (UTC).
    end_time : Time
        Timeline end time (UTC).
    overhead_model : OverheadModel
        Overhead timing parameters used.
    calibration_policy : CalibrationPolicy
        Calibration cadence policy used.
    metadata : dict
        Generation parameters, version info, etc.
    """

    blocks: list[TimelineBlock]
    site: "Site"
    start_time: Time
    end_time: Time
    overhead_model: OverheadModel
    calibration_policy: CalibrationPolicy
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def science_blocks(self) -> list[TimelineBlock]:
        """All science observation blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.SCIENCE]

    @property
    def calibration_blocks(self) -> list[TimelineBlock]:
        """All calibration blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.CALIBRATION]

    @property
    def total_science_time(self) -> float:
        """Total science observation time in seconds."""
        return sum(b.duration for b in self.science_blocks)

    @property
    def total_calibration_time(self) -> float:
        """Total calibration time in seconds."""
        return sum(b.duration for b in self.calibration_blocks)

    @property
    def total_slew_time(self) -> float:
        """Total slew time in seconds."""
        return sum(b.duration for b in self.blocks if b.block_type == BlockType.SLEW)

    @property
    def total_idle_time(self) -> float:
        """Total idle time in seconds."""
        return sum(b.duration for b in self.blocks if b.block_type == BlockType.IDLE)

    @property
    def total_time(self) -> float:
        """Total timeline span in seconds."""
        return (self.end_time - self.start_time).sec

    @property
    def efficiency(self) -> float:
        """Science time as a fraction of total time."""
        total = self.total_time
        if total <= 0:
            return 0.0
        return self.total_science_time / total

    @property
    def n_science_scans(self) -> int:
        """Number of science scan blocks."""
        return len(self.science_blocks)

    def __len__(self) -> int:
        """Return the number of blocks in the timeline."""
        return len(self.blocks)

    def __iter__(self) -> Iterator[TimelineBlock]:
        """Iterate over timeline blocks."""
        return iter(self.blocks)

    def __str__(self) -> str:
        """Human-readable timeline summary."""
        sci_h = self.total_science_time / 3600.0
        cal_h = self.total_calibration_time / 3600.0
        slew_h = self.total_slew_time / 3600.0
        idle_h = self.total_idle_time / 3600.0

        lines = [
            f"ObservingTimeline: {self.start_time.iso} to {self.end_time.iso}",
            f"  Science:     {sci_h:5.1f}h ({self.efficiency:5.1%}), {self.n_science_scans} blocks",
            f"  Calibration: {cal_h:5.1f}h, {len(self.calibration_blocks)} blocks",
            f"  Slew:        {slew_h:5.1f}h",
            f"  Idle:        {idle_h:5.1f}h",
        ]

        patch_times: dict[str, float] = {}
        patch_counts: dict[str, int] = {}
        for b in self.science_blocks:
            patch_times[b.patch_name] = patch_times.get(b.patch_name, 0.0) + b.duration
            patch_counts[b.patch_name] = patch_counts.get(b.patch_name, 0) + 1

        if patch_times:
            parts = [
                f"{name} ({t / 3600:.1f}h, {patch_counts[name]} blks)"
                for name, t in sorted(patch_times.items(), key=lambda x: -x[1])
            ]
            lines.append(f"  Patches:     {', '.join(parts)}")

        return "\n".join(lines)

    def validate(self) -> list[str]:
        """Check timeline for common issues.

        Returns
        -------
        list of str
            Warning messages for any issues found. Empty if clean.
        """
        warnings_list = []
        sorted_blocks = sorted(self.blocks, key=lambda b: b.t_start.unix)

        # Check for overlaps
        for i in range(len(sorted_blocks) - 1):
            if sorted_blocks[i].t_stop.unix > sorted_blocks[i + 1].t_start.unix + 0.01:
                warnings_list.append(
                    f"Overlap: '{sorted_blocks[i].patch_name}' ends at "
                    f"{sorted_blocks[i].t_stop.iso} but "
                    f"'{sorted_blocks[i + 1].patch_name}' starts at "
                    f"{sorted_blocks[i + 1].t_start.iso}"
                )

        # Check time range
        for b in sorted_blocks:
            if b.t_start.unix < self.start_time.unix - 0.01:
                warnings_list.append(
                    f"Block '{b.patch_name}' starts before timeline: {b.t_start.iso}"
                )
            if b.t_stop.unix > self.end_time.unix + 0.01:
                warnings_list.append(f"Block '{b.patch_name}' ends after timeline: {b.t_stop.iso}")

        return warnings_list
