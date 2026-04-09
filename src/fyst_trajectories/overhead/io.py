"""TOAST-compatible ECSV timeline I/O.

Reads and writes observation timelines in TOAST v5 (ECSV) format
with FYST-specific extensions for calibration blocks.
"""

import json
import math
from pathlib import Path

from astropy.table import Table
from astropy.time import Time

from ..site import (
    AxisLimits,
    Site,
    SunAvoidanceConfig,
    TelescopeLimits,
    get_fyst_site,
)
from .models import (
    BlockType,
    CalibrationPolicy,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)

__all__ = [
    "read_timeline",
    "write_timeline",
]


def _nasmyth_rotation(az: float, el: float, site: Site) -> float:
    """Compute Nasmyth field rotation from AltAz coordinates.

    Uses ``nasmyth_sign * elevation + parallactic_angle`` where the
    parallactic angle is derived from azimuth, elevation, and site latitude.

    Parameters
    ----------
    az : float
        Azimuth in degrees.
    el : float
        Elevation in degrees.
    site : Site
        Site configuration with latitude and nasmyth_sign.

    Returns
    -------
    float
        Nasmyth rotation in degrees.
    """
    # NOTE: This computes the parallactic angle from AltAz coordinates,
    # which is mathematically equivalent to the HA-based formula in
    # coordinates.get_parallactic_angle(). We use the AltAz form here
    # because TimelineBlock stores az/el directly. See
    # TestNasmythConsistency in tests/overhead/test_io.py for the
    # equivalence check.
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    lat_rad = math.radians(site.latitude)

    sin_az = math.sin(az_rad)
    cos_az = math.cos(az_rad)
    sin_el = math.sin(el_rad)
    cos_el = math.cos(el_rad)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Parallactic angle from AltAz (IAU convention)
    numerator = -sin_az * cos_lat
    denominator = sin_lat * cos_el - cos_lat * sin_el * cos_az
    pa = math.degrees(math.atan2(numerator, denominator))

    return site.nasmyth_sign * el + pa


def write_timeline(
    timeline: ObservingTimeline,
    path: str | Path,
) -> None:
    """Write a timeline to a TOAST-compatible ECSV file.

    The output uses TOAST canonical column names (``start_time``,
    ``stop_time`` as ISO strings, ``name``, ``azmin``, ``azmax``,
    ``el``, ``boresight_angle``, ``scan_index``, ``subscan_index``)
    plus FYST extension columns (``block_type``, ``scan_type``,
    ``rising``, plus metadata columns for science blocks:
    ``ra_center``, ``dec_center``, ``width``, ``height``, ``velocity``,
    ``scan_params_json``).

    Parameters
    ----------
    timeline : ObservingTimeline
        Timeline to write.
    path : str or Path
        Output file path. Should end in ``.ecsv``.
    """
    path = Path(path)

    rows = []
    for block in timeline.blocks:
        # Prefer the stored boresight_angle on the block; fall back to
        # recomputing from az/el for timelines that were built without
        # populating the field (backward compatibility).
        bangle = block.boresight_angle
        if bangle == 0.0:
            bangle = _nasmyth_rotation(
                0.5 * (block.az_min + block.az_max),
                block.elevation,
                timeline.site,
            )

        is_science = block.block_type == BlockType.SCIENCE
        meta = block.metadata if is_science else {}
        rows.append(
            {
                "start_time": block.t_start.iso,
                "stop_time": block.t_stop.iso,
                "boresight_angle": bangle,
                "name": block.patch_name,
                "azmin": block.az_min,
                "azmax": block.az_max,
                "el": block.elevation,
                "scan_index": block.scan_index,
                "subscan_index": block.subscan_index,
                "block_type": str(block.block_type),
                "scan_type": block.scan_type,
                "rising": block.rising,
                "ra_center": float(meta.get("ra_center", 0.0)),
                "dec_center": float(meta.get("dec_center", 0.0)),
                "width": float(meta.get("width", 0.0)),
                "height": float(meta.get("height", 0.0)),
                "velocity": float(meta.get("velocity", 0.0)),
                "scan_params_json": json.dumps(meta.get("scan_params", {})),
            }
        )

    if not rows:
        rows = [_empty_row()]

    table = Table(rows)

    table.meta["site_name"] = timeline.site.name
    table.meta["telescope_name"] = "FYST"
    table.meta["site_lat"] = timeline.site.latitude
    table.meta["site_lon"] = timeline.site.longitude
    table.meta["site_alt"] = timeline.site.elevation
    table.meta["retune_duration"] = timeline.overhead_model.retune_duration
    table.meta["max_scan_duration"] = timeline.overhead_model.max_scan_duration
    table.meta["min_scan_duration"] = timeline.overhead_model.min_scan_duration
    table.meta["retune_cadence"] = timeline.calibration_policy.retune_cadence
    table.meta["pointing_cadence"] = timeline.calibration_policy.pointing_cadence
    table.meta["focus_cadence"] = timeline.calibration_policy.focus_cadence
    table.meta.update(timeline.metadata)

    table.write(str(path), format="ascii.ecsv", overwrite=True)


def read_timeline(path: str | Path) -> ObservingTimeline:
    """Read a timeline from a TOAST-compatible ECSV file.

    Handles both standard TOAST format (science blocks only, no
    ``block_type`` column) and FYST extended format. Files written by
    older versions of fyst-trajectories that used ``start_timestamp``
    / ``scan`` column names are still read correctly.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    ObservingTimeline
        Loaded timeline.
    """
    path = Path(path)
    table = Table.read(str(path), format="ascii.ecsv")

    meta = table.meta
    site = _site_from_meta(meta)

    overhead = OverheadModel(
        retune_duration=meta.get("retune_duration", 5.0),
        max_scan_duration=meta.get("max_scan_duration", 3600.0),
        min_scan_duration=meta.get("min_scan_duration", 60.0),
    )

    cal_policy = CalibrationPolicy(
        retune_cadence=meta.get("retune_cadence", 0.0),
        pointing_cadence=meta.get("pointing_cadence", 1800.0),
        focus_cadence=meta.get("focus_cadence", 7200.0),
    )

    has_block_type = "block_type" in table.colnames
    has_scan_type = "scan_type" in table.colnames
    has_rising = "rising" in table.colnames
    has_mjd = "start_timestamp" in table.colnames
    has_scan = "scan" in table.colnames  # legacy column
    has_boresight = "boresight_angle" in table.colnames
    has_metadata = "ra_center" in table.colnames

    blocks = []
    for row in table:
        if has_mjd:
            t_start = Time(float(row["start_timestamp"]), format="mjd", scale="utc")
            t_stop = Time(float(row["stop_timestamp"]), format="mjd", scale="utc")
        else:
            t_start = Time(str(row["start_time"]), scale="utc")
            t_stop = Time(str(row["stop_time"]), scale="utc")

        block_type = str(row["block_type"]) if has_block_type else "science"
        scan_type = str(row["scan_type"]) if has_scan_type else ""
        rising = bool(row["rising"]) if has_rising else (float(row["azmin"]) % 360 < 180)

        scan_col = "scan" if has_scan else "scan_index"
        subscan_col = "subscan" if has_scan else "subscan_index"

        block_meta: dict = {}
        if has_metadata and block_type == "science":
            block_meta = {
                "ra_center": float(row["ra_center"]),
                "dec_center": float(row["dec_center"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
                "velocity": float(row["velocity"]),
                "scan_params": json.loads(str(row["scan_params_json"]))
                if "scan_params_json" in table.colnames
                else {},
            }

        boresight = float(row["boresight_angle"]) if has_boresight else 0.0

        block = TimelineBlock(
            t_start=t_start,
            t_stop=t_stop,
            block_type=block_type,
            patch_name=str(row["name"]),
            az_min=float(row["azmin"]),
            az_max=float(row["azmax"]),
            elevation=float(row["el"]),
            scan_index=int(row[scan_col]),
            subscan_index=int(row[subscan_col]),
            rising=rising,
            scan_type=scan_type,
            boresight_angle=boresight,
            metadata=block_meta,
        )
        blocks.append(block)

    if blocks:
        tl_start = min(b.t_start for b in blocks)
        tl_end = max(b.t_stop for b in blocks)
    else:
        tl_start = Time("2000-01-01T00:00:00", scale="utc")
        tl_end = tl_start

    return ObservingTimeline(
        blocks=blocks,
        site=site,
        start_time=tl_start,
        end_time=tl_end,
        overhead_model=overhead,
        calibration_policy=cal_policy,
        metadata={k: v for k, v in meta.items() if k not in _KNOWN_META_KEYS},
    )


def _site_from_meta(meta: dict) -> Site:
    """Reconstruct a ``Site`` from ECSV table metadata.

    If ``site_lat``/``site_lon``/``site_alt`` are present and match the
    FYST coordinates to 4 decimal places, ``get_fyst_site()`` is used so
    the returned site has the full FYST default limits and atmosphere.
    Otherwise a custom ``Site`` is constructed using the metadata
    coordinates together with the default FYST telescope limits and
    sun avoidance settings (which are not currently persisted).
    """
    fyst = get_fyst_site()
    lat = meta.get("site_lat")
    lon = meta.get("site_lon")
    alt = meta.get("site_alt")

    if lat is None or lon is None or alt is None:
        return fyst

    lat = float(lat)
    lon = float(lon)
    alt = float(alt)

    if round(lat, 4) == round(fyst.latitude, 4) and round(lon, 4) == round(fyst.longitude, 4):
        return fyst

    # Non-FYST site: build a custom Site using the stored coordinates plus
    # the default FYST mechanical limits and sun-avoidance config (which
    # are not serialised separately in v0.3 ECSV files).
    return Site(
        name=str(meta.get("site_name", "custom")),
        description=str(meta.get("site_description", "")),
        latitude=lat,
        longitude=lon,
        elevation=alt,
        atmosphere=None,
        telescope_limits=TelescopeLimits(
            azimuth=AxisLimits(
                min=fyst.telescope_limits.azimuth.min,
                max=fyst.telescope_limits.azimuth.max,
                max_velocity=fyst.telescope_limits.azimuth.max_velocity,
                max_acceleration=fyst.telescope_limits.azimuth.max_acceleration,
            ),
            elevation=AxisLimits(
                min=fyst.telescope_limits.elevation.min,
                max=fyst.telescope_limits.elevation.max,
                max_velocity=fyst.telescope_limits.elevation.max_velocity,
                max_acceleration=fyst.telescope_limits.elevation.max_acceleration,
            ),
        ),
        sun_avoidance=SunAvoidanceConfig(
            enabled=fyst.sun_avoidance.enabled,
            exclusion_radius=fyst.sun_avoidance.exclusion_radius,
            warning_radius=fyst.sun_avoidance.warning_radius,
        ),
        nasmyth_port=fyst.nasmyth_port,
        plate_scale=fyst.plate_scale,
    )


def _empty_row() -> dict:
    """Create an empty row for an empty timeline.

    Returns
    -------
    dict
        Row with TOAST-compatible column names and zero values.
    """
    t0 = Time("2000-01-01T00:00:00", scale="utc")
    return {
        "start_time": t0.iso,
        "stop_time": t0.iso,
        "boresight_angle": 0.0,
        "name": "",
        "azmin": 0.0,
        "azmax": 0.0,
        "el": 0.0,
        "scan_index": 0,
        "subscan_index": 0,
        "block_type": "idle",
        "scan_type": "",
        "rising": True,
        "ra_center": 0.0,
        "dec_center": 0.0,
        "width": 0.0,
        "height": 0.0,
        "velocity": 0.0,
        "scan_params_json": "{}",
    }


_KNOWN_META_KEYS = frozenset(
    {
        "site_name",
        "telescope_name",
        "site_lat",
        "site_lon",
        "site_alt",
        "retune_duration",
        "max_scan_duration",
        "min_scan_duration",
        "retune_cadence",
        "pointing_cadence",
        "focus_cadence",
    }
)
