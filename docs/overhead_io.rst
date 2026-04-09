Timeline I/O
============

:func:`~fyst_trajectories.overhead.write_timeline` and :func:`~fyst_trajectories.overhead.read_timeline`
provide round-trip serialization in TOAST-compatible ECSV format.

Writing a Timeline
------------------

::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        generate_timeline,
        write_timeline,
    )

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T04:00:00",
    )

    write_timeline(timeline, "schedule.ecsv")

Reading a Timeline
------------------

::

    from fyst_trajectories.overhead import read_timeline

    timeline = read_timeline("schedule.ecsv")
    print(f"Loaded {len(timeline)} blocks")
    print(f"Efficiency: {timeline.efficiency:.1%}")

ECSV Format
-----------

The ECSV file uses TOAST v5 column names for compatibility with the
CMB analysis pipeline:

+---------------------+--------+----------------------------------------------+
| Column              | Type   | Description                                  |
+=====================+========+==============================================+
| ``start_timestamp`` | float  | Start time as MJD                            |
+---------------------+--------+----------------------------------------------+
| ``stop_timestamp``  | float  | Stop time as MJD                             |
+---------------------+--------+----------------------------------------------+
| ``name``            | str    | Patch name or calibration type               |
+---------------------+--------+----------------------------------------------+
| ``azmin``           | float  | Minimum azimuth (deg)                        |
+---------------------+--------+----------------------------------------------+
| ``azmax``           | float  | Maximum azimuth (deg)                        |
+---------------------+--------+----------------------------------------------+
| ``el``              | float  | Elevation (deg)                              |
+---------------------+--------+----------------------------------------------+
| ``scan``            | int    | Scan counter                                 |
+---------------------+--------+----------------------------------------------+
| ``subscan``         | int    | Sub-scan index                               |
+---------------------+--------+----------------------------------------------+
| ``boresight_angle`` | float  | Boresight rotation angle (deg)               |
+---------------------+--------+----------------------------------------------+
| ``block_type``      | str    | FYST extension: science/calibration/slew/idle|
+---------------------+--------+----------------------------------------------+
| ``scan_type``       | str    | FYST extension: pattern or cal type          |
+---------------------+--------+----------------------------------------------+
| ``rising``          | bool   | FYST extension: rising-side flag             |
+---------------------+--------+----------------------------------------------+

Header metadata includes site coordinates, overhead model parameters, and
calibration policy cadences.

TOAST Compatibility
-------------------

Standard TOAST schedule files (without ``block_type``, ``scan_type``,
``rising`` columns) are read as all-science timelines. The reader auto-detects
the format and applies sensible defaults.

To produce a TOAST-only file, filter to science blocks before writing::

    from fyst_trajectories.overhead import ObservingTimeline, write_timeline

    science_only = ObservingTimeline(
        blocks=timeline.science_blocks,
        site=timeline.site,
        start_time=timeline.start_time,
        end_time=timeline.end_time,
        overhead_model=timeline.overhead_model,
        calibration_policy=timeline.calibration_policy,
    )
    write_timeline(science_only, "toast_schedule.ecsv")
