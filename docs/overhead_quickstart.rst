Overhead Quickstart
===================

Basic Usage
-----------

Generate a 4-hour observing timeline with 2 patches::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        generate_timeline,
        compute_budget,
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
        ObservingPatch(
            name="Wide01",
            ra_center=180.0,
            dec_center=-30.0,
            width=20.0,
            height=10.0,
            scan_type="pong",
            velocity=0.5,
        ),
    ]

    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T04:00:00",
    )

    print(timeline)

Efficiency Statistics
---------------------

Use :func:`~fyst_trajectories.overhead.compute_budget` to get a summary::

    stats = compute_budget(timeline)
    print(f"Efficiency: {stats['efficiency']:.1%}")
    print(f"Science:     {stats['science_time'] / 3600:.1f}h")
    print(f"Calibration: {stats['calibration_time'] / 3600:.1f}h")
    print(f"Slew:        {stats['slew_time'] / 3600:.1f}h")
    print(f"Idle:        {stats['idle_time'] / 3600:.1f}h")

The returned dict also contains per-patch breakdowns and calibration
type counts in ``stats['per_patch']`` and ``stats['calibration_breakdown']``.

Saving a Timeline
------------------

Write to TOAST-compatible ECSV and read it back::

    from fyst_trajectories.overhead import write_timeline, read_timeline

    write_timeline(timeline, "my_timeline.ecsv")
    loaded = read_timeline("my_timeline.ecsv")
    print(f"Loaded {len(loaded)} blocks")

See :doc:`overhead_io` for format details and TOAST compatibility.
