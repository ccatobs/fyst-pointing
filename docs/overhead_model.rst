Overhead Model and Calibration Policy
======================================

Two configuration objects control overhead timing:
:class:`~fyst_trajectories.overhead.OverheadModel` for activity durations, and
:class:`~fyst_trajectories.overhead.CalibrationPolicy` for how often each calibration
is performed.

OverheadModel
-------------

Controls the duration of each non-science activity::

    from fyst_trajectories.overhead import OverheadModel

    model = OverheadModel(
        retune_duration=5.0,          # KID probe tone reset (s)
        pointing_cal_duration=180.0,  # pointing correction scan (s)
        focus_duration=300.0,         # focus check (s)
        skydip_duration=300.0,        # elevation nod (s)
        planet_cal_duration=600.0,    # planet calibration scan (s)
        settle_time=5.0,              # post-slew settling (s)
        min_scan_duration=60.0,       # minimum useful science scan (s)
        max_scan_duration=3600.0,     # forced split threshold (s)
    )

``min_scan_duration`` prevents short, wasteful scans. ``max_scan_duration``
forces long observations to split into sub-scans with retune breaks.

CalibrationPolicy
-----------------

Controls *when* each calibration type is triggered. Cadences are in seconds.
A cadence of 0 means "every scan boundary"::

    from fyst_trajectories.overhead import CalibrationPolicy

    policy = CalibrationPolicy(
        retune_cadence=0.0,           # every scan boundary
        pointing_cadence=1800.0,      # every 30 min
        focus_cadence=7200.0,         # every 2 hours
        skydip_cadence=10800.0,       # every 3 hours
        planet_cal_cadence=43200.0,   # every 12 hours
        planet_targets=("jupiter", "saturn", "mars", "uranus", "neptune"),
        planet_min_elevation=20.0,    # planet must be above this
    )

Planet calibrations are only scheduled when at least one planet target
is above ``planet_min_elevation``.

Comparison with Other Instruments
---------------------------------

+---------------------------+----------+----------+-----------+
| Parameter                 | FYST     | NIKA2    | SO        |
|                           | default  | typical  | schedlib  |
+===========================+==========+==========+===========+
| Retune cadence            | 0 (every)| 0        | 600s      |
+---------------------------+----------+----------+-----------+
| Pointing cal cadence      | 1800s    | 3600s    | 7200s     |
+---------------------------+----------+----------+-----------+
| Focus cadence             | 7200s    | 3600s    | 10800s    |
+---------------------------+----------+----------+-----------+
| Skydip cadence            | 10800s   | 7200s    | 21600s    |
+---------------------------+----------+----------+-----------+
| Planet cal cadence        | 43200s   | N/A      | daily     |
+---------------------------+----------+----------+-----------+
| Retune duration           | 5s       | 3-5s     | 10s       |
+---------------------------+----------+----------+-----------+
| Pointing cal duration     | 180s     | 120-180s | 300s      |
+---------------------------+----------+----------+-----------+
| Max scan duration         | 3600s    | 1200s    | 3600s     |
+---------------------------+----------+----------+-----------+

Customizing for Different Strategies
-------------------------------------

**Quick commissioning** (frequent calibrations)::

    from fyst_trajectories.overhead import CalibrationPolicy, OverheadModel

    commissioning_policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=900.0,    # every 15 min
        focus_cadence=1800.0,      # every 30 min
        skydip_cadence=3600.0,     # hourly
    )

    commissioning_overhead = OverheadModel(
        max_scan_duration=600.0,   # short scans
    )

**Deep survey** (maximize science time)::

    survey_policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=7200.0,   # every 2 hours
        focus_cadence=14400.0,     # every 4 hours
        skydip_cadence=21600.0,    # every 6 hours
    )

    survey_overhead = OverheadModel(
        max_scan_duration=3600.0,  # full hour scans
        min_scan_duration=120.0,   # accept 2-min scans
    )
