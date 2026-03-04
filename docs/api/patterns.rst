Patterns Package
================

Scan pattern implementations for telescope trajectory generation.

Overview
--------

Use ``TrajectoryBuilder`` with config objects to generate trajectories.
The pattern type is automatically inferred from the config class::

    from astropy.time import Time

    from fyst_pointing import get_fyst_site
    from fyst_pointing.patterns import PongScanConfig, TrajectoryBuilder

    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=2.0, height=2.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

Available patterns: ``constant_el``, ``daisy``, ``linear``, ``planet``, ``pong``, ``sidereal``.

TrajectoryBuilder
-----------------

.. autoclass:: fyst_pointing.patterns.TrajectoryBuilder
   :members:
   :undoc-members:

**Detector offset support**::

    from astropy.time import Time

    from fyst_pointing.primecam import get_primecam_offset

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=1.0, height=1.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .for_detector(get_primecam_offset("i1"))
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

Base Classes
------------

.. autoclass:: fyst_pointing.patterns.ScanPattern
   :members:

.. autoclass:: fyst_pointing.patterns.CelestialPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.AltAzPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.TrajectoryMetadata
   :members:

Configuration Classes
---------------------

.. autoclass:: fyst_pointing.patterns.ScanConfig
   :members:

.. autoclass:: fyst_pointing.patterns.ConstantElScanConfig
   :members:
   :show-inheritance:

.. tip::

   For field-based observations, use
   :func:`~fyst_pointing.planning.plan_constant_el_scan` instead of manually
   constructing ``ConstantElScanConfig``. It auto-computes the azimuth range,
   duration, and number of scans from a ``FieldRegion``.

.. autoclass:: fyst_pointing.patterns.PongScanConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.DaisyScanConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.SiderealTrackConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.PlanetTrackConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.LinearMotionConfig
   :members:
   :show-inheritance:

Pattern Classes
---------------

.. autoclass:: fyst_pointing.patterns.ConstantElScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.PongScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.DaisyScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.SiderealTrackPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.PlanetTrackPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_pointing.patterns.LinearMotionPattern
   :members:
   :show-inheritance:

Pattern Selection
-----------------

+-------------------+------------------+-------------------+
| Pattern           | Base Class       | Key Config Params |
+===================+==================+===================+
| ``sidereal``      | CelestialPattern | ``ra``, ``dec``   |
+-------------------+------------------+-------------------+
| ``planet``        | AltAzPattern     | ``body``          |
+-------------------+------------------+-------------------+
| ``pong``          | CelestialPattern | ``width``,        |
|                   |                  | ``height``,       |
|                   |                  | ``spacing``       |
+-------------------+------------------+-------------------+
| ``daisy``         | CelestialPattern | ``radius``,       |
|                   |                  | ``velocity``      |
+-------------------+------------------+-------------------+
| ``constant_el``   | AltAzPattern     | ``az_start``,     |
|                   |                  | ``az_stop``,      |
|                   |                  | ``elevation``     |
+-------------------+------------------+-------------------+
| ``linear``        | AltAzPattern     | ``az_velocity``,  |
|                   |                  | ``el_velocity``   |
+-------------------+------------------+-------------------+

Registry Functions (Advanced)
-----------------------------

For interactive discovery or dynamic scenarios where pattern names are
determined at runtime::

    from fyst_pointing import list_patterns, get_pattern

    # List available patterns
    print(list_patterns())

    # Get pattern class by name (useful for plugins or config-driven selection)
    PatternClass = get_pattern("pong")

.. autofunction:: fyst_pointing.patterns.list_patterns

.. autofunction:: fyst_pointing.patterns.get_pattern

.. autofunction:: fyst_pointing.patterns.register_pattern
