Overhead Timeline Generation
============================

Sequences science scans with calibration breaks and slew overhead
over a given observing window.

.. autofunction:: fyst_trajectories.overhead.generate_timeline

Constraints
-----------

.. autoclass:: fyst_trajectories.overhead.Constraint
   :members:

.. autoclass:: fyst_trajectories.overhead.ElevationConstraint
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.SunAvoidanceConstraint
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.MoonAvoidanceConstraint
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.MinDurationConstraint
   :members:
   :show-inheritance:

Utilities
---------

.. autofunction:: fyst_trajectories.overhead.estimate_slew_time

.. autofunction:: fyst_trajectories.overhead.get_observable_windows

.. autofunction:: fyst_trajectories.overhead.get_transit_time

.. autofunction:: fyst_trajectories.overhead.get_max_elevation
