Planning Module
===============

High-level planning functions that translate astronomer-friendly inputs
into pattern configurations and trajectories.

.. automodule:: fyst_trajectories.planning
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PongComputedParams, ConstantElComputedParams,
                     DaisyComputedParams, ComputedParams,
                     validate_computed_params

Computed Parameter Schemas
--------------------------

Each planner function returns a :class:`ScanBlock` whose
``computed_params`` attribute follows a scan-type-specific schema.

.. autoclass:: fyst_trajectories.planning.PongComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.ConstantElComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.DaisyComputedParams
   :members:

.. autofunction:: fyst_trajectories.planning.validate_computed_params
