Trajectory Container
====================

Container for telescope trajectory data with time-stamped position and
velocity setpoints for Az/El axes.

.. automodule:: fyst_pointing.trajectory
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Derived Dynamics Properties
---------------------------

The following read-only properties compute higher-order derivatives on
demand from the stored velocity arrays using ``np.gradient``. They are
not stored fields -- each access recomputes the result.

- ``az_accel`` / ``el_accel``: Acceleration in degrees/second^2
  (gradient of velocity with respect to time).
- ``az_jerk`` / ``el_jerk``: Jerk in degrees/second^3
  (gradient of acceleration with respect to time).

Example::

    accel = trajectory.az_accel          # np.ndarray, same shape as times
    max_jerk = np.abs(trajectory.el_jerk).max()

Coordinate System Fields
------------------------

- ``coordsys``: Coordinate system of trajectory points (``"altaz"`` for patterns)
- ``epoch``: Optional epoch annotation (e.g., ``"J2000"``)
- ``metadata.input_frame``: Input coordinate frame used for generation
- ``metadata.epoch``: Epoch of input coordinates

Usage Examples
--------------

**Manual creation**::

    import numpy as np
    from fyst_pointing import Trajectory

    trajectory = Trajectory(
        times=np.array([0, 1, 2, 3, 4]),
        az=np.array([100, 101, 102, 101, 100]),
        el=np.full(5, 45.0),
        az_vel=np.array([1, 1, 0, -1, -1]),
        el_vel=np.zeros(5),
    )

**Pattern generation** (recommended)::

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

    print(trajectory.pattern_type)   # "pong"
    print(trajectory.center_ra)      # 180.0
    print(trajectory.pattern_params) # {'width': 2.0, ...}

**Export**::

    from fyst_pointing.trajectory_utils import to_path_format, to_arrays

    # For OCS /path endpoint (preferred: free function)
    points = to_path_format(trajectory)
    payload = {
        "start_time": trajectory.start_time.unix,
        "coordsys": "Horizon",
        "points": points,
    }

    # Simple arrays (preferred: free function)
    times, az, el = to_arrays(trajectory)

    # Method-style calls also work (thin wrappers)
    points = trajectory.to_path_format()
    times, az, el = trajectory.to_arrays()

**Absolute times**::

    from astropy.time import Time

    from fyst_pointing.trajectory_utils import get_absolute_times

    trajectory.start_time = Time("2026-03-15T04:00:00", scale="utc")

    # Preferred: free function
    abs_times = get_absolute_times(trajectory)

    # Method-style call also works (thin wrapper)
    abs_times = trajectory.get_absolute_times()

**Validation**::

    from fyst_pointing import get_fyst_site
    from fyst_pointing.trajectory_utils import validate_trajectory

    site = get_fyst_site()

    # Preferred: free function
    validate_trajectory(trajectory, site)

    # Method-style call also works (thin wrapper)
    trajectory.validate(site)

**Print formatted summary**::

    from fyst_pointing.trajectory_utils import print_trajectory

    print_trajectory(trajectory)              # First 5 and last 5 points
    print_trajectory(trajectory, head=10)     # Customize head count
    print_trajectory(trajectory, tail=None)   # Skip tail section

**Plot trajectory**::

    from fyst_pointing.trajectory_utils import plot_trajectory

    # Display interactive plot
    fig = plot_trajectory(trajectory, show=True)

    # Get figure for saving
    fig = plot_trajectory(trajectory, show=False)
    fig.savefig("trajectory.png")
