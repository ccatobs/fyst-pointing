"""Cross-validation tests for inject_retune().

Tests per-pattern efficiency, turnaround overlap behavior, and
theoretical efficiency matching across different retune cadences.
"""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
)
from fyst_trajectories.trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    Trajectory,
)
from fyst_trajectories.trajectory_utils import inject_retune


@pytest.fixture
def site():
    """FYST site configuration."""
    return get_fyst_site()


@pytest.fixture
def start_time():
    """Nighttime start at FYST."""
    return Time("2026-06-15T04:00:00", scale="utc")


def _make_long_trajectory(
    duration: float = 600.0,
    timestep: float = 0.1,
    turnaround_intervals: list[tuple[float, float]] | None = None,
) -> Trajectory:
    """Create a synthetic trajectory for efficiency tests.

    Parameters
    ----------
    duration : float
        Total duration in seconds.
    timestep : float
        Time step in seconds.
    turnaround_intervals : list of (start, end)
        Time intervals to flag as turnaround.
    """
    times = np.arange(0, duration, timestep)
    n = len(times)
    az = np.linspace(100, 200, n)
    el = np.full(n, 45.0)
    az_vel = np.gradient(az, times)
    el_vel = np.zeros(n)
    scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=int)

    if turnaround_intervals:
        for t_start, t_end in turnaround_intervals:
            mask = (times >= t_start) & (times < t_end)
            scan_flag[mask] = SCAN_FLAG_TURNAROUND

    return Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel, scan_flag=scan_flag)


class TestPerPatternEfficiency:
    """Verify science fraction after inject_retune for each scan pattern."""

    def test_ce_scan_efficiency(self, site, start_time):
        """CE scan with 30s/5s retune should have ~80-87% science fraction."""
        field = FieldRegion(ra_center=24.0, dec_center=-32.0, width=40.0, height=10.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=50.0,
            velocity=1.0,
            site=site,
            start_time=start_time,
            rising=True,
            timestep=0.1,
        )
        traj = block.trajectory
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        science_frac = result.science_mask.sum() / len(result.times)
        # CE scans have turnarounds that can absorb some retune time,
        # so efficiency should be in the 80-87% range
        assert 0.78 <= science_frac <= 0.90, (
            f"CE science fraction {science_frac:.3f} outside expected range [0.78, 0.90]"
        )

        # Retune flags should exist
        retune_count = (result.scan_flag == SCAN_FLAG_RETUNE).sum()
        assert retune_count > 0

    def test_pong_scan_efficiency(self, site, start_time):
        """Pong scan with 30s/5s retune should have ~80-87% science fraction.

        Pong scans have no turnaround flags (scan_flag is None),
        so inject_retune treats all samples as science and places
        retunes purely by time.
        """
        field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
        block = plan_pong_scan(
            field=field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )
        traj = block.trajectory
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        science_frac = result.science_mask.sum() / len(result.times)
        # With scan_flag turnaround flagging + retune injection, science
        # fraction is lower than for patterns without turnaround flags.
        assert 0.65 <= science_frac <= 0.90, (
            f"Pong science fraction {science_frac:.3f} outside expected range [0.65, 0.90]"
        )

    def test_daisy_scan_efficiency(self, site, start_time):
        """Daisy scan with 30s/5s retune should have ~80-87% science fraction.

        Daisy scans are continuous (no turnarounds), so retune events
        always consume science time.
        """
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=1.0,
            velocity=0.5,
            turn_radius=0.5,
            avoidance_radius=0.1,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=300.0,
        )
        traj = block.trajectory
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        science_frac = result.science_mask.sum() / len(result.times)
        assert 0.78 <= science_frac <= 0.90, (
            f"Daisy science fraction {science_frac:.3f} outside expected range [0.78, 0.90]"
        )

    def test_retune_flags_at_correct_intervals(self, site, start_time):
        """Retune events should appear at approximately the configured interval.

        Uses a daisy scan (no turnarounds) for clean interval verification.
        """
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=1.0,
            velocity=0.5,
            turn_radius=0.5,
            avoidance_radius=0.1,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=300.0,
        )
        traj = block.trajectory
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Find retune event start times
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        retune_times = result.times[retune_mask]
        events = _group_retune_events(retune_times)

        # With 300s duration, 30s interval + 5s duration, effective
        # spacing is ~35s (next retune measured from end of previous).
        assert len(events) >= 6
        assert len(events) <= 10

        expected_gap = 30.0 + 5.0
        for i in range(1, len(events)):
            gap = events[i] - events[i - 1]
            assert abs(gap - expected_gap) < 1.0, (
                f"Retune interval {gap:.1f}s deviates from expected {expected_gap:.1f}s"
            )


class TestTurnaroundOverlap:
    """Verify turnaround snapping reduces dead time for CE scans."""

    def test_ce_turnaround_snapping_reduces_dead_time(self, site, start_time):
        """CE scan: snapping should preserve more science than time-based."""
        field = FieldRegion(ra_center=24.0, dec_center=-32.0, width=40.0, height=10.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=50.0,
            velocity=1.0,
            site=site,
            start_time=start_time,
            rising=True,
            timestep=0.1,
        )
        traj = block.trajectory

        # Skip if scan is too short for meaningful comparison
        if traj.duration < 120.0:
            pytest.skip("CE scan too short for turnaround overlap test")

        result_snap = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        result_time = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )

        frac_snap = result_snap.science_mask.sum() / len(result_snap.times)
        frac_time = result_time.science_mask.sum() / len(result_time.times)

        # Turnaround snapping should preserve at least as much science
        # time (>=) since it can overlap retunes with existing dead time.
        # Allow tiny tolerance for floating-point edge effects.
        assert frac_snap >= frac_time - 0.005, (
            f"Snapping ({frac_snap:.4f}) should be >= time-based ({frac_time:.4f})"
        )

    def test_synthetic_turnaround_overlap_count(self):
        """With turnarounds at retune due times, snapping should use them.

        Creates a synthetic trajectory with turnarounds at 28-31s,
        58-61s, 88-91s -- near the 30s, 60s, 90s due times. With
        prefer_turnarounds=True, retunes should snap to these.
        """
        turnarounds = [(28.0, 31.0), (58.0, 61.0), (88.0, 91.0)]
        traj = _make_long_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=turnarounds,
        )

        # With snapping
        result_snap = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        # Without snapping
        result_time = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )

        # With snapping, retunes overlap turnaround positions, so the
        # additional science lost should be less
        snap_science = result_snap.science_mask.sum()
        time_science = result_time.science_mask.sum()

        # Snapping should preserve more (or equal) science samples
        assert snap_science >= time_science, (
            f"Snapping preserved {snap_science} science samples vs {time_science} without snapping"
        )


class TestTheoreticalEfficiency:
    """Verify inject_retune efficiency matches theoretical predictions.

    Theoretical formula: efficiency = (interval - duration) / interval
    This assumes long trajectories where edge effects are negligible.
    """

    @pytest.mark.parametrize(
        "interval, duration, expected",
        [
            (30.0, 5.0, 0.833),  # 25/30 = 83.3%
            (60.0, 5.0, 0.917),  # 55/60 = 91.7%
            (30.0, 2.0, 0.933),  # 28/30 = 93.3%
        ],
        ids=["30s/5s", "60s/5s", "30s/2s"],
    )
    def test_efficiency_matches_theory(self, interval, duration, expected):
        """Long trajectory efficiency should match theoretical value within 3%."""
        traj = _make_long_trajectory(duration=600.0, timestep=0.1)
        result = inject_retune(
            traj,
            retune_interval=interval,
            retune_duration=duration,
            prefer_turnarounds=False,
        )

        science_frac = result.science_mask.sum() / len(result.times)
        assert abs(science_frac - expected) < 0.03, (
            f"Science fraction {science_frac:.4f} deviates from "
            f"theoretical {expected:.4f} by more than 3%"
        )

    def test_longer_trajectory_closer_to_theory(self):
        """Longer trajectories should have smaller edge effects.

        A 1200s trajectory should be closer to 83.3% than a 120s one
        with 30s/5s retune.
        """
        expected = 0.833

        short = _make_long_trajectory(duration=120.0, timestep=0.1)
        long = _make_long_trajectory(duration=1200.0, timestep=0.1)

        short_result = inject_retune(
            short,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )
        long_result = inject_retune(
            long,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )

        short_err = abs(short_result.science_mask.sum() / len(short_result.times) - expected)
        long_err = abs(long_result.science_mask.sum() / len(long_result.times) - expected)

        assert long_err <= short_err + 0.001, (
            f"Long trajectory error ({long_err:.4f}) should be <= "
            f"short trajectory error ({short_err:.4f})"
        )


def _group_retune_events(retune_times: np.ndarray) -> list[float]:
    """Group retune flag timestamps into distinct events by start time.

    Returns the start time of each distinct retune event.
    """
    if len(retune_times) == 0:
        return []

    events = [retune_times[0]]
    for i in range(1, len(retune_times)):
        # Gap > 0.2s means a new event
        if retune_times[i] - retune_times[i - 1] > 0.2:
            events.append(retune_times[i])
    return events


class TestZeroVelocityGuard:
    """N-6: defensive guard for zero-velocity + prefer_turnarounds=True.

    The turnaround-snapping path in ``inject_retune`` scans for
    ``SCAN_FLAG_TURNAROUND`` samples in the input trajectory, but it
    still relies on the assumption that the velocity profile is
    meaningful.  The PrimeCam wrapper historically supplies
    identically-zero velocities, which would silently collapse all
    turnaround detection and produce wrong results.  The guard warns
    and falls back to time-based placement so callers notice.
    """

    def test_warns_on_zero_velocities_with_turnaround_snap(self):
        """Zero velocities + prefer_turnarounds=True must warn and fall back."""
        # Build a 300s trajectory whose az/el velocities are exactly zero.
        duration = 300.0
        timestep = 0.1
        times = np.arange(0, duration, timestep)
        n = len(times)
        traj = Trajectory(
            times=times,
            az=np.linspace(100.0, 200.0, n),
            el=np.full(n, 45.0),
            az_vel=np.zeros(n),
            el_vel=np.zeros(n),
            scan_flag=np.full(n, SCAN_FLAG_SCIENCE, dtype=int),
        )

        with pytest.warns(PointingWarning, match="zero velocities"):
            result = inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                prefer_turnarounds=True,
            )

        # The fallback must still produce retune flags via the time-based
        # placement branch; the original scan_flag must not be mutated.
        retune_count = int((result.scan_flag == SCAN_FLAG_RETUNE).sum())
        assert retune_count > 0, "Time-based fallback should still place retune samples"
        # The input trajectory must remain unchanged (inject_retune is pure).
        assert traj.scan_flag is not None
        assert not (traj.scan_flag == SCAN_FLAG_RETUNE).any()

    def test_no_warning_when_prefer_turnarounds_false(self):
        """Zero velocities + prefer_turnarounds=False must NOT warn.

        Verifies the guard is scoped to the turnaround-snapping path and
        does not emit spurious warnings for the default time-based path.
        """
        import warnings as _warnings

        duration = 120.0
        timestep = 0.1
        times = np.arange(0, duration, timestep)
        n = len(times)
        traj = Trajectory(
            times=times,
            az=np.linspace(100.0, 200.0, n),
            el=np.full(n, 45.0),
            az_vel=np.zeros(n),
            el_vel=np.zeros(n),
            scan_flag=np.full(n, SCAN_FLAG_SCIENCE, dtype=int),
        )

        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                prefer_turnarounds=False,
            )

        matches = [
            r
            for r in records
            if issubclass(r.category, PointingWarning) and "zero velocities" in str(r.message)
        ]
        assert not matches, (
            f"Zero-velocity warning leaked into prefer_turnarounds=False path: {matches}"
        )

    def test_no_warning_with_real_velocities(self):
        """Real velocities + prefer_turnarounds=True must NOT warn about zero vel."""
        import warnings as _warnings

        traj = _make_long_trajectory(
            duration=300.0,
            timestep=0.1,
            turnaround_intervals=[(28.0, 31.0), (58.0, 61.0)],
        )

        # az_vel is computed from np.gradient, so it is nonzero.
        assert not np.all(traj.az_vel == 0.0)

        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                prefer_turnarounds=True,
            )

        matches = [
            r
            for r in records
            if issubclass(r.category, PointingWarning) and "zero velocities" in str(r.message)
        ]
        assert not matches, (
            f"Zero-velocity guard fired for trajectory with real velocities: {matches}"
        )
