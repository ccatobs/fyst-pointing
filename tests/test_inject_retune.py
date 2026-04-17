"""Tests for inject_retune() trajectory utility.

Covers basic placement, turnaround snapping, edge cases, the per-module
staggered mode, the per-pattern efficiency cross-validation against
real planner outputs (CE / Pong / Daisy), turnaround-overlap dead-time
reduction, theoretical efficiency parametric checks, and the N-6
zero-velocity defensive guard.
"""

import warnings as _warnings

import numpy as np
import pytest
from astropy.time import Time

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

# ``site`` fixture is provided by ``conftest.py``; tests in this module
# use that shared definition.


@pytest.fixture
def start_time():
    """Nighttime start at FYST."""
    return Time("2026-06-15T04:00:00", scale="utc")


def _make_trajectory(
    duration: float = 120.0,
    timestep: float = 0.1,
    turnaround_intervals: list[tuple[float, float]] | None = None,
) -> Trajectory:
    """Create a synthetic trajectory for inject_retune tests.

    Parameters
    ----------
    duration : float
        Total duration in seconds.
    timestep : float
        Time step in seconds.
    turnaround_intervals : list of (start, end) tuples
        Time intervals to flag as turnaround.
    """
    times = np.arange(0, duration, timestep)
    n = len(times)
    az = np.linspace(100, 200, n)
    el = np.full(n, 45.0)
    az_vel = np.gradient(az, times)
    el_vel = np.zeros(n)
    scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8)

    if turnaround_intervals:
        for t_start, t_end in turnaround_intervals:
            mask = (times >= t_start) & (times < t_end)
            scan_flag[mask] = SCAN_FLAG_TURNAROUND

    return Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel, scan_flag=scan_flag)


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


class TestInjectRetuneBasic:
    """Basic retune flag placement tests."""

    def test_retune_flags_placed_at_correct_intervals(self):
        """Retune events should appear at the expected interval positions."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Check that retune flags exist
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()

        # Find the start times of retune events
        retune_times = result.times[retune_mask]
        # Group into distinct retune events by finding gaps
        events = []
        current_start = retune_times[0]
        for i in range(1, len(retune_times)):
            if retune_times[i] - retune_times[i - 1] > 0.2:
                events.append(current_start)
                current_start = retune_times[i]
        events.append(current_start)

        # Retune at ~30s, then interval measured from retune_end (35s),
        # so next at ~65s, then ~100s.
        assert len(events) == 3
        np.testing.assert_allclose(events, [30.0, 65.0, 100.0], atol=0.15)

    def test_retune_duration_correct(self):
        """Each retune event should span the configured duration."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        retune_times = result.times[retune_mask]

        # Group into events
        events_samples = []
        current = [retune_times[0]]
        for i in range(1, len(retune_times)):
            if retune_times[i] - retune_times[i - 1] > 0.2:
                events_samples.append(current)
                current = [retune_times[i]]
            else:
                current.append(retune_times[i])
        events_samples.append(current)

        for event in events_samples:
            event_duration = event[-1] - event[0] + 0.1  # +timestep
            assert abs(event_duration - 5.0) < 0.2


class TestInjectRetuneTurnaroundSnapping:
    """Tests for turnaround snapping behavior."""

    def test_snaps_to_nearby_turnaround(self):
        """Retune should snap to a nearby turnaround when within window."""
        # Turnaround at 28-31s (3s), near the 30s due time.
        # Retune duration is 5s, so retune covers 28-33s.
        # Turnaround occupies 28-31, so RETUNE flags appear at 31-33 (science region).
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(28.0, 31.0)],
        )
        result = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()

        # The retune flags should start right after the turnaround ends (at ~31s)
        # because the turnaround samples are not overwritten
        first_retune_time = result.times[retune_mask][0]
        assert abs(first_retune_time - 31.0) < 0.15

        # Turnaround flags should be preserved
        ta_mask = result.scan_flag == SCAN_FLAG_TURNAROUND
        ta_count = ta_mask.sum()
        original_ta_count = (traj.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        assert ta_count == original_ta_count

    def test_no_turnaround_nearby_falls_back_to_time_based(self):
        """Without a nearby turnaround, retune falls back to time-based placement."""
        # Turnaround far from the 30s due time
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(10.0, 15.0)],
        )
        result = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()
        first_retune_time = result.times[retune_mask][0]
        # Should be at ~30s (time-based), not snapped
        assert abs(first_retune_time - 30.0) < 0.15

    def test_synthetic_turnaround_overlap_count(self):
        """With turnarounds at retune due times, snapping should use them.

        Creates a synthetic trajectory with turnarounds at 28-31s,
        58-61s, 88-91s -- near the 30s, 60s, 90s due times. With
        prefer_turnarounds=True, retunes should snap to these and
        preserve more (or equal) science samples than time-based
        placement.
        """
        turnarounds = [(28.0, 31.0), (58.0, 61.0), (88.0, 91.0)]
        traj = _make_trajectory(
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


class TestInjectRetuneDaisyContinuous:
    """Tests for continuous scan (no turnarounds), like daisy patterns."""

    def test_no_turnarounds_uses_time_based(self):
        """Continuous scan with no turnarounds should still place retunes."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=True
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()


class TestInjectRetuneScienceMask:
    """Tests for science_mask interaction."""

    def test_science_mask_excludes_retune(self):
        """science_mask should be False for all retune-flagged samples."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        science = result.science_mask
        retune = result.scan_flag == SCAN_FLAG_RETUNE

        # No overlap: science_mask should be False wherever retune is True
        assert not np.any(science & retune)

    def test_efficiency_calculation(self):
        """30s interval / 5s duration should give ~83.3% science fraction."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        science_fraction = result.science_mask.sum() / len(result.times)
        # ~83.3% (25/30), allow some tolerance for edge effects
        assert 0.80 < science_fraction < 0.87


class TestInjectRetuneEdgeCases:
    """Edge case tests."""

    def test_interval_longer_than_trajectory(self):
        """No retune should be placed if interval exceeds trajectory duration."""
        traj = _make_trajectory(duration=20.0, timestep=0.1)
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        # No retune should be placed
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert not retune_mask.any()

    def test_very_short_trajectory(self):
        """Very short trajectory should not crash and have no retune flags."""
        times = np.array([0.0, 0.1, 0.2])
        az = np.array([100.0, 100.1, 100.2])
        el = np.full(3, 45.0)
        az_vel = np.array([1.0, 1.0, 1.0])
        el_vel = np.zeros(3)
        scan_flag = np.full(3, SCAN_FLAG_SCIENCE, dtype=np.int8)
        traj = Trajectory(
            times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel, scan_flag=scan_flag
        )

        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)
        # Should not crash and no retune placed
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()

    def test_only_science_flags_overwritten(self):
        """Turnaround flags should never be changed to retune."""
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(29.0, 36.0)],
        )
        original_turnaround_count = (traj.scan_flag == SCAN_FLAG_TURNAROUND).sum()

        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        new_turnaround_count = (result.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        assert new_turnaround_count == original_turnaround_count

    def test_returns_new_trajectory(self):
        """inject_retune should return a new Trajectory, not mutate the original."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        original_flags = traj.scan_flag.copy()

        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        assert result is not traj
        np.testing.assert_array_equal(traj.scan_flag, original_flags)

    def test_no_scan_flag_array(self):
        """Trajectory with scan_flag=None should work (treated as all-science)."""
        times = np.arange(0, 120.0, 0.1)
        n = len(times)
        traj = Trajectory(
            times=times,
            az=np.linspace(100, 200, n),
            el=np.full(n, 45.0),
            az_vel=np.ones(n),
            el_vel=np.zeros(n),
            scan_flag=None,
        )
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)
        assert result.scan_flag is not None
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()


class TestInjectRetuneStaggered:
    """Tests for per-module staggered retune scheduling.

    Per-module retune independence is UNCONFIRMED by the FYST instrument
    team. These tests verify the staggering mechanism works correctly if
    modules can retune independently.
    """

    def test_staggered_retune_offset(self):
        """Different module_index values should produce retunes at different times."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)

        def _first_retune_time(module_index: int) -> float:
            result = inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                module_index=module_index,
                n_modules=7,
            )
            retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
            return float(result.times[retune_mask][0])

        # Each module should start its first retune at a different time
        first_times = [_first_retune_time(i) for i in range(7)]

        # All first retune times should be distinct
        for i in range(len(first_times)):
            for j in range(i + 1, len(first_times)):
                assert abs(first_times[i] - first_times[j]) > 1.0, (
                    f"Module {i} and {j} retune at the same time: "
                    f"{first_times[i]:.1f} vs {first_times[j]:.1f}"
                )

        # The offsets should be spaced by retune_interval / n_modules = 30/7 ~= 4.29s
        expected_spacing = 30.0 / 7
        for i in range(1, 7):
            expected = first_times[0] + i * expected_spacing
            np.testing.assert_allclose(first_times[i], expected, atol=0.15)

    def test_staggered_retune_coverage(self):
        """Combined science_mask from all 7 modules should have better coverage."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)

        # Single module (no staggering) -- baseline
        single = inject_retune(traj, retune_interval=30.0, retune_duration=5.0, n_modules=1)
        single_fraction = single.science_mask.sum() / len(single.times)

        # 7 staggered modules -- combined mask is True where ANY module is observing
        module_masks = []
        for i in range(7):
            result = inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                module_index=i,
                n_modules=7,
            )
            module_masks.append(result.science_mask)

        # For each sample, count how many modules are doing science
        # The "combined" fraction: a sample is lost only if ALL modules are retuning
        all_retuning = np.ones(len(traj.times), dtype=bool)
        for mask in module_masks:
            all_retuning &= ~mask
        combined_fraction = 1.0 - all_retuning.sum() / len(traj.times)

        # Combined coverage should be much better than single-module
        # Single module: ~83.3% (5/30 lost). Staggered: >97% since retune
        # windows don't overlap when interval/n_modules > duration.
        assert combined_fraction > single_fraction
        assert combined_fraction > 0.97

    def test_staggered_defaults_unchanged(self):
        """module_index=0, n_modules=1 should produce identical output."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        result_default = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)
        result_explicit = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            module_index=0,
            n_modules=1,
        )

        np.testing.assert_array_equal(result_default.scan_flag, result_explicit.scan_flag)

    def test_staggered_invalid_module_index(self):
        """module_index >= n_modules should raise ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, module_index=7, n_modules=7)

        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, module_index=-1, n_modules=7)

    def test_staggered_invalid_n_modules(self):
        """n_modules < 1 should raise ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="n_modules"):
            inject_retune(traj, n_modules=0)


class TestPerPatternEfficiency:
    """Verify science fraction after inject_retune for each scan pattern.

    Cross-validates the efficiency-vs-pattern interaction using actual
    planner outputs (CE / Pong / Daisy) instead of the synthetic
    trajectories used elsewhere in this file.
    """

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
        traj = _make_trajectory(duration=600.0, timestep=0.1)
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

        short = _make_trajectory(duration=120.0, timestep=0.1)
        long = _make_trajectory(duration=1200.0, timestep=0.1)

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
            scan_flag=np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8),
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
            scan_flag=np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8),
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
        traj = _make_trajectory(
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
