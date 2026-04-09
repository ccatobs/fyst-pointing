"""Tests for inject_retune() trajectory utility."""

import numpy as np

from fyst_trajectories.trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    Trajectory,
)
from fyst_trajectories.trajectory_utils import inject_retune


def _make_trajectory(
    duration: float = 120.0,
    timestep: float = 0.1,
    turnaround_intervals: list[tuple[float, float]] | None = None,
) -> Trajectory:
    """Create a simple trajectory for testing.

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
    scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=int)

    if turnaround_intervals:
        for t_start, t_end in turnaround_intervals:
            mask = (times >= t_start) & (times < t_end)
            scan_flag[mask] = SCAN_FLAG_TURNAROUND

    return Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel, scan_flag=scan_flag)


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
        scan_flag = np.full(3, SCAN_FLAG_SCIENCE, dtype=int)
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
        import pytest

        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, module_index=7, n_modules=7)

        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, module_index=-1, n_modules=7)

    def test_staggered_invalid_n_modules(self):
        """n_modules < 1 should raise ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        import pytest

        with pytest.raises(ValueError, match="n_modules"):
            inject_retune(traj, n_modules=0)
