"""Tests for plan_daisy_scan."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns.configs import DaisyScanConfig
from fyst_trajectories.planning import ScanBlock, plan_daisy_scan


@pytest.fixture
def start_time():
    """Provide a standard start time when the target is observable."""
    return Time("2026-03-15T04:00:00", scale="utc")


class TestPlanDaisyScan:
    """Tests for plan_daisy_scan."""

    def test_basic_plan(self, site, start_time):
        """plan_daisy_scan returns a ScanBlock with daisy config."""
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, DaisyScanConfig)
        assert block.duration == pytest.approx(60.0)
        assert block.trajectory.n_points > 0
        assert "Daisy scan" in block.summary

    def test_trajectory_has_valid_bounds(self, site, start_time):
        """Generated trajectory must stay within telescope elevation limits."""
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        traj = block.trajectory
        limits = site.telescope_limits
        assert traj.el.min() >= limits.elevation.min
        assert traj.el.max() <= limits.elevation.max

    def test_unobservable_target_raises(self, site, start_time):
        """Test that an unobservable target raises TargetNotObservableError."""
        with pytest.raises(TargetNotObservableError):
            plan_daisy_scan(
                ra=180.0,
                dec=80.0,  # Not visible from FYST
                radius=0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                site=site,
                start_time=start_time,
                timestep=0.1,
                duration=60.0,
            )

    def test_with_detector_offset(self, site, start_time):
        """Test that detector offset is applied."""
        block_no_offset = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        block_with_offset = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
            detector_offset=offset,
        )

        # Trajectories should differ when offset is applied
        assert not np.allclose(block_no_offset.trajectory.az, block_with_offset.trajectory.az)
