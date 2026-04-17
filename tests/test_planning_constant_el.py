"""Tests for plan_constant_el_scan."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.patterns.configs import ConstantElScanConfig
from fyst_trajectories.planning import FieldRegion, ScanBlock, plan_constant_el_scan
from fyst_trajectories.planning._ce_geometry import (
    _compute_ce_az_range,
    _compute_ce_duration,
)


class TestPlanConstantElScan:
    """Tests for plan_constant_el_scan."""

    @pytest.fixture
    def ecdfs_field(self):
        """E-CDF-S field region used in the scan strategy script."""
        return FieldRegion(
            ra_center=53.117,
            dec_center=-27.808,
            width=5.0,
            height=6.7,
        )

    @pytest.fixture
    def search_time(self):
        """Provide a base search time for E-CDF-S CE scans."""
        return Time("2026-03-15T17:00:00", scale="utc")

    def test_basic_plan(self, site, ecdfs_field, search_time):
        """plan_constant_el_scan returns a ScanBlock with correct types."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, ConstantElScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert "az_start" in block.computed_params
        assert "az_stop" in block.computed_params
        assert "az_throw" in block.computed_params
        assert "n_scans" in block.computed_params
        assert "Constant-El scan" in block.summary

    def test_elevation_in_trajectory(self, site, ecdfs_field, search_time):
        """Trajectory should be at the requested elevation."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert np.allclose(block.trajectory.el, 50.0)

    def test_rising_vs_setting_different_times(self, site, ecdfs_field, search_time):
        """Rising and setting passes should have different start times."""
        rising = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )
        setting = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=False,
            angle=170.0,
        )

        assert rising.computed_params["start_time_iso"] != setting.computed_params["start_time_iso"]

    def test_string_start_time(self, site, ecdfs_field):
        """start_time as a string should work."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time="2026-03-15T17:00:00",
            rising=True,
            angle=170.0,
        )

        assert block.duration > 0

    def test_azimuth_throw_positive(self, site, ecdfs_field, search_time):
        """Computed azimuth throw should be positive."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert block.computed_params["az_throw"] > 0

    def test_unreachable_field_raises(self, site):
        """A field that never reaches the target elevation should raise."""
        # Dec = +70 is never reachable at el=50 from FYST
        field = FieldRegion(ra_center=180.0, dec_center=70.0, width=1.0, height=1.0)
        with pytest.raises(ValueError, match="Could not find elevation crossing"):
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-15T00:00:00", scale="utc"),
                rising=True,
            )

    def test_science_mask_excludes_turnarounds(self, site, ecdfs_field, search_time):
        """CE scan trajectory should have turnaround samples excluded by science_mask."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        traj = block.trajectory
        assert traj.scan_flag is not None
        science = traj.science_mask
        # science_mask should exclude some turnaround samples
        assert science.sum() < traj.n_points
        # But the majority should be science
        assert science.sum() > traj.n_points * 0.5


class TestCEGeometryWrapHandling:
    """Regression tests for the RA = 0 / az = 0/360 wrap handling.

    Both bugs were documented in ``_ce_geometry.py`` as known edge cases.
    The audit (``docs/reviews/methodology_audit.md`` Per-Area C Finding 1)
    flagged the azimuth-wrap case as plausible at FYST's −23° latitude
    for sources that transit through north (dec ≳ +20°).
    """

    def test_az_range_handles_north_transit(self, site):
        """``_compute_ce_az_range`` returns a contiguous range for north-transiting sources.

        At FYST (lat = −22.99°), a source at RA = 0°, dec = +35° transits
        through north around 04:50 UTC on 2026-09-16, with corner azimuths
        straddling the 0/360 discontinuity. The naive ``min``/``max``
        computation reports a ~358° throw; the unwrapped result should
        be a few-degree throw matching the field width.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=0.0, dec_center=35.0, width=4.0, height=4.0)
        obs_start = Time("2026-09-16T04:30:00", scale="utc")
        obs_end = Time("2026-09-16T05:10:00", scale="utc")

        az_min, az_max = _compute_ce_az_range(
            field,
            angle=0.0,
            coords_obj=coords,
            obs_start=obs_start,
            obs_end=obs_end,
            padding=0.5,
        )

        throw = az_max - az_min
        # The field is 4° wide; the temporal sweep adds ~10° of azimuth
        # variation as it transits. A wrapped (broken) result would be
        # close to 358°.
        assert throw < 30.0, f"az_throw {throw:.2f}° suggests az-wrap was not handled"
        assert throw > field.width

    def test_az_range_normal_field_unchanged(self, site):
        """``_compute_ce_az_range`` is unchanged for fields away from the discontinuity.

        A southern-hemisphere field that transits well away from north
        should never trigger the wrap-detection branch.
        """
        coords = Coordinates(site)
        # ECDFS-like field; pick an obs window when it's actually visible
        # so the per-time azimuth values are coherent.
        field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        obs_start = Time("2026-03-15T08:30:00", scale="utc")
        obs_end = Time("2026-03-15T09:30:00", scale="utc")
        az_min, az_max = _compute_ce_az_range(
            field,
            angle=170.0,
            coords_obj=coords,
            obs_start=obs_start,
            obs_end=obs_end,
            padding=0.5,
        )
        # Throw should be modest and well inside [0, 360)
        assert 0.0 <= az_min < 360.0
        assert 0.0 <= az_max < 360.0
        assert (az_max - az_min) < 60.0

    def test_ra_wrap_handled_for_field_near_ra_zero(self, site):
        """``_compute_ce_duration`` correctly identifies edges for RA ≈ 0 fields.

        A 3°-wide field centred at RA = 1° has corners at RA ≈ −0.5°
        and ≈ +2.5°. After ``% 360``, naive ``min``/``max`` would return
        the wrong leading/trailing edges. The wrap-detection branch
        should re-centre the values around the field centre.
        """
        coords = Coordinates(site)
        # Use a southern-hemisphere field that transits comfortably
        # above the requested elevation at FYST (lat = −23°).
        field = FieldRegion(ra_center=1.0, dec_center=-25.0, width=3.0, height=3.0)
        # Pick a search start time before the field rises through el=40
        # (transit happens ~03:30 UTC at this RA on this date).
        base_time = Time("2026-09-15T00:00:00", scale="utc")

        t_start, t_end, duration = _compute_ce_duration(
            field,
            angle=0.0,
            elevation=40.0,
            coords_obj=coords,
            base_search_time=base_time,
            rising=True,
        )

        # Without the wrap fix, ``min(ra_vals)``/``max(ra_vals)`` would be
        # 2.5 and 359.5; the leading edge would be searched at RA = 359.5
        # (which crosses el=40 hours later than the true RA = -0.5 edge)
        # and the reported duration would be a few-hour overestimate.
        # With the fix, the duration is the short interval between the
        # two true RA edges crossing el=40.
        assert duration > 0
        assert duration < 30 * 60  # 30 min — true value ~10 min for 3° width

    def test_north_transit_planning_succeeds(self, site):
        """End-to-end: ``plan_constant_el_scan`` works for a north-transiting source.

        Without the az-wrap fix, ``_compute_ce_az_range`` returns a ~358°
        throw, which overflows the configured azimuth range and either
        crashes downstream validation or produces a nonsense scan.
        """
        field = FieldRegion(ra_center=0.0, dec_center=35.0, width=2.0, height=2.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=30.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-09-16T04:30:00", scale="utc"),
            rising=False,
            angle=0.0,
        )
        # az_throw must be small (matches field width plus temporal sweep)
        assert block.computed_params["az_throw"] < 30.0
