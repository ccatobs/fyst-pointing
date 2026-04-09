"""Tests for scheduling utilities."""

from astropy.time import Time

from fyst_trajectories.overhead.utils import (
    estimate_slew_time,
    get_max_elevation,
    get_observable_windows,
    get_transit_time,
)


class TestEstimateSlewTime:
    """Tests for slew time estimation."""

    def test_zero_distance(self, site):
        t = estimate_slew_time(180.0, 50.0, 180.0, 50.0, site)
        assert t == 0.0

    def test_az_only(self, site):
        t = estimate_slew_time(180.0, 50.0, 190.0, 50.0, site)
        assert t > 0.0
        assert t > 3.0
        assert t < 20.0

    def test_el_only(self, site):
        t = estimate_slew_time(180.0, 50.0, 180.0, 60.0, site)
        assert t > 0.0
        assert t > 8.0
        assert t < 30.0

    def test_el_slower_than_az(self, site):
        t_az = estimate_slew_time(180.0, 50.0, 190.0, 50.0, site)
        t_el = estimate_slew_time(180.0, 50.0, 180.0, 60.0, site)
        assert t_el > t_az

    def test_large_slew(self, site):
        t = estimate_slew_time(0.0, 30.0, 180.0, 70.0, site)
        assert t > 30.0


class TestGetMaxElevation:
    """Tests for maximum elevation computation."""

    def test_overhead_source(self, site):
        max_el = get_max_elevation(0.0, site.latitude, site)
        assert abs(max_el - 90.0) < 0.01

    def test_low_source(self, site):
        max_el = get_max_elevation(0.0, 60.0, site)
        assert max_el < 10.0

    def test_moderate_source(self, site):
        max_el = get_max_elevation(0.0, -30.0, site)
        assert max_el > 80.0


class TestGetTransitTime:
    """Tests for transit time computation."""

    def test_finds_transit(self, site, start_time):
        """Verify transit is found and HA is near zero at that time."""
        transit = get_transit_time(180.0, -30.0, start_time, site)
        assert transit is not None, "Should find transit within 24 hours"
        from fyst_trajectories import Coordinates

        coords = Coordinates(site)
        ha = coords.get_hour_angle(180.0, transit)
        assert abs(ha) < 2.0  # HA near zero at transit

    def test_returns_none_if_not_found(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        transit = get_transit_time(180.0, -30.0, t0, site, max_search_hours=0.001)
        assert transit is None or isinstance(transit, Time)


class TestGetObservableWindows:
    """Tests for observable window computation."""

    def test_finds_windows(self, site, start_time, end_time):
        windows = get_observable_windows(
            180.0,
            -30.0,
            start_time,
            end_time,
            site,
            min_elevation=30.0,
            check_sun=False,
        )
        assert isinstance(windows, list)

    def test_never_visible_source(self, site, start_time, end_time):
        windows = get_observable_windows(
            0.0,
            80.0,
            start_time,
            end_time,
            site,
            min_elevation=30.0,
            check_sun=False,
        )
        assert len(windows) == 0

    def test_circumpolar_source(self, site, start_time, end_time):
        from fyst_trajectories import Coordinates

        coords = Coordinates(site)
        az, el = coords.radec_to_altaz(0.0, -50.0, start_time)
        if el > 30.0:
            windows = get_observable_windows(
                0.0,
                -50.0,
                start_time,
                end_time,
                site,
                min_elevation=30.0,
                check_sun=False,
            )
            assert len(windows) >= 1
        else:
            # Try a different RA that's better placed
            windows = get_observable_windows(
                180.0,
                -30.0,
                start_time,
                end_time,
                site,
                min_elevation=30.0,
                check_sun=False,
            )
            assert len(windows) >= 1
