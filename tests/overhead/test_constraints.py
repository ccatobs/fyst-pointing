"""Tests for scheduling constraints."""

import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead.constraints import (
    ElevationConstraint,
    MinDurationConstraint,
    MoonAvoidanceConstraint,
    SunAvoidanceConstraint,
)
from fyst_trajectories.overhead.models import ObservingPatch


@pytest.fixture
def patch():
    return ObservingPatch(
        name="test",
        ra_center=180.0,
        dec_center=-30.0,
        width=4.0,
        height=4.0,
        scan_type="pong",
        velocity=0.5,
    )


@pytest.fixture
def coords():
    return Coordinates(get_fyst_site())


@pytest.fixture
def time():
    return Time("2026-06-15T04:00:00", scale="utc")


class TestElevationConstraint:
    """Tests for ElevationConstraint."""

    def test_in_range(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 50.0, coords) == 1.0

    def test_below_min(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 20.0, coords) == 0.0

    def test_above_max(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 85.0, coords) == 0.0

    def test_at_boundary(self, patch, time, coords):
        c = ElevationConstraint(el_min=30.0, el_max=80.0)
        assert c.score(patch, time, 180.0, 30.0, coords) == 1.0
        assert c.score(patch, time, 180.0, 80.0, coords) == 1.0

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="el_min"):
            ElevationConstraint(el_min=80.0, el_max=30.0)


class TestSunAvoidanceConstraint:
    """Tests for SunAvoidanceConstraint."""

    def test_safe_position(self, patch, time, coords):
        c = SunAvoidanceConstraint(min_angle=45.0)
        sun_az, sun_el = coords.get_sun_altaz(time)
        safe_az = (sun_az + 180.0) % 360.0
        score = c.score(patch, time, safe_az, 50.0, coords)
        assert score == 1.0

    def test_near_sun(self, patch, coords):
        daytime = Time("2026-06-15T16:00:00", scale="utc")
        c = SunAvoidanceConstraint(min_angle=45.0)
        sun_az, sun_el = coords.get_sun_altaz(daytime)
        if sun_el > 10.0:
            score = c.score(patch, daytime, sun_az, sun_el, coords)
            assert score == 0.0
        else:
            pytest.skip("Sun not above horizon at test time")

    def test_negative_angle(self):
        with pytest.raises(ValueError, match="non-negative"):
            SunAvoidanceConstraint(min_angle=-1.0)


class TestMoonAvoidanceConstraint:
    """Tests for MoonAvoidanceConstraint."""

    def test_negative_angle(self):
        with pytest.raises(ValueError, match="non-negative"):
            MoonAvoidanceConstraint(min_angle=-1.0)


class TestMinDurationConstraint:
    """Tests for MinDurationConstraint."""

    def test_sufficient_duration(self, patch, time, coords):
        c = MinDurationConstraint(min_duration=60.0)
        az, el = coords.radec_to_altaz(180.0, -30.0, time)
        if el > 30.0:
            score = c.score(patch, time, az, el, coords)
            assert score == 1.0

    def test_negative_duration(self):
        with pytest.raises(ValueError, match="non-negative"):
            MinDurationConstraint(min_duration=-1.0)
