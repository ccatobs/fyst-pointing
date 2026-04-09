"""Scheduling test fixtures."""

import pytest
from astropy.time import Time

from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingPatch,
    OverheadModel,
)


@pytest.fixture
def start_time():
    """Nighttime observation start at FYST (UTC, ~22:00 local)."""
    return Time("2026-06-15T02:00:00", scale="utc")


@pytest.fixture
def end_time():
    """Observation end time (8 hours after start)."""
    return Time("2026-06-15T10:00:00", scale="utc")


@pytest.fixture
def sample_patch():
    """Provide a Pong scan test patch centered at RA=180, Dec=-30."""
    return ObservingPatch(
        name="test_field",
        ra_center=180.0,
        dec_center=-30.0,
        width=4.0,
        height=4.0,
        scan_type="pong",
        velocity=0.5,
    )


@pytest.fixture
def ce_patch():
    """Provide a constant-elevation test patch for Deep56."""
    return ObservingPatch(
        name="ce_test",
        ra_center=24.0,
        dec_center=-32.0,
        width=40.0,
        height=10.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )


@pytest.fixture
def overhead_model():
    """Provide the default overhead model."""
    return OverheadModel()


@pytest.fixture
def calibration_policy():
    """Provide the default calibration policy."""
    return CalibrationPolicy()
