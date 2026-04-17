"""Pytest fixtures for fyst-trajectories tests.

Notes
-----
Hard-coded observation times in tests (e.g. ``Time("2026-03-15T04:00:00")``)
should remain within the IERS Earth Orientation Parameter prediction window
(typically ~1 year from the current date).  If tests start producing IERS
warnings or degraded accuracy, update the test dates forward.
"""

import pytest

from fyst_trajectories import Coordinates, get_fyst_site


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


# The ``slow`` marker is registered in pyproject.toml under
# ``[tool.pytest.ini_options].markers``; no ``pytest_configure`` hook
# needed here.


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def site():
    """Provide a default FYST site for testing."""
    return get_fyst_site()


@pytest.fixture
def coordinates(site):
    """Provide a Coordinates instance for testing."""
    return Coordinates(site)
