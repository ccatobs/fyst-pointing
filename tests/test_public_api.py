"""Public API lock-in tests for the planning surface.

These tests pin the import paths and object identities that downstream
consumers (``scan_patterns``, ``primecam_camera_mapping_simulations``)
rely on. A regression in ``planning/__init__.py`` or the top-level
``fyst_trajectories/__init__.py`` re-exports will fail here.
"""


def test_planning_public_api():
    """Planning subpackage exposes the five public symbols."""
    from fyst_trajectories.planning import (
        FieldRegion,
        ScanBlock,
        plan_constant_el_scan,
        plan_daisy_scan,
        plan_pong_scan,
    )

    assert FieldRegion is not None
    assert ScanBlock is not None
    assert plan_constant_el_scan is not None
    assert plan_daisy_scan is not None
    assert plan_pong_scan is not None


def test_top_level_public_api():
    """Top-level :mod:`fyst_trajectories` re-exports the planning symbols."""
    from fyst_trajectories import (
        FieldRegion,
        ScanBlock,
        plan_constant_el_scan,
        plan_daisy_scan,
        plan_pong_scan,
    )

    assert FieldRegion is not None
    assert ScanBlock is not None
    assert plan_constant_el_scan is not None
    assert plan_daisy_scan is not None
    assert plan_pong_scan is not None


def test_identity_across_import_paths():
    """Symbols imported from both paths are the same object."""
    from fyst_trajectories import FieldRegion as F1
    from fyst_trajectories import ScanBlock as S1
    from fyst_trajectories import plan_constant_el_scan as pce1
    from fyst_trajectories import plan_daisy_scan as pda1
    from fyst_trajectories import plan_pong_scan as pp1
    from fyst_trajectories.planning import FieldRegion as F2
    from fyst_trajectories.planning import ScanBlock as S2
    from fyst_trajectories.planning import plan_constant_el_scan as pce2
    from fyst_trajectories.planning import plan_daisy_scan as pda2
    from fyst_trajectories.planning import plan_pong_scan as pp2

    assert F1 is F2
    assert S1 is S2
    assert pp1 is pp2
    assert pce1 is pce2
    assert pda1 is pda2
