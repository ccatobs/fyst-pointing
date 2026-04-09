"""Tests for the timeline generator."""

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ElevationConstraint,
    ObservingPatch,
    OverheadModel,
    SunAvoidanceConstraint,
    generate_timeline,
)


def assert_timeline_valid(timeline, site):
    blocks = sorted(timeline.blocks, key=lambda b: b.t_start.unix)

    for i in range(len(blocks) - 1):
        assert blocks[i].t_stop.unix <= blocks[i + 1].t_start.unix + 0.1, (
            f"Overlap: '{blocks[i].patch_name}' ends at {blocks[i].t_stop.iso} "
            f"but '{blocks[i + 1].patch_name}' starts at {blocks[i + 1].t_start.iso}"
        )

    for b in blocks:
        assert b.t_start.unix >= timeline.start_time.unix - 0.1
        assert b.t_stop.unix <= timeline.end_time.unix + 0.1

    for b in blocks:
        if b.block_type == "science":
            assert b.elevation >= site.telescope_limits.elevation.min - 0.1
            assert b.elevation <= site.telescope_limits.elevation.max + 0.1

    if timeline.n_science_scans > 0:
        assert timeline.efficiency > 0.0
        assert timeline.efficiency <= 1.0


class TestGenerateTimeline:
    """Tests for generate_timeline()."""

    def test_single_patch_one_night(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test_field",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
        )
        assert_timeline_valid(timeline, site)
        assert timeline.n_science_scans > 0
        assert len(timeline.calibration_blocks) > 0

    def test_no_patches(self):
        site = get_fyst_site()
        timeline = generate_timeline(
            patches=[],
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T04:00:00",
        )
        assert timeline.n_science_scans == 0
        idle_blocks = [b for b in timeline.blocks if b.block_type == "idle"]
        assert len(idle_blocks) > 0

    def test_multiple_patches_priority(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="high_priority",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
                priority=1.0,
            ),
            ObservingPatch(
                name="low_priority",
                ra_center=200.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
                priority=10.0,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
        )
        assert_timeline_valid(timeline, site)

        high_time = sum(
            b.duration for b in timeline.science_blocks if b.patch_name == "high_priority"
        )
        low_time = sum(
            b.duration for b in timeline.science_blocks if b.patch_name == "low_priority"
        )
        assert high_time >= low_time

    def test_calibration_blocks_present(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
            calibration_policy=CalibrationPolicy(retune_cadence=0.0),
        )
        cal_types = {b.scan_type for b in timeline.calibration_blocks}
        assert "retune" in cal_types

    def test_max_scan_duration_splits(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        overhead = OverheadModel(max_scan_duration=600.0)  # 10 min max
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
            overhead_model=overhead,
        )
        assert_timeline_valid(timeline, site)
        for b in timeline.science_blocks:
            assert b.duration <= overhead.max_scan_duration + 1.0

    def test_sun_avoidance_respected(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        constraints = [
            ElevationConstraint(el_min=30.0, el_max=80.0),
            SunAvoidanceConstraint(min_angle=45.0),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
            constraints=constraints,
        )
        assert_timeline_valid(timeline, site)

        coords = Coordinates(site)
        for b in timeline.science_blocks:
            mid_time = b.t_start + (b.t_stop - b.t_start) / 2
            az_mid = (b.az_min + b.az_max) / 2.0
            assert coords.is_sun_safe(az_mid, b.elevation, mid_time)

    def test_constant_el_patch(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="ce_test",
                ra_center=24.0,
                dec_center=-32.0,
                width=40.0,
                height=10.0,
                scan_type="constant_el",
                velocity=1.0,
                elevation=50.0,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
        )
        assert_timeline_valid(timeline, site)

    def test_short_timeline(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T02:10:00",
        )
        assert_timeline_valid(timeline, site)

    def test_validate_method(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
        )
        warnings = timeline.validate()
        assert warnings == []
