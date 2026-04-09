"""Tests for scheduling data models."""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories.overhead.models import (
    BlockType,
    CalibrationPolicy,
    CalibrationSpec,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)


class TestObservingPatch:
    """Tests for ObservingPatch dataclass."""

    def test_dec_bounds(self):
        patch = ObservingPatch(
            name="test",
            ra_center=0.0,
            dec_center=-30.0,
            width=10.0,
            height=20.0,
            scan_type="constant_el",
            velocity=1.0,
        )
        assert patch.dec_min == -40.0
        assert patch.dec_max == -20.0

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="width must be positive"):
            ObservingPatch(
                name="bad",
                ra_center=0.0,
                dec_center=0.0,
                width=-1.0,
                height=1.0,
                scan_type="pong",
                velocity=1.0,
            )

    def test_invalid_scan_type(self):
        with pytest.raises(ValueError, match="scan_type"):
            ObservingPatch(
                name="bad",
                ra_center=0.0,
                dec_center=0.0,
                width=1.0,
                height=1.0,
                scan_type="invalid",
                velocity=1.0,
            )

    def test_invalid_velocity(self):
        with pytest.raises(ValueError, match="velocity must be positive"):
            ObservingPatch(
                name="bad",
                ra_center=0.0,
                dec_center=0.0,
                width=1.0,
                height=1.0,
                scan_type="pong",
                velocity=-1.0,
            )


class TestCalibrationSpec:
    """Tests for CalibrationSpec dataclass."""

    def test_valid_types(self):
        for name in ("retune", "pointing_cal", "focus", "skydip", "planet_cal", "beam_map"):
            spec = CalibrationSpec(name=name, duration=10.0)
            assert spec.name == name

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown calibration type"):
            CalibrationSpec(name="invalid", duration=10.0)

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="duration must be positive"):
            CalibrationSpec(name="retune", duration=-1.0)


class TestTimelineBlock:
    """Tests for TimelineBlock dataclass."""

    def test_duration(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(300, format="sec")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t1,
            block_type="science",
            patch_name="test",
            az_min=100.0,
            az_max=200.0,
            elevation=50.0,
            scan_index=0,
        )
        assert abs(block.duration - 300.0) < 0.01

    def test_invalid_block_type(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        with pytest.raises(ValueError, match="block_type"):
            TimelineBlock(
                t_start=t0,
                t_stop=t0,
                block_type="invalid",
                patch_name="test",
                az_min=0.0,
                az_max=0.0,
                elevation=0.0,
                scan_index=0,
            )

    def test_string_block_type_coerced(self):
        """String block_type should be coerced to BlockType enum."""
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(60, format="sec"),
            block_type="science",
            patch_name="test",
            az_min=0.0,
            az_max=0.0,
            elevation=0.0,
            scan_index=0,
        )
        assert block.block_type is BlockType.SCIENCE
        # Equality with the string value must still hold (str subclass).
        assert block.block_type == "science"

    def test_enum_block_type(self):
        """Passing the enum directly should work without coercion."""
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(60, format="sec"),
            block_type=BlockType.CALIBRATION,
            patch_name="test",
            az_min=0.0,
            az_max=0.0,
            elevation=0.0,
            scan_index=0,
        )
        assert block.block_type is BlockType.CALIBRATION

    def test_boresight_angle_default(self):
        """boresight_angle defaults to 0.0."""
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(60, format="sec"),
            block_type="science",
            patch_name="test",
            az_min=0.0,
            az_max=0.0,
            elevation=0.0,
            scan_index=0,
        )
        assert block.boresight_angle == 0.0

    def test_boresight_angle_stored(self):
        """boresight_angle is preserved when provided."""
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(60, format="sec"),
            block_type="science",
            patch_name="test",
            az_min=0.0,
            az_max=0.0,
            elevation=0.0,
            scan_index=0,
            boresight_angle=42.5,
        )
        assert block.boresight_angle == 42.5


class TestBlockType:
    """Tests for the BlockType enum."""

    def test_values(self):
        assert BlockType.SCIENCE.value == "science"
        assert BlockType.CALIBRATION.value == "calibration"
        assert BlockType.SLEW.value == "slew"
        assert BlockType.IDLE.value == "idle"

    def test_str(self):
        assert str(BlockType.SCIENCE) == "science"
        assert str(BlockType.IDLE) == "idle"

    def test_equals_string(self):
        """BlockType inherits from str so == string comparisons work."""
        assert BlockType.SCIENCE == "science"
        assert BlockType.CALIBRATION == "calibration"


class TestOverheadModel:
    """Tests for OverheadModel dataclass."""

    def test_negative_duration(self):
        with pytest.raises(ValueError, match="non-negative"):
            OverheadModel(retune_duration=-1.0)

    def test_min_ge_max(self):
        with pytest.raises(ValueError, match="min_scan_duration"):
            OverheadModel(min_scan_duration=4000.0, max_scan_duration=3600.0)

    def test_get_calibration_duration(self):
        model = OverheadModel(retune_duration=5.0, focus_duration=300.0)
        assert model.get_calibration_duration("retune") == 5.0
        assert model.get_calibration_duration("focus") == 300.0

    def test_get_calibration_duration_unknown(self):
        model = OverheadModel()
        with pytest.raises(ValueError, match="not a valid CalibrationType"):
            model.get_calibration_duration("invalid")


class TestCalibrationPolicy:
    """Tests for CalibrationPolicy dataclass."""

    def test_negative_cadence(self):
        with pytest.raises(ValueError, match="non-negative"):
            CalibrationPolicy(pointing_cadence=-1.0)


class TestObservingTimeline:
    """Tests for ObservingTimeline dataclass."""

    def test_empty_timeline(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(3600, format="sec")
        tl = ObservingTimeline(
            blocks=[],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        assert tl.n_science_scans == 0
        assert tl.efficiency == 0.0
        assert tl.total_science_time == 0.0

    def test_efficiency(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(1000, format="sec")
        science = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(500, format="sec"),
            block_type="science",
            patch_name="test",
            az_min=100.0,
            az_max=200.0,
            elevation=50.0,
            scan_index=0,
        )
        cal = TimelineBlock(
            t_start=t0 + TimeDelta(500, format="sec"),
            t_stop=t0 + TimeDelta(600, format="sec"),
            block_type="calibration",
            patch_name="retune",
            az_min=100.0,
            az_max=100.0,
            elevation=50.0,
            scan_index=0,
        )
        tl = ObservingTimeline(
            blocks=[science, cal],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        assert tl.n_science_scans == 1
        assert abs(tl.efficiency - 0.5) < 0.01

    def test_validate_clean(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(1000, format="sec")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(500, format="sec"),
            block_type="science",
            patch_name="test",
            az_min=100.0,
            az_max=200.0,
            elevation=50.0,
            scan_index=0,
        )
        tl = ObservingTimeline(
            blocks=[block],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        warnings = tl.validate()
        assert warnings == []

    def test_validate_overlap(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(1000, format="sec")
        b1 = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(600, format="sec"),
            block_type="science",
            patch_name="a",
            az_min=100.0,
            az_max=200.0,
            elevation=50.0,
            scan_index=0,
        )
        b2 = TimelineBlock(
            t_start=t0 + TimeDelta(500, format="sec"),
            t_stop=t0 + TimeDelta(800, format="sec"),
            block_type="science",
            patch_name="b",
            az_min=100.0,
            az_max=200.0,
            elevation=50.0,
            scan_index=1,
        )
        tl = ObservingTimeline(
            blocks=[b1, b2],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        warnings = tl.validate()
        assert len(warnings) == 1
        assert "Overlap" in warnings[0]
