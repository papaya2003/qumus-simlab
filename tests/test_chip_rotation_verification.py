"""Ordered-corner physical orientation is modulo360, never square symmetry."""
from copy import deepcopy
from math import cos, radians, sin
from types import SimpleNamespace

import pytest

from qumus_simlab.core.types import Pose
from qumus_simlab.flows.library.search import MICROSCOPE, STAGE, MeasureChipRotation
from qumus_simlab.flows.step import Verdict
from qumus_simlab.perception.base import CornerDetection


class ChipStore:
    def __init__(self, chip_id, desired):
        self.sample = {"id": chip_id, "kind": "chip", "props": {"size_mm": [10., 10.],
            "pose_on_stage": {"x": 0., "y": 0., "rotation": desired}, "corner_stage_xy": [3., 7.]}}

    def get_sample(self, chip_id):
        assert chip_id == self.sample["id"]
        return deepcopy(self.sample)

    def upsert_sample(self, chip_id, kind, *, props):
        assert chip_id == self.sample["id"] and kind == "chip"
        self.sample["props"].update(props)


def measured_case(chip_id, desired, observed, edge=10., *, missing=False):
    """Only public stage/corner readbacks; no world or motion adapter."""
    store, stage_pose = ChipStore(chip_id, desired), Pose(x=.3, y=.4)
    frame = object()
    calls = []
    def capture():
        calls.append("capture")
        return frame
    def find_corner(actual_frame, corner):
        assert actual_frame is frame and corner == "br"
        if missing:
            return None
        # These are observed world/stage offsets for the ordered BL→BR edge.
        angle = radians(observed)
        return CornerDetection(corner="br", confidence=1.,
            dx_mm=stage_pose.x-3.+edge*cos(angle), dy_mm=stage_pose.y-7.+edge*sin(angle))
    gate = SimpleNamespace(
        stage=lambda identifier: SimpleNamespace(get_sample_pose=lambda: stage_pose) if identifier == STAGE else None,
        microscope=lambda identifier: SimpleNamespace(capture=capture) if identifier == MICROSCOPE else None)
    context = SimpleNamespace(store=store, gate=gate, analyzer=SimpleNamespace(find_corner=find_corner),
        bench={"stations": {STAGE: {"x": 0., "y": 0.}}})
    return context, calls


@pytest.mark.parametrize("chip_id", ["source_chip", "receiver_chip"])
@pytest.mark.parametrize("desired,observed", [
    (125.5, 125.4722), (284.5, -75.5217), (-75.5, -75.5217),
    (359., -1.01), (-179.99, 179.99), (180., -180.), (720.5, .48),
])
def test_shared_verifier_accepts_only_full_turn_equivalence_and_stores_observed_angle(chip_id, desired, observed):
    ctx, captures = measured_case(chip_id, desired, observed)
    step = MeasureChipRotation(chip_id, name=f"measure_design_rotation_{chip_id}")
    result = step.verify(ctx)
    assert result.verdict == Verdict.OK
    assert result.measurements["rotation_deg"] == pytest.approx(observed, abs=.00005)
    assert result.expected["camera_rotation_deg"] == desired
    assert result.expected["max_difference_deg"] == 3.
    assert result.expected["edge_tolerance_mm"] == .3
    step.on_success(ctx, result)
    assert ctx.store.sample["props"]["rotation_deg"] == result.measurements["rotation_deg"]
    assert ctx.store.sample["props"]["pose_on_stage"]["rotation"] == desired
    assert captures == ["capture"]


@pytest.mark.parametrize("desired,observed,edge", [
    (284.5, -72.48, 10.), (0., 3.01, 10.), (359., 2.01, 10.),
    (0., 90., 10.), (0., 180., 10.), (284.5, -75.5217, 10.31),
])
def test_wraparound_never_relaxes_angular_or_edge_tolerances(desired, observed, edge):
    ctx, _ = measured_case("chip", desired, observed, edge)
    before = deepcopy(ctx.store.sample)
    result = MeasureChipRotation("chip").verify(ctx)
    assert result.verdict == Verdict.UNCLEAR
    assert ctx.store.sample == before


def test_missing_ordered_corner_remains_correctable():
    ctx, captures = measured_case("chip", 284.5, -75.5, missing=True)
    result = MeasureChipRotation("chip").verify(ctx)
    assert result.verdict == Verdict.CORRECTABLE
    assert result.measurements == {"found": False}
    assert captures == ["capture"]
