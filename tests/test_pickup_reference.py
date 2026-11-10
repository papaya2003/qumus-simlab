"""Current source-quality and coverage/endpoint distinctions; no G/K bootstrap."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from qumus_simlab.config import load_yaml
from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library.stacking import PickUpFlake, _require_source_plane
from qumus_simlab.flows.step import Verdict
from qumus_simlab.runtime.store import RunStore


@pytest.fixture
def endpoint():
    store = RunStore(":memory:", clock=SimClock(), bench="sim")
    try:
        run_id = store.create_run("stack_device", {})
        stage_state = {"fine_xy_mm": [0., 0.]}
        stage = SimpleNamespace(motion_active=False, kinematics=lambda: deepcopy(stage_state))
        holder = SimpleNamespace(motion_active=False)
        ctx = SimpleNamespace(store=store, run_id=run_id, verification=load_yaml("verification.yaml"),
            analyzer=SimpleNamespace(plane_sharpness=lambda frame, plane: frame.meta.get("sharpness_chip")),
            gate=SimpleNamespace(stage=lambda _: stage, manipulator=lambda _: holder))
        frame = Frame("microscope", 0., meta={"focus_plane": "chip", "objective": 20, "calibration": "unit-source-profile",
                     "sharpness_chip": 9., "stage_kinematics": deepcopy(stage_state)})
        yield ctx, PickUpFlake("source", "stamp_01", "Gr"), frame

    finally:
        store.close()


@pytest.mark.parametrize("quality", [None, 0., float("nan"), True])
def test_unknown_or_blurred_chip_cannot_borrow_bright_stamp_quality(endpoint, quality):
    ctx, _, frame = endpoint
    frame.meta.update(sharpness=99., sharpness_stamp=99., sharpness_chip=quality)
    with pytest.raises(NeedsHumanError, match="Source absence is unobservable"):
        _require_source_plane(ctx, frame, frame.meta)


def test_valid_source_quality_uses_captured_plane_without_world_query(endpoint):
    ctx, _, frame = endpoint
    _require_source_plane(ctx, frame, frame.meta)
    assert not hasattr(ctx, "world")


def test_changed_source_stage_or_unknown_quality_adapter_cannot_assert_absence(endpoint):
    ctx, _, frame = endpoint
    frame.meta["stage_kinematics"]["fine_xy_mm"] = [.01, 0.]
    with pytest.raises(NeedsHumanError, match="pose changed"):
        _require_source_plane(ctx, frame, frame.meta)
    frame.meta["stage_kinematics"]["fine_xy_mm"] = [0., 0.]
    ctx.analyzer.plane_sharpness = None
    with pytest.raises(NeedsHumanError, match="Source absence is unobservable"):
        _require_source_plane(ctx, frame, frame.meta)


@pytest.mark.parametrize("phase", [{}, {"coverage": {"covered": True}}, {"complete": True}])
def test_coverage_or_completion_alone_cannot_confirm_or_commit_transfer(endpoint, phase):
    ctx, step, _ = endpoint
    ctx.store.update_run_meta(ctx.run_id, normal_transfer={"phases": {step.name: phase}})
    before = ctx.store.list_samples()
    result = step.verify(ctx)
    assert result.verdict == Verdict.UNCLEAR and result.measurements["transfer_confirmed"] is False
    with pytest.raises(NeedsHumanError, match="Coverage alone"):
        step.on_success(ctx, result)
    assert ctx.store.list_samples() == before


def test_covered_endpoint_requires_original_native_source_evidence(endpoint):
    ctx, step, _ = endpoint
    ctx.store.update_run_meta(ctx.run_id, normal_transfer={"phases": {
        step.name: {"complete": True, "coverage": {"covered": True}}}})
    with pytest.raises(NeedsHumanError, match="captured native source reference"):
        step.verify(ctx)
    assert not ctx.store.list_samples()
