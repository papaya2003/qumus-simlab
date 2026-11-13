"""Motion event precision is independent of bounded immutable raster cadence."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import PreconditionFailed, SafetyViolation
from qumus_simlab.core.types import Pose
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.safety.gate import SafetyGate
from qumus_simlab.testbench.lab import build_sim_lab
from qumus_simlab.testbench.persist import lab_state, load_lab_state, save_lab_state
from qumus_simlab.testbench.world import SimObject
from tests.conftest import quiet_bench


def fixture(tmp_path):
    clock = SimClock()
    store = RunStore(":memory:", clock=clock)
    lab = build_sim_lab(clock, config=quiet_bench(stage__sigma_mm=0), image_dir=tmp_path)
    scope, stage = lab.devices.get("microscope"), lab.devices.get("transfer_stage")
    scope.renderer.size = (128, 84)
    lab.world.add(SimObject("motion_chip", "chip", Pose(x=scope.center[0], y=scope.center[1]),
                            on="transfer_stage", size_mm=(1.0, 1.0)))
    gate = SafetyGate(lab.devices, clock=clock, sink=store.emit, frames=lab.twin.delta)
    return lab, clock, store, gate, stage, scope


def test_unrelated_commands_and_short_focus_moves_do_not_create_redundant_pngs(tmp_path):
    lab, clock, store, gate, _, _ = fixture(tmp_path)
    gate.microscope("microscope").set_focus(-0.5)
    before = {p for p in tmp_path.rglob("*.png")}
    for _ in range(30):
        gate.heater("exfoliation_heater").off()
        gate.camera("overview_camera").capture()
    assert {p for p in tmp_path.rglob("*.png")} == before
    mark, start = store.last_event_id(), clock.now()
    for i in range(101):
        gate.microscope("microscope").set_focus(-0.5+i*0.01)
    events = store.events_after(mark, limit=10000)
    physical = [e for e in events if e.kind == "lab.frame" and "geometry" in e.payload]
    previews = [e.payload["microscope_preview"] for e in events if "microscope_preview" in e.payload]
    assert len(physical) >= 100
    assert len(previews) <= int((clock.now()-start)/0.5)+1
    assert all(p["immutable"] and p["verification"] is False for p in previews)
    assert len({p["image"] for p in previews}) == len(previews)
    assert all(Path(p["image"]).is_file() for p in previews)
    assert lab.twin.state()["geometry"]["microscope"]["focus_z"] == pytest.approx(0.5)


def test_interrupted_motion_checkpoints_last_actual_pose_on_existing_path(tmp_path):
    lab, clock, _, gate, stage, _ = fixture(tmp_path)
    original = lab.twin.delta
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("simulated worker interruption")
        return original()

    gate.frames = interrupt
    with pytest.raises(KeyboardInterrupt):
        gate.stage("transfer_stage").move_to(Pose(x=0, y=80))
    partial = stage.get_pose()
    assert 0 < partial.y < 80 and stage.motion_active is False
    checkpoint = tmp_path / "world.pkl"
    save_lab_state(lab, clock, checkpoint)
    fresh = build_sim_lab(SimClock(), config=quiet_bench(stage__sigma_mm=0))
    load_lab_state(fresh, fresh.devices.get("microscope").clock, checkpoint)
    assert fresh.devices.get("transfer_stage").get_pose() == partial
    assert fresh.world.surfaces["transfer_stage"].offset == lab.world.surfaces["transfer_stage"].offset


def test_fileless_bench_capture_has_frozen_projected_png_bytes(tmp_path):
    lab = build_sim_lab(SimClock(), config=quiet_bench())
    scope = lab.devices.get("microscope")
    assert scope.renderer.writer is None
    original = scope.capture()
    assert original.path is None and original.save is None
    assert original.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert original.data == scope.renderer.render_bytes("microscope", original.meta)
    before = original.data
    scope.set_focus(0.5)
    assert original.data == before and original.meta["focus_z"] == 0
    assert scope.preview().path is None and scope.preview_snapshot().path is None
    assert not list(tmp_path.iterdir())


def test_file_backed_capture_pixels_are_available_without_retaining_sweep_files(tmp_path):
    _, _, _, _, _, scope = fixture(tmp_path)
    before = set(tmp_path.rglob("*.png"))
    frames = [scope.capture() for _ in range(3)]
    assert all(f.data and f.path is None for f in frames)
    assert set(tmp_path.rglob("*.png")) == before
    captured = frames[0].data
    scope.set_focus(.2)
    saved = Path(frames[0].saved_path())
    assert saved.read_bytes() == captured
    assert frames[0].meta["focus_z"] == 0


def test_swept_clearance_uses_actual_translated_integrated_workcell():
    cfg = quiet_bench()
    cfg["reactor"] = {"enabled": True}
    clock = SimClock()
    lab = build_sim_lab(clock, config=cfg, store=RunStore(":memory:", clock=clock))
    gate = SafetyGate(lab.devices, clock=clock)
    assert lab.bench["microscope"]["center"]["x"] != gate.limits.geometry.objective_xy[0]
    holder = lab.world.manip
    over = lab.bench["transfer_arm"]["over_stage"]
    holder.x, holder.y, holder.z = over["x"], over["y"], 8.0
    # At the translated objective, -3 mm focus would sweep its housing (bottom
    # 12 - 3 = 9, limits.yaml) into the stamp (top 8 + 1). Base-file coordinates would incorrectly miss the collision.
    with pytest.raises(SafetyViolation, match="objective/stamp"):
        gate.microscope("microscope").set_focus(-3)
    assert lab.devices.get("microscope").get_focus() == 0


@pytest.mark.parametrize("old_motion_markers", [False, True])
def test_mid_observer_checkpoint_restores_pose_without_running_motion(tmp_path, old_motion_markers):
    lab, clock, _, gate, stage, _ = fixture(tmp_path)
    checkpoint = tmp_path / "midstroke.pkl"
    original = lab.twin.delta
    calls = 0

    def checkpoint_in_motion():
        nonlocal calls
        calls += 1
        if calls == 3:
            assert stage.motion_active
            save_lab_state(lab, clock, checkpoint)
            raise KeyboardInterrupt("interrupted after the observer checkpoint")
        return original()

    gate.frames = checkpoint_in_motion
    with pytest.raises(KeyboardInterrupt):
        gate.stage("transfer_stage").move_to(Pose(x=0, y=80))
    partial = stage.get_pose()
    assert 0 < partial.y < 80
    state = pickle.loads(checkpoint.read_bytes())
    assert "motion_active" not in state["devices"]["transfer_stage"]
    assert "last_motion_duration_s" not in state["devices"]["transfer_stage"]
    if old_motion_markers:
        # Earlier mechanics-v2 snapshots included these transient attributes;
        # their presence must never resume a controller or change the pose.
        state["devices"]["transfer_stage"].update(motion_active=True, last_motion_duration_s=8)
        checkpoint.write_bytes(pickle.dumps(state))
    saved_bytes = checkpoint.read_bytes()
    fresh_clock = SimClock()
    fresh = build_sim_lab(fresh_clock, config=quiet_bench(stage__sigma_mm=0))
    assert load_lab_state(fresh, fresh_clock, checkpoint)
    restored = fresh.devices.get("transfer_stage")
    assert restored.get_pose() == partial
    assert fresh.world.surfaces["transfer_stage"].offset == lab.world.surfaces["transfer_stage"].offset
    assert restored.motion_active is False and restored.last_motion_duration_s == 0
    assert all(not dev.motion_active for dev in fresh.devices.devices.values())
    fresh.devices.all_safe_state()
    assert restored.motion_active is False
    assert checkpoint.read_bytes() == saved_bytes


@pytest.mark.parametrize("section,field", [
    ("microscope", "best_focus_sigma"), ("microscope", "focus_tilt_per_mm"),
    ("microscope", "depth_of_focus_10x"), ("stamp", "dome_radius_mm"),
    ("stamp", "chip_thickness_mm"), ("stamp", "touch_z"),
    ("stamp", "optic_shift_mm"), ("stamp", "through_stamp_contrast"),
    ("stamp", "fringe_wavelength_mm"),
])
def test_checkpoint_rejects_changed_contact_surface_and_optical_model(tmp_path, section, field):
    original, clock, _, gate, _, _ = fixture(tmp_path)
    gate.stage("transfer_stage").move_fine_to(0.1, -0.2)
    checkpoint = tmp_path / "old_model.pkl"
    save_lab_state(original, clock, checkpoint)
    saved_bytes = checkpoint.read_bytes()
    fresh_clock = SimClock()
    fresh = build_sim_lab(fresh_clock, config=quiet_bench(stage__sigma_mm=0))
    fresh.world.model[section][field] += 0.001
    before = pickle.dumps(lab_state(fresh, fresh_clock))
    with pytest.raises(PreconditionFailed, match="incompatible stage/focus mechanics"):
        load_lab_state(fresh, fresh_clock, checkpoint)
    assert pickle.dumps(lab_state(fresh, fresh_clock)) == before
    assert checkpoint.read_bytes() == saved_bytes
