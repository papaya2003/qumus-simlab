"""White illumination is reported; all numerical spectral response is synthetic."""
# Imported pytest fixtures are injected by name.
# ruff: noqa: F811

from __future__ import annotations

import copy
import io
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from qumus_simlab.config import load_yaml
from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library.stacking import (
    PROCEDURE_ID,
    TRANSFER_PROFILE_ID,
    _frame_record,
    _phase_capture,
)
from qumus_simlab.safety.gate import SafetyGate
from qumus_simlab.testbench.lab import build_sim_lab
from qumus_simlab.testbench.optics import WHITE_LIGHT_PROFILE, white_light_response
from qumus_simlab.testbench.persist import mechanics_signature
from qumus_simlab.testbench.render import Canvas, MicroscopeRenderer
from qumus_simlab.testbench.world import StampState
from tests.conftest import display_matches, quiet_bench


def geometry(gap, *, radius=1.5, contact=0., dx=0., dy=0.):
    return {"signed_apex_gap_mm": gap, "dome_radius_mm": radius,
            "centre_dx_mm": dx, "centre_dy_mm": dy,
            "current_contact_radius_mm": contact, "has_contact": gap < 0,
            "profile_id": WHITE_LIGHT_PROFILE}


def projected(gap, *, contact=0., radius=1.5):
    canvas = Canvas(480, 320, (160, 160, 160))
    MicroscopeRenderer._white_light_rings(canvas, geometry(gap, contact=contact, radius=radius), 480/.65, 320/.43)
    return Image.frombytes("RGB", (480, 320), bytes(canvas.px))


def optical_lab(tmp_path):
    clock = SimClock()
    lab = build_sim_lab(clock, config=quiet_bench(), image_dir=tmp_path)
    scope = lab.devices.get("microscope")
    chip = lab.world.objects["chip_001"]
    chip.pose, chip.on, chip.held_by = Pose(x=scope.center[0], y=scope.center[1]), "transfer_stage", None
    chip.flakes.clear()
    chip.focus_offset = 0
    lab.world.objects["stamp_01"].held_by = "transfer_arm"
    lab.world.stamps["stamp_01"] = StampState(5.3, 0, 0, 165)
    origin = lab.bench["transfer_arm"]["origin"]
    lab.world.manip.x, lab.world.manip.y = scope.center[0]-origin["x"], scope.center[1]-origin["y"]
    lab.world.manip.z = 5.3002
    lab.devices.get("transfer_arm").sigma = 0
    gate = SafetyGate(lab.devices, clock=clock)
    gate.microscope("microscope").set_objective(20)
    return lab, clock, gate, scope


def test_broadband_response_is_coloured_continuous_and_decoheres():
    values = [white_light_response(i*1e-6) for i in range(1200)]
    assert any(max(v[:3])-min(v[:3]) > 15 for v in values)
    assert max(abs(a-b) for g in (.0001, .0003, .0006)
               for a, b in zip(white_light_response(g), white_light_response(g+1e-9), strict=True)) < .01
    # rings first show near a 2.8 um apex gap and are below one RGB8 step at the table end
    assert white_light_response(.0028)[3] > .03 > white_light_response(.0035)[3]
    assert 255*white_light_response(.008)[3] < .1
    assert all(0 <= v[3] <= .45 and all(0 <= c <= 255*v[3] for c in v[:3]) for v in values)


def test_same_gap_is_stable_and_gap_or_radius_changes_actual_native_raster():
    original = projected(.00015)
    assert original.tobytes() == projected(.00015).tobytes()
    assert ImageChops.difference(original, projected(.00020)).getbbox()
    assert ImageChops.difference(original, projected(.00015, radius=2.)).getbbox()
    pixels = list(original.get_flattened_data())
    assert len(set(pixels)) > 100
    assert any(max(p)-min(p) > 5 for p in pixels)


def test_outer_annuli_remain_after_contact_without_drawing_inside_contact():
    contact = .04
    image = projected(-.0002, contact=contact)
    background = Image.new("RGB", image.size, (160, 160, 160))
    assert image.crop((230, 150, 250, 170)).tobytes() == background.crop((230, 150, 250, 170)).tobytes()
    # Positive air gap exists beyond sqrt(2*R*depth), independently of the
    # transient contact radius; the renderer must retain that outer annulus.
    x = round(240+math.sqrt(2*1.5*.0005)*480/.65)
    assert image.getpixel((x, 160)) != (160, 160, 160)


def test_optical_packet_comes_from_same_projection_and_preview_resolves_small_changes(tmp_path):
    lab, clock, gate, scope = optical_lab(tmp_path)
    frame = scope.capture()
    optical = frame.meta["optical_gap"]
    apex = lab.world.apex()
    assert optical == geometry(-apex[3], dx=apex[2][0]-scope.center[0], dy=apex[2][1]-scope.center[1])
    signature = scope.preview_signature()
    # Ten nanometres is far below the generic half-pixel position quantum.
    gate.manipulator("transfer_arm").move_axis_to("z", lab.world.manip.z-.00001)
    assert scope.preview_signature() != signature
    assert scope.capture().meta["optical_gap"]["signed_apex_gap_mm"] < optical["signed_apex_gap_mm"]
    gate.manipulator("transfer_arm").move_axis_to("z", 5.2998)
    clock.sleep(.2)
    contact = scope.capture().meta["optical_gap"]
    assert contact["has_contact"] and contact["signed_apex_gap_mm"] < 0 and contact["current_contact_radius_mm"] > 0
    signature = scope.preview_signature()
    clock.sleep(.2)  # Existing Clock-driven front/drift, never renderer animation.
    assert scope.preview_signature() != signature


def test_capture_save_and_preview_are_frozen_and_rendering_does_not_advance_physics(tmp_path):
    lab, clock, _, scope = optical_lab(tmp_path)
    frame = scope.capture()
    before = pickle.dumps((lab.world.evaluation_snapshot(), lab.world.rng.getstate(),
                           lab.world._observation_rng.getstate(), clock.now()))
    frozen, data = copy.deepcopy(frame.meta), frame.data
    signature, mechanics = scope.preview_signature(), mechanics_signature(lab)
    for _ in range(3):
        assert scope.renderer.render_bytes("microscope", frozen) == data
        assert scope.preview_signature() == signature
        for preview in (scope.preview(), scope.preview_snapshot()):
            assert preview.meta["optical_gap"] == frozen["optical_gap"]
            assert preview.meta["calibration"] == frozen["calibration"]
            assert display_matches(Path(preview.saved_path()).read_bytes(), data)
    assert Path(frame.saved_path()).read_bytes() == data
    assert frame.meta == frozen and mechanics_signature(lab) == mechanics
    assert pickle.dumps((lab.world.evaluation_snapshot(), lab.world.rng.getstate(),
                         lab.world._observation_rng.getstate(), clock.now())) == before
    changed_timestamp = {**frozen, "captured_at": frozen["captured_at"]+50}
    assert scope.renderer.render_bytes("microscope", changed_timestamp) == data
    assert Image.open(io.BytesIO(data)).size == (480, 320)


def test_unchanged_twin_polls_reuse_preview_without_new_image_files(tmp_path):
    lab, _, _, _ = optical_lab(tmp_path)
    lab.twin.delta()
    paths = {p: p.read_bytes() for p in tmp_path.rglob("*.*g")}
    assert paths
    for _ in range(5):
        delta = lab.twin.delta()
        assert not delta or "microscope_preview" not in delta
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.*g")} == paths


@pytest.mark.parametrize("legacy", ["missing", "previous"])
def test_old_fringe_profile_refuses_without_relabelling_or_holder_motion(tmp_path, monkeypatch, legacy):
    lab, clock, gate, scope = optical_lab(tmp_path)
    original = scope.capture()
    pixels, meta = Path(original.saved_path()).read_bytes(), copy.deepcopy(original.meta)
    old = copy.deepcopy(meta)
    if legacy == "missing":
        old["calibration"]["photometry_profile"].pop("fringes")
    else:
        old["calibration"]["photometry_profile"]["fringes"] = "fixed-palette-v0"
    reference = Frame(original.device_id, original.timestamp, path=original.saved_path(), meta=old)
    phase = {"name": "fixed_reference_profile", "profile": TRANSFER_PROFILE_ID,
             "started_at": clock.now(), "observed_at": original.timestamp,
             "observations": 1, "reference": _frame_record(reference), "difference_active": True}
    frozen_reference = copy.deepcopy(phase["reference"])
    state = {}
    store = SimpleNamespace(bench="sim", get_run=lambda _: {"meta": state},
        update_run_meta=lambda _, **fields: state.update(fields), emit=lambda *args, **kwargs: None)
    ctx = SimpleNamespace(run_id="profile_guard", clock=clock, gate=gate, store=store,
        bench=lab.bench, params={"procedure_contract": PROCEDURE_ID, "transfer_profile": TRANSFER_PROFILE_ID},
        verification=load_yaml("verification.yaml"), check_cancel=lambda: None)
    monkeypatch.setattr(lab.devices.get("transfer_arm"), "move_axis_to",
                        lambda *args, **kwargs: pytest.fail("profile refusal must precede holder motion"))
    with pytest.raises(NeedsHumanError, match="calibration/profile/pose/focus changed"):
        _phase_capture(ctx, phase)
    assert phase["held_reason"] == "capture_binding_changed"
    assert Path(original.saved_path()).read_bytes() == pixels and original.meta == meta
    assert phase["reference"] == frozen_reference and reference.meta == old
