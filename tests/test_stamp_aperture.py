"""Saved-pixel aperture fits and bounded Gate-controlled scans; no hardware."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys

import pytest
from PIL import Image, ImageDraw

from qumus_simlab.config import load_yaml
from qumus_simlab.core.errors import DangerousError, NeedsHumanError, PreconditionFailed
from qumus_simlab.flows.aperture import (
    LEGACY_STRUCTURE_PROFILE,
    STRUCTURE_PROFILE,
    ApertureScan,
    CenterStampAperture,
    aperture_edges,
    fit_aperture,
    validate_aperture_alignment,
)
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.step import Verdict
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.safety.gate import SafetyGate
from qumus_simlab.testbench.world import SimFlake
from tests.test_white_light_optics import optical_lab


def raster_records(tmp_path, *, center=(.25, -.17), axes=("+world_x", "-world_y"), occluded=False, shape="diamond", size=(320, 214)):
    records = []
    width, height = size
    sx, sy = 2.6/width, 1.72/height
    offset = 4.7/math.sqrt(2)
    positions = ((offset, offset), (-offset, -offset), (offset, -offset), (-offset, offset)) if shape == "diamond" else ((4.7, 0), (-4.7, 0), (0, 4.7), (0, -4.7))
    for index, (hx, hy) in enumerate(positions):
        image = Image.new("RGB", (width, height), (184, 179, 163))
        points = []
        local = ((math.sqrt(2)*5, 0), (0, math.sqrt(2)*5), (-math.sqrt(2)*5, 0), (0, -math.sqrt(2)*5)) if shape == "diamond" else [(5*math.cos(n*math.tau/720), 5*math.sin(n*math.tau/720)) for n in range(720)]
        for x, y in local:
            wx, wy = center[0]+hx+x, center[1]+hy+y
            # Invert each declared signed axis; an axis swap is also covered.
            coord = {"x": wx, "y": wy}
            u = coord[axes[0][-1]]*(1 if axes[0].startswith("+") else -1)/sx
            v = coord[axes[1][-1]]*(1 if axes[1].startswith("+") else -1)/sy
            points.append((width/2+u, height/2+v))
        ImageDraw.Draw(image).polygon(points, fill=(122, 96, 168))
        if occluded and index == 3:
            ImageDraw.Draw(image).rectangle((0, 0, width, height), fill=(184, 179, 163))
        path = tmp_path / f"arc-{index}.png"
        image.save(path)
        records.append({"image": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "holder_xy_mm": [hx, hy], "camera_world_mm": [740, 420],
            "calibration": {"source": "synthetic", "image_size_px": [width, height],
                "mm_per_pixel": [sx, sy], "u_axis": axes[0], "v_axis": axes[1],
                "photometry_profile": {"stamp_structure": STRUCTURE_PROFILE if shape == "diamond" else LEGACY_STRUCTURE_PROFILE}},
            "stamp_structure": {"aperture_center_mm": [999, -999]}, "first_contact_center_mm": [-123, 456]})
    return records


@pytest.mark.parametrize("size", [(320, 214), (480, 320)])
@pytest.mark.parametrize("border", ["right", "bottom"])
def test_annotations_just_outside_roi_cannot_become_observed_aperture_edges(tmp_path, size, border):
    records = raster_records(tmp_path, size=size)
    original_points = [aperture_edges(record) for record in records]
    original_fit = fit_aperture(records, (0, 0))
    for record in records:
        with Image.open(record["image"]) as source:
            image = source.copy()
        width, height = image.size
        # Both endpoints of a boundary comparison must stay inside the same
        # annotation-free ROI. White text at the first excluded row/column
        # must not turn adjacent paper pixels into an apparent aperture edge.
        line = ((width-8, 0, width-8, height-1) if border == "right"
                else (0, height-32, width-1, height-32))
        ImageDraw.Draw(image).line(line, fill=(245, 245, 245))
        image.save(record["image"])
        with open(record["image"], "rb") as source:
            record["sha256"] = hashlib.sha256(source.read()).hexdigest()
    assert [aperture_edges(record) for record in records] == original_points
    assert fit_aperture(records, (0, 0)) == original_fit


@pytest.mark.parametrize("axes", [("+world_x", "-world_y"), ("-world_x", "+world_y"), ("+world_y", "+world_x")])
def test_observed_diamond_edges_fit_signed_mm_axes_and_ignore_hidden_centres(tmp_path, axes):
    records = raster_records(tmp_path, axes=axes)
    result = fit_aperture(records, (0, 0))
    assert result["center_offset_mm"] == pytest.approx([.25, -.17], abs=.012)
    assert result["inradius_mm"] == pytest.approx(5, abs=.012)
    assert result["shape"] == "diamond" and len(result["edge_lines"]) == 2
    assert 0 < result["uncertainty_mm"] < .025
    changed = copy.deepcopy(records)
    for record in changed:
        record["stamp_structure"]["aperture_center_mm"] = [0, 0]
        record["first_contact_center_mm"] = result["center_offset_mm"]
    assert fit_aperture(changed, (0, 0)) == result


def test_legacy_circle_raster_fit_remains_available_only_with_legacy_profile(tmp_path):
    records = raster_records(tmp_path, shape="circle")
    result = fit_aperture(records, (0, 0), shape="circle")
    assert result["center_offset_mm"] == pytest.approx([.25, -.17], abs=.012)
    assert result["radius_mm"] == pytest.approx(5, abs=.012)
    with pytest.raises(ValueError, match="shape/profile"):
        fit_aperture(records, (0, 0), shape="diamond")


@pytest.mark.parametrize("change", ["occluded", "missing", "tampered", "calibration"])
def test_missing_obscured_or_unbound_aperture_edges_fail_closed(tmp_path, change):
    records = raster_records(tmp_path, occluded=change == "occluded")
    if change == "missing":
        records.pop()
    elif change == "tampered":
        with open(records[0]["image"], "ab") as handle:
            handle.write(b"changed evidence")
    elif change == "calibration":
        records[0]["calibration"]["mm_per_pixel"] = [1, 1]
    with pytest.raises(ValueError):
        fit_aperture(records, (0, 0))


def test_inconsistent_visible_boundaries_cannot_claim_a_diamond(tmp_path):
    records = raster_records(tmp_path)
    with Image.open(records[0]["image"]) as original:
        image = original.copy()
    ImageDraw.Draw(image).rectangle((0, 70, image.width, 105), fill=(184, 179, 163))
    image.save(records[0]["image"])
    with open(records[0]["image"], "rb") as source:
        records[0]["sha256"] = hashlib.sha256(source.read()).hexdigest()
    with pytest.raises(ValueError, match="occluded|ambiguous|Observed"):
        fit_aperture(records, (0, 0))


def scan_context(tmp_path):
    lab, clock, _, scope = optical_lab(tmp_path / "images")
    scope.renderer.size = (320, 214)
    lab.world.model["stamp"]["structure"]["aperture_offset_mm"] = [.25, -.17]
    stamp = lab.world.stamps["stamp_01"]
    stamp.apex_dx, stamp.apex_dy, stamp.optic_shift = .08, -.04, .3
    lab.world.objects["chip_001"].flakes.append(SimFlake("focus_fixture", "Gr", 1, 0, 0, 20000,
        outline_mm=((- .08, -.05), (.08, -.05), (0, .09))))
    store = RunStore(tmp_path / "run.db", clock=clock)
    run = store.create_run("stack_device", {})
    gate = SafetyGate(lab.devices, clock=clock, sink=lambda k, p: store.emit(k, p, run_id=run))
    gate.manipulator("transfer_arm").move_axis_to("z", 6.5)
    gate.microscope("microscope").set_objective(5)
    gate.microscope("microscope").set_focus(.3)
    ctx = FlowContext(run, store, gate, clock, lab.locator, lab.bench, load_yaml("verification.yaml"),
                      params={"device_design": {"fixture": "explicit synthetic test"}, "stamp_id": "stamp_01"}, analyzer=lab.analyzer)
    return ctx, lab


@pytest.fixture(scope="module")
def complete_scan(tmp_path_factory):
    ctx, lab = scan_context(tmp_path_factory.mktemp("diamond-aperture"))
    step = CenterStampAperture("stamp_01")
    step.act(ctx, None)
    return ctx, lab, step


def test_gate_scan_centres_observed_aperture_distinct_from_contact_apex(complete_scan):
    ctx, lab, step = complete_scan
    observation = step.verify(ctx)
    assert observation.verdict == Verdict.OK
    result = observation.measurements
    assert len(result["frames"]) == 9
    assert result["initial_fit"]["center_offset_mm"] == pytest.approx([.25, -.17], abs=.012)
    assert result["centered_holder_xy_mm"] == pytest.approx([39.75, 90.17], abs=.012)
    assert ctx.store.get_run(ctx.run_id)["meta"]["normal_transfer"]["aperture_alignment"] == result
    commands = [e for e in ctx.store.events_after(0) if e.kind == "device.command" and e.payload["device"] == "transfer_arm"]
    assert commands and all(e.payload["params"].get("axis") in ("x", "y", "z") for e in commands)
    assert all(frame["event_id"] > 0 and frame["calibration"] for frame in result["frames"])
    assert not ctx.store.get_run(ctx.run_id)["meta"]["normal_transfer"].get("placements")
    assert (lab.world.stamps["stamp_01"].apex_dx, lab.world.stamps["stamp_01"].apex_dy) != (.25, -.17)
    assert validate_aperture_alignment(ctx.store, ctx.run_id, result) == result
    changed = copy.deepcopy(result)
    changed["frames"][0]["holder_xy_mm"][0] += .1
    with pytest.raises(ValueError, match="receipt changed"):
        validate_aperture_alignment(ctx.store, ctx.run_id, changed)


@pytest.mark.parametrize("change", ["deleted_capture", "capture_payload", "capture_timestamp",
                                   "deleted_focus", "focus_payload", "focus_timestamp"])
def test_original_capture_and_focus_events_are_required_not_only_derived_receipts(complete_scan, change):
    ctx, _, step = complete_scan
    body = step.result
    ident = body["wafer_focus_event_id"] if "focus" in change else body["frames"][0]["event_id"]
    class Rollback(Exception):
        pass
    with pytest.raises(Rollback), ctx.store._transaction():
        if change.startswith("deleted"):
            ctx.store._x("DELETE FROM event WHERE id=?", (ident,))
        elif change.endswith("timestamp"):
            ctx.store._x("UPDATE event SET ts=ts+1 WHERE id=?", (ident,))
        else:
            payload = copy.deepcopy(ctx.store.events_after(ident-1, limit=1)[0].payload)
            payload["changed_original_evidence"] = True
            ctx.store._x("UPDATE event SET payload_json=? WHERE id=?", (json.dumps(payload), ident))
        with pytest.raises(ValueError, match="capture event|refocus observation"):
            validate_aperture_alignment(ctx.store, ctx.run_id, body)
        raise Rollback


def test_aperture_module_imports_in_fresh_interpreter_without_library_import_order():
    result = subprocess.run([sys.executable, "-c", "from qumus_simlab.flows.aperture import CenterStampAperture; print(CenterStampAperture.__name__)"],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CenterStampAperture"


def test_new_scan_refuses_circle_policy_even_though_legacy_pixel_replay_is_supported(tmp_path):
    ctx, _ = scan_context(tmp_path)
    with pytest.raises(PreconditionFailed, match="diamond opening"):
        CenterStampAperture("stamp_01", config=ApertureScan(shape="circle")).act(ctx, None)


def test_rejected_swept_clearance_never_starts_lateral_scan(tmp_path):
    ctx, _ = scan_context(tmp_path)
    ctx.gate.limits.geometry.objective_bottom_at_focus_zero_z = 7.0
    before = ctx.gate.manipulator("transfer_arm").get_positions()
    with pytest.raises(DangerousError):
        CenterStampAperture("stamp_01").act(ctx, None)
    assert ctx.gate.manipulator("transfer_arm").get_positions() == before


def test_scan_refuses_low_holder_before_motion(tmp_path):
    ctx, lab = scan_context(tmp_path)
    lab.world.manip.z = 5.4
    with pytest.raises(PreconditionFailed, match="clear height"):
        CenterStampAperture("stamp_01").act(ctx, None)


def test_interrupted_scan_cannot_be_replayed_blindly(tmp_path):
    ctx, _ = scan_context(tmp_path)
    ctx.store.update_run_meta(ctx.run_id, normal_transfer={"aperture_alignment": {"status": "in_progress"}})
    with pytest.raises(NeedsHumanError, match="review"):
        CenterStampAperture("stamp_01").act(ctx, None)


def test_finite_scan_capture_budget_stops_before_unbounded_retry(tmp_path):
    ctx, _ = scan_context(tmp_path)
    with pytest.raises(NeedsHumanError, match="capture count exhausted"):
        CenterStampAperture("stamp_01", config=ApertureScan(max_captures=4)).act(ctx, None)
    result = ctx.store.get_run(ctx.run_id)["meta"]["normal_transfer"]["aperture_alignment"]
    assert len(result["frames"]) == 4 and result["status"] != "observed"


def _redraw(record, draw):
    with Image.open(record["image"]) as source:
        image = source.copy()
    draw(ImageDraw.Draw(image))
    image.save(record["image"])
    with open(record["image"], "rb") as source:
        record["sha256"] = hashlib.sha256(source.read()).hexdigest()


def test_paper_coloured_chip_speck_is_not_an_aperture_edge(tmp_path):
    """A small chip speck fixed under a moving holder must not bend the fit (#126)."""
    records = raster_records(tmp_path, size=(480, 320))
    clean = fit_aperture(records, (0, 0))
    for record in records:
        # The chip stays put while the holder scans: same pixels in every exposure.
        _redraw(record, lambda draw: draw.rectangle((349, 101, 350, 102), fill=(184, 179, 163)))
        _redraw(record, lambda draw: draw.rectangle((120, 150, 121, 151), fill=(122, 96, 168)))
    result = fit_aperture(records, (0, 0))
    assert result["max_residual_mm"] <= clean["max_residual_mm"] + 1e-9
    assert result["center_offset_mm"] == pytest.approx(clean["center_offset_mm"], abs=1e-9)


def test_large_paper_occlusion_still_refuses_the_diamond(tmp_path):
    records = raster_records(tmp_path, size=(480, 320))
    # 12x12 px is above the speck size and lies across the visible edge region.
    _redraw(records[0], lambda draw: draw.rectangle((0, 120, 479, 131), fill=(184, 179, 163)))
    with pytest.raises(ValueError, match="occluded|ambiguous|Observed"):
        fit_aperture(records, (0, 0))
