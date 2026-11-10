"""Paired20X producer controls with declared mask doubles; no World or jobs."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library import search
from qumus_simlab.perception import rgb_material
from qumus_simlab.records.recognition import validate_rgb_capture_binding, validate_rgb_outline
from qumus_simlab.runtime.store import RunStore


@pytest.fixture
def context(tmp_path, monkeypatch):
    clock = SimClock(start=123.)
    store = RunStore(":memory:", clock=clock, bench="sim")
    rid = store.create_run("search_chip", {"chip_id": "chip", "material": "Gr"})
    store.upsert_sample("chip", "chip", props={"rotation_deg": 15., "corner_stage_xy": [0., 0.]})
    store.upsert_sample("flake", "flake", location="chip", status="candidate", props={"material": "Gr", "stage_xy": [0., 0.]})
    rotation = store.emit("step.verified", {"name": "measure_chip_rotation", "verdict": "ok", "measurements": {"rotation_deg": 15.}}, run_id=rid)
    frames = {}
    for objective in (20, 100):
        path = tmp_path/f"native{objective}.png"
        Image.new("RGB", (128, 96), (142, 111, 110)).save(path)
        scale = 20/objective
        frames[objective] = Frame("microscope", clock.now(), path=str(path), meta={"objective": objective,
            "focus_z": 0., "focus_plane": "chip", "fov_mm": [.128*scale, .192*scale],
            "calibration": {"source": "declared_unit", "u_axis": "+world_x", "v_axis": "-world_y",
                "mm_per_pixel": [.001*scale, .002*scale], "image_size_px": [128, 96]}})
    labels = np.zeros((96, 128), dtype=np.uint16)
    labels[38:58, 54:74] = 2
    def declared(rgb, material, *, objective, image_sha256):
        return rgb_material.RGBRecognition(labels.copy(), ({"label_id": 2,
            "thickness_label": "monolayer" if objective == 20 else "bulk", "area_px": int((labels == 2).sum()),
            "centroid_px": [63.5, 47.5]},), rgb_material.material_policy(material, objective)[1], image_sha256, objective)
    monkeypatch.setattr(rgb_material, "segment_rgb", declared)
    ctx = SimpleNamespace(store=store, run_id=rid, clock=clock, analyzer=rgb_material.RGBMaterialAnalyzer(),
        verification={"revisit": {"refind_radius_um": 100}}, bench={"microscope": {"revisit_objectives": [20, 100]}},
        params={}, project_id=None, check_cancel=lambda: None)
    yield ctx, frames, labels, rotation
    store.close()


def produce(ctx, frame):
    capture = ctx.store.emit("device.command", {"device": "microscope", "command": "capture", "params": {}}, run_id=ctx.run_id)
    detection = ctx.analyzer.revisit_target(frame, "Gr")
    return search._rgb_revisit_evidence(ctx, ctx.analyzer, frame, detection, "chip", "flake", capture), capture


def test_exact20_source_has_original_capture_rotation_and_immutable_masks(context):
    ctx, frames, _, rotation = context
    result, capture = produce(ctx, frames[20])
    g = result["optical_geometry"]
    assert g["source"] == "rgb_optical_outline" and g["chip_rotation_deg"] == 15.
    validate_rgb_outline(g, material="Gr")
    row = ctx.store.events_after(result["recognition_event_id"]-1, limit=1)[0]
    assert row.kind == "microscope.rgb_recognition" and row.run_id == ctx.run_id
    assert row.payload["recognition_chain"] == result["recognition_20x"] == g["recognition_chain"]
    assert row.payload["capture_event_id"] == capture and row.payload["chip_rotation_event_id"] == rotation
    assert rotation < capture < row.id
    assert result["mask_20x"] != result["geometry_mask_20x"]
    ctx.store.set_run_status(ctx.run_id, "succeeded")
    validate_rgb_capture_binding(ctx.store, ctx.store.get_sample("flake"), g)


def test_original_capture_receipt_rejects_coherent_scale_and_outline_rewrite(context):
    ctx, frames, _, _ = context
    result, _ = produce(ctx, frames[20])
    g = deepcopy(result["optical_geometry"])
    ctx.store.set_run_status(ctx.run_id, "succeeded")
    g["recognition_chain"]["calibration"]["mm_per_pixel"][0] *= 2
    g["outline_mm"] = [[x*2, y] for x, y in g["outline_mm"]]
    with pytest.raises(ValueError, match="original capture"):
        validate_rgb_capture_binding(ctx.store, ctx.store.get_sample("flake"), g)


def test_unsupported20_topology_still_persists_distinct_original_masks(context):
    ctx, frames, labels, _ = context
    labels[43:53, 59:69] = 0
    result, _ = produce(ctx, frames[20])
    assert result["optical_geometry"] is None and result["recognition_status"] == "unsupported"
    assert len(result["recognition_20x"]["derivation"]["rings_px"]) == 2
    original = np.asarray(Image.open(result["mask_20x"]))
    derived = np.asarray(Image.open(result["geometry_mask_20x"]))
    assert np.array_equal(original == 2, derived != 0) and not derived[47, 63]
    assert ctx.store.events_after(result["recognition_event_id"]-1, limit=1)[0].payload["recognition_chain"] == result["recognition_20x"]


@pytest.mark.parametrize("rgb_backend", [True, False])
def test_real_revisit_step_keeps20_geometry_when100_classification_arrives(context, monkeypatch, rgb_backend):
    ctx, frames, _, _ = context
    class Scope:
        objective = 20
        def get_objective(self):
            return self.objective
        def get_focus(self):
            return 0.
        def set_objective(self, value):
            self.objective = value
        def capture(self):
            ctx.store.emit("device.command", {"device": "microscope", "command": "capture", "params": {}}, run_id=ctx.run_id)
            return frames[self.objective]
    scope = Scope()
    ctx.gate = SimpleNamespace(microscope=lambda name: scope)
    monkeypatch.setattr(search, "_move_stage", lambda *a, **kw: None)
    monkeypatch.setattr(search, "autofocus", lambda *a, **kw: 10.)
    from qumus_simlab.flows.library import pickup_evidence
    monkeypatch.setattr(pickup_evidence, "capture_neighbourhood", lambda *a: {"clear": False, "images": []})
    if not rgb_backend:
        actual = ctx.analyzer
        def legacy_detections(frame, material):
            detection = actual.revisit_target(frame, material)
            detection.meta.pop("recognition_chain")
            detection.meta["geometry_source"] = "declared_legacy_unit"
            return [detection]
        ctx.analyzer = SimpleNamespace(find_flakes=legacy_detections)
    step = search.RevisitCandidates("chip", "Gr", 1)
    monkeypatch.setattr(step, "_targets", lambda ctx: [ctx.store.get_sample("flake")])
    step.act(ctx, None)
    result = step._results["flake"]
    assert result["found"] and result["labels"] == {"20x": "monolayer", "100x": "bulk"}
    assert result["optical_geometry"]["image"] == frames[20 if rgb_backend else 100].path
    if rgb_backend:
        assert result["optical_geometry"]["recognition_chain"]["image"]["objective"] == 20
        assert result["recognition_100x"]["image"]["objective"] == 100
    else:
        assert result["optical_geometry"]["source"] == "simulated_optical_outline"
        assert "recognition_20x" not in result
    step.on_success(ctx, None)
    props = ctx.store.get_sample("flake")["props"]
    assert props["optical_geometry"]["image"] == frames[20 if rgb_backend else 100].path
    if rgb_backend:
        assert props["mask_20x"] != props["geometry_mask_20x"]
    else:
        assert "mask_20x" not in props and "geometry_mask_20x" not in props


def test_new_rgb_failure_cannot_reuse_previous_metadata_geometry(context):
    ctx, _, _, _ = context
    ctx.store.upsert_sample("flake", "flake", props={"optical_geometry": {"source": "simulated_optical_outline"}, "pickup_assessment": {"clear": True}})
    step = search.RevisitCandidates("chip", "Gr", 1)
    step._results = {"flake": {"found": False, "labels": {}, "images": {}, "recognition_backend": "v1_lab_rgb",
                               "recognition_status": "unsupported", "recognition_reason": "No RGB material candidate"}}
    step._publish_results(ctx)
    props = ctx.store.get_sample("flake")["props"]
    assert props["optical_geometry"] is None and props["pickup_assessment"] is None
    assert props["recognition_20x"] is None and props["mask_20x"] is None


def test_latest_original_rotation_receipt_is_used_and_must_precede_capture(context):
    ctx, frames, _, _ = context
    latest = ctx.store.emit("step.verified", {"name": "measure_chip_rotation", "verdict": "ok", "measurements": {"rotation_deg": 15.}}, run_id=ctx.run_id)
    result, _ = produce(ctx, frames[20])
    event = ctx.store.events_after(result["recognition_event_id"]-1, limit=1)[0]
    assert event.payload["chip_rotation_event_id"] == latest
    ctx.store.emit("step.verified", {"name": "measure_chip_rotation", "verdict": "unclear", "measurements": {}}, run_id=ctx.run_id)
    with pytest.raises(PreconditionFailed, match="original observed chip rotation"):
        produce(ctx, frames[20])
