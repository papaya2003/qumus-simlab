"""Native measurement event wiring; estimator accuracy has independent image tests."""
import json
from types import SimpleNamespace

import pytest

from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE
from qumus_simlab.flows.library import search


@pytest.mark.parametrize("objective", [20, 100])
def test_native_revisit_preserves_original_events_and_never_grants_admission(monkeypatch, objective):
    rotation = {"name": "measure_chip_rotation", "verdict": "ok", "measurements": {"rotation_deg": 13.}}
    capture = {"device": search.MICROSCOPE, "command": "capture", "result": {"image": "original.png"}}
    rows = [{"id": 7, "payload_json": json.dumps(rotation)},
            {"id": 9, "run_id": "job1", "kind": "device.command", "payload_json": json.dumps(capture)}]
    emitted = []
    def emit(kind, payload, **kwargs):
        emitted.append((kind, payload, kwargs))
        return 10
    store = SimpleNamespace(_x=lambda *a: SimpleNamespace(fetchone=lambda: rows.pop(0)), emit=emit)
    ctx = SimpleNamespace(store=store, run_id="job1")
    frame = SimpleNamespace(meta={"objective": objective, "calibration": {
        "material_capture_profile": V1_MATERIAL_CAPTURE_PROFILE}})
    chain = {"image": {"sha256": "a"*64, "objective": objective}}
    calls = []
    def persist(*args):
        calls.append(args)
        return chain
    measurement = {"measurement": {"status": "measured", "reason": None}, "qualification": {"status": "unqualified"}}
    monkeypatch.setattr(search, "_chip_rotation", lambda *a: 13.)
    monkeypatch.setattr("qumus_simlab.perception.measurement.measure_recognition_boundary", lambda c: measurement if c is chain else None)
    result = search._rgb_revisit_evidence(ctx, SimpleNamespace(persist_detection_masks=persist), frame,
        SimpleNamespace(material="Gr", thickness_label="bulk"), "chip1", "flake1", 9)
    assert len(calls) == 1
    assert result[f"measurement_{objective}x"] is measurement
    assert result["optical_geometry"] is None
    assert result["native_source_admission"].startswith("blocked")
    evidence = result[f"native_rotation_{objective}x"]
    assert evidence["rotation_event"]["payload"] == rotation
    assert evidence["capture_event"]["payload"] == capture
    assert emitted[0][0] == "microscope.rgb_measurement"
    assert emitted[0][1]["admission_status"] == "blocked"


@pytest.mark.parametrize("problem", ["missing_rotation", "foreign_capture"])
def test_native_revisit_refuses_missing_or_foreign_original_events(monkeypatch, problem):
    rotation = {"name": "measure_chip_rotation", "verdict": "ok", "measurements": {"rotation_deg": 0.}}
    rows = [None if problem == "missing_rotation" else {"id": 7, "payload_json": json.dumps(rotation)},
            {"id": 9, "run_id": "other_job", "kind": "device.command", "payload_json": "{}"}]
    ctx = SimpleNamespace(run_id="job1", store=SimpleNamespace(
        _x=lambda *a: SimpleNamespace(fetchone=lambda: rows.pop(0))))
    monkeypatch.setattr(search, "_chip_rotation", lambda *a: 0.)
    an = SimpleNamespace(persist_detection_masks=lambda *a: pytest.fail("Unbound capture reached persistence"))
    frame = SimpleNamespace(meta={"objective": 100})
    with pytest.raises(search.PreconditionFailed):
        search._native_rgb_revisit_evidence(ctx, an, frame, SimpleNamespace(material="Gr"), "chip1", "flake1", 9)


def test_publication_preserves_native_anchor_and_clears_stale_evidence():
    from qumus_simlab.core.clock import SimClock
    from qumus_simlab.runtime.store import RunStore

    store = RunStore(":memory:", clock=SimClock(), bench="sim")
    store.upsert_sample("chip", "chip", props={"rotation_deg": 0.})
    store.upsert_sample("flake", "flake", location="chip", props={"material": "Gr"})
    ctx = SimpleNamespace(store=store)
    step = search.RevisitCandidates("chip", "Gr", 1)
    evidence = {"native_source_anchor_event_id": 17,
                "native_source_anchor": {"frame": "native100_derived_centroid_chip_axes_v1"},
                "measurement_event_20x": 12, "measurement_event_100x": 16,
                "native_source_admission": "blocked"}
    result = {"found": True, "labels": {}, "images": {}, "stage_xy": [0., 0.],
              "recognition_backend": "v1_lab_rgb", **evidence}
    step._results = {"flake": result}
    step._publish_results(ctx)
    props = store.get_sample("flake")["props"]
    assert all(props[key] == value for key, value in evidence.items())
    assert props["optical_geometry"] is None
    step._results = {"flake": {"found": False, "labels": {}, "images": {},
                               "recognition_backend": "v1_lab_rgb"}}
    step._publish_results(ctx)
    props = store.get_sample("flake")["props"]
    assert all(props[key] is None for key in evidence)
    assert store.get_sample("flake")["status"] == "lost"
    store.close()
