"""Original journal joins, independent of the separately tested RGB estimator."""
import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from qumus_simlab.records.recognition import validate_native_anchor_event


def fixture(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE event (id INTEGER, run_id TEXT, kind TEXT, payload_json TEXT)")
    def put(i, kind, payload, job="job"):
        db.execute("INSERT INTO event VALUES (?,?,?,?)", (i, job, kind, json.dumps(payload)))
        return {"id": i, "run_id": job, "kind": kind, "payload": payload}
    rotation = put(1, "step.verified", {"name": "measure_chip_rotation"})
    for objective, capture_id, measurement_id in ((20, 2, 3), (100, 4, 5)):
        capture = put(capture_id, "device.command", {"device": "microscope", "command": "capture"})
        put(measurement_id, "microscope.rgb_measurement", {
            "objective": objective, "source_chip_id": "chip", "flake_id": "flake",
            "capture_event_id": capture_id,
            "chip_rotation_evidence": {"rotation_event": rotation, "capture_event": capture},
            "measurement": {"original_recognition_chain": {"objective": objective}}})
    anchor = {"frame": "native100_derived_centroid_chip_axes_v1", "admission_status": "blocked"}
    put(6, "microscope.native_source_anchor", {"source_chip_id": "chip", "flake_id": "flake",
        "measurement_event_20x": 3, "measurement_event_100x": 5, "anchor": anchor,
        "admission_status": "blocked"})
    calls = []
    def register(first, current):
        assert first["original_recognition_chain"]["objective"] == 20
        assert current[0]["original_recognition_chain"]["objective"] == 100
        calls.append("registration")
        return {"qualified": False}
    def rebuild(current, original, **kwargs):
        assert current["original_recognition_chain"]["objective"] == 100
        assert original["objective"] == 20
        assert kwargs["chip_rotation_evidence"]["capture_event"]["id"] == 4
        assert kwargs["registration_evidence"] == {"qualified": False}
        calls.append("anchor")
        return copy.deepcopy(anchor)
    monkeypatch.setattr("qumus_simlab.perception.observations.register_rgb_measurements", register)
    monkeypatch.setattr("qumus_simlab.perception.observations.native100_source_anchor", rebuild)
    return db, SimpleNamespace(_x=db.execute), calls


def test_original_journal_is_replayed_without_granting_admission(monkeypatch):
    db, store, calls = fixture(monkeypatch)
    assert validate_native_anchor_event(store, 6)["admission_status"] == "blocked"
    assert calls == ["registration", "anchor"]
    db.close()


@pytest.mark.parametrize("problem", ["foreign_job", "source", "rotation", "capture", "order", "anchor", "admission"])
def test_modified_or_foreign_evidence_is_rejected(monkeypatch, problem):
    db, store, _ = fixture(monkeypatch)
    if problem == "foreign_job":
        db.execute("UPDATE event SET run_id='other' WHERE id=5")
    else:
        i = 6 if problem in ("order", "anchor", "admission") else 5
        body = json.loads(db.execute("SELECT payload_json FROM event WHERE id=?", (i,)).fetchone()[0])
        if problem == "source":
            body["source_chip_id"] = "other"
        elif problem == "rotation":
            body["chip_rotation_evidence"]["rotation_event"]["payload"]["name"] = "forged"
        elif problem == "capture":
            body["capture_event_id"] = 2
        elif problem == "order":
            body["measurement_event_100x"] = 6
        elif problem == "anchor":
            body["anchor"]["frame"] = "other"
        else:
            body["admission_status"] = "qualified"
        db.execute("UPDATE event SET payload_json=? WHERE id=?", (json.dumps(body), i))
    with pytest.raises(ValueError):
        validate_native_anchor_event(store, 6)
    db.close()
