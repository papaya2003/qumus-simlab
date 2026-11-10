"""Owner-inclusive policy and native-source/unchanged-carried optical binding."""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE
from qumus_simlab.flows.library.stacking import _contact_classification_snapshot, _contact_optical
from qumus_simlab.perception.contact_coverage import InteriorCalibration
from qumus_simlab.perception.measurement import (
    boundary_measurement_status,
    owner_error_budget_policy,
    validate_owner_error_budget,
)
from qumus_simlab.runtime.store import RunStore


def test_owner_budget_is_inclusive_engineering_acceptance_not_scientific_qualification():
    policy = owner_error_budget_policy()
    assert validate_owner_error_budget({"owner_error_budget": policy, "boundary_uncertainty_mm": .001}) == .001
    assert policy["rotation_tolerance_deg"] == 1. and policy["translation_tolerance_um"] == 1.
    assert set(policy["includes"]) == {"measurement", "registration", "positioning"}
    assert policy["scientific_qualification"] == "not_calibrated_or_certified"
    assert boundary_measurement_status()["status"] == "unqualified"


def test_missing_or_changed_owner_policy_cannot_claim_the_inclusive_budget():
    policy = owner_error_budget_policy()
    for observation in ({}, {"owner_error_budget": policy, "boundary_uncertainty_mm": 0},
                        {"owner_error_budget": {**policy, "inclusive_error_budget_mm": .002},
                         "boundary_uncertainty_mm": .001}):
        with pytest.raises(ValueError):
            validate_owner_error_budget(observation)


@pytest.fixture
def native_source(tmp_path, monkeypatch):
    store = RunStore(":memory:", bench="sim")
    run = store.create_run("search_chip", {})
    events = {}
    for objective in (20, 100):
        events[objective] = store.emit("microscope.rgb_measurement", {"objective": objective,
            "source_chip_id": "chip1", "flake_id": "hBN1", "material": "hBN", "thickness_label": "0-10"},
            run_id=run)
    anchor = store.emit("microscope.native_source_anchor", {
        "measurement_event_20x": events[20], "measurement_event_100x": events[100]}, run_id=run)
    image = tmp_path/"observed-native-source.png"
    Image.new("RGB", (64, 64), (100, 90, 120)).save(image)
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    geometry = {"source": "rgb_native100_measured_region", "frame": "native100_derived_centroid_chip_axes_v1",
        "units": "mm", "source_chip_id": "chip1", "job_id": run, "image": str(image),
        "native_source_anchor_event_id": anchor, "recognition_chain": {"material": "hBN",
            "image": {"objective": 100, "sha256": sha},
            "calibration": {"material_capture_profile": V1_MATERIAL_CAPTURE_PROFILE}}}
    # The observation's source identity survives physical pickup. Mutable props
    # deliberately disagree: calibration must use original classifier events.
    store.upsert_sample("hBN1", "flake", location="stamp-current", props={"material": "hBN",
        "optical_geometry": geometry, "labels": {"100x": "20-30"}, "thickness_label": "20-30"})
    source = SimpleNamespace(material="hBN", flake_id="hBN1", image_sha256=sha, observation=geometry)
    monkeypatch.setattr("qumus_simlab.records.recognition.native_source_observation",
                        lambda actual_store, identifier: deepcopy(geometry)
                        if actual_store is store and identifier == anchor else None)
    yield store, SimpleNamespace(sources=[source]), events
    store.close()


def test_native_source_labels_bind_original_journal_and_preserve_actual_carried_forward_law(native_source):
    store, design, _ = native_source
    snapshot = _contact_classification_snapshot(store, design)
    assert snapshot["source"]["native_labels"] == {"20x": "0-10", "100x": "0-10"}
    assert snapshot["source"]["source_measurement_mode"] == V1_MATERIAL_CAPTURE_PROFILE
    assert snapshot["source"]["rendering_mode"] == "synthetic_complete_outline_flat_tint_v1"
    assert snapshot["carried"] == InteriorCalibration(("0-10",)).snapshot()
    assert snapshot["empty"] == InteriorCalibration().snapshot()
    optical = _contact_optical({"contact_interior_calibration": snapshot, "flake_ids": ["hBN1", "Gr1"]}, "Gr1")
    assert optical.generators() == InteriorCalibration(("0-10",)).generators()


def test_old_native_event_without_observed_classification_cannot_invent_one_from_props(native_source):
    store, design, events = native_source
    row = store._x("SELECT payload_json FROM event WHERE id=?", (events[100],)).fetchone()
    body = json.loads(row["payload_json"])
    del body["thickness_label"]
    store._x("UPDATE event SET payload_json=? WHERE id=?", (json.dumps(body), events[100]))
    with pytest.raises(ValueError, match="observed nanometre"):
        _contact_classification_snapshot(store, design)


def test_native_contact_classification_rejects_changed_design_image(native_source):
    store, design, _ = native_source
    design.sources[0].image_sha256 = "0"*64
    with pytest.raises(ValueError, match="original100 RGB"):
        _contact_classification_snapshot(store, design)
