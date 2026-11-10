"""Immutable raster receipts must reproduce actual image-derived measurements."""
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.perception.measurement import (
    measure_recognition_boundary,
    validate_measurement_receipt,
)
from qumus_simlab.perception.observations import (
    _boundary_distance_interval,
    native100_source_anchor,
    register_rgb_measurements,
)


@pytest.fixture
def saved_measurement(tmp_path, monkeypatch):
    shape = (16, 24)
    seed = np.zeros(shape, dtype=bool)
    seed[5:10, 7:14] = True
    chain = {"version": "v1-rgb-mask-chain-v1", "backend": "v1_lab_rgb", "material": "Gr",
        "algorithm_sha256": "a"*64, "material_policy_sha256": "b"*64,
        "material_policy": {"material": "Gr", "objective": 20}, "selected_label_ids": [1],
        "calibration": {"image_size_px": [24, 16], "mm_per_pixel": [.000182, .000182],
                        "u_axis": "+world_x", "v_axis": "-world_y"},
        "derivation": {"policy": "center-priority-original-component-v1",
            "operation": "select_original_label_only", "morphology": "none", "selected_label_id": 1}}
    for key, array in (("image", np.full((*shape, 3), 100, dtype=np.uint8)),
                       ("original_mask", seed.astype(np.uint16)),
                       ("geometry_mask", seed.astype(np.uint8)*255)):
        path = tmp_path/(key+".png")
        Image.fromarray(array).save(path)
        chain[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    chain["image"].update(objective=20, size_px=[24, 16])

    def analytic_measurement(rgb, mask, calibration, **kwargs):
        assert not rgb.flags.writeable and not mask.flags.writeable
        return {"status": "measured", "physical_qualification": "unqualified",
            "derived_mask": mask.copy(), "uncertain_mask": np.zeros_like(mask),
            "edge_mask": np.zeros_like(mask), "method_id": "analytic-validation-test-double",
            "algorithm_sha256": "c"*64, "provenance": {}, "diagnostics": {}, "reason": None}
    monkeypatch.setattr("qumus_simlab.perception.boundary.refine_rgb_boundary", analytic_measurement)
    return measure_recognition_boundary(chain, output_dir=tmp_path/"measured")


def resign_receipt(receipt):
    """Model a caller who can recompute hashes, but cannot change image evidence."""
    from pathlib import Path

    value = deepcopy(receipt)
    artifact = value.pop("receipt_artifact")
    payload = json.dumps(value, sort_keys=True, indent=2).encode()+b"\n"
    path = Path(artifact["path"]).with_name("caller-rewritten-receipt.json")
    path.write_bytes(payload)
    receipt["receipt_artifact"] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
    return receipt


def test_validation_reproduces_region_and_preserves_all_original_artifacts(saved_measurement):
    original = deepcopy(saved_measurement)
    assert validate_measurement_receipt(saved_measurement) == original
    assert saved_measurement == original
    assert saved_measurement["qualification"]["status"] == "unqualified"


def test_rehashed_claimed_region_cannot_replace_image_geometry(saved_measurement):
    changed = deepcopy(saved_measurement)
    changed["measurement"]["centroid_exposure_mm"][0] += .01
    with pytest.raises(ValueError, match="does not reproduce"):
        validate_measurement_receipt(resign_receipt(changed))


def test_rehashed_pixel_artifact_must_reproduce_estimator(saved_measurement):
    from pathlib import Path

    changed = deepcopy(saved_measurement)
    artifact = changed["artifacts"]["derived_mask"]
    path = Path(artifact["path"]).with_name("caller-rewritten-mask.png")
    with Image.open(artifact["path"]) as image:
        pixels = np.asarray(image).copy()
    pixels[0, 0] = 255
    Image.fromarray(pixels).save(path)
    changed["artifacts"]["derived_mask"] = {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="do not reproduce"):
        validate_measurement_receipt(resign_receipt(changed))


def test_changed_original_is_not_a_replayable_measurement(saved_measurement):
    from pathlib import Path

    path = Path(saved_measurement["original_recognition_chain"]["image"]["path"])
    path.write_bytes(b"changed original")
    with pytest.raises(ValueError, match="bytes changed"):
        validate_measurement_receipt(saved_measurement)


def simple_region(*, xy=(0, 0), hole=False):
    x, y = xy
    return {"version": "polygon-region-v1", "components": [{
        "exterior": [[x-.01, y-.006], [x+.01, y-.006], [x+.01, y+.006], [x-.01, y+.006]],
        "holes": [[[x-.002, y-.002], [x-.002, y+.002], [x+.002, y+.002], [x+.002, y-.002]]]
                  if hole else []}]}


def registration_receipt(objective, *, xy=(0, 0), material="Gr", digest="d"):
    pitch = .000182 if objective == 20 else .000036
    return {"original_recognition_chain": {"image": {"objective": objective, "sha256": digest*64,
                                                                   "path": "analytic-test-image.png"},
        "selected_label_ids": [1], "material": material, "calibration": {"mm_per_pixel": [pitch, pitch]}},
        "recognition_chain_sha256": digest*64, "receipt_artifact": {"sha256": digest*64},
        "wrapper_dependencies": {"measurement.py": "a"*64},
        "measurement": {"status": "measured", "region_exposure_mm": simple_region(xy=xy),
                        "centroid_exposure_mm": list(xy), "provenance": {}, "algorithm_sha256": "c"*64}}


def test_registration_records_image_translation_without_inventing_physical_precision(monkeypatch):
    monkeypatch.setattr("qumus_simlab.perception.measurement.validate_measurement_receipt", deepcopy)
    original = registration_receipt(20)
    current = registration_receipt(100, xy=(.003, -.002), digest="e")
    result = register_rgb_measurements(original, [current])
    assert result["status"] == "unknown" and result["physical_qualification"] == "unqualified"
    assert result["registration_uncertainty_mm"] is None
    candidate = result["candidates"][0]
    assert candidate["translation_mm"] == pytest.approx([.003, -.002])
    assert candidate["boundary_discrepancy_interval_mm"][0] == pytest.approx(0, abs=1e-14)
    assert candidate["native100_region_exposure_mm"] == current["measurement"]["region_exposure_mm"]


def test_duplicate_image_matches_cannot_create_a_unique_source_identity(monkeypatch):
    monkeypatch.setattr("qumus_simlab.perception.measurement.validate_measurement_receipt", deepcopy)
    result = register_rgb_measurements(registration_receipt(20),
                                      [registration_receipt(100), registration_receipt(100, digest="f")])
    assert result["image_ranking"] == "ambiguous_overlapping_numerical_residual_intervals"
    assert result["status"] == "unknown"


def test_registration_discrepancy_includes_real_hole_boundaries():
    interval = _boundary_distance_interval(simple_region(hole=True), simple_region(), .0001)
    assert interval[0] >= .004-1e-12
    assert interval[1] >= interval[0] and interval[1]-interval[0] < .000051


def rotation_evidence(*, angle=37., image_sha256="e"*64):
    return {"version": "observed-chip-rotation-v1", "source_chip_id": "chip-actual-observed",
        "native100_image_sha256": image_sha256,
        "rotation_event": {"id": 7, "run_id": "actual-job", "kind": "step.verified",
            "payload": {"name": "measure_chip_rotation", "verdict": "ok",
                        "measurements": {"rotation_deg": angle}}},
        "capture_event": {"id": 10, "run_id": "actual-job", "kind": "device.command",
                          "payload": {"device": "microscope", "command": "capture", "params": {}}}}


def test_native100_centroid_rebase_preserves_physical_placement_and_holes(monkeypatch):
    import math

    from qumus_simlab.core.regions import _geos, as_region, transform_region

    monkeypatch.setattr("qumus_simlab.perception.measurement.validate_measurement_receipt", deepcopy)
    monkeypatch.setattr("qumus_simlab.records.recognition.validate_recognition_chain", lambda chain, **kw: chain)
    current = registration_receipt(100, xy=(.003, -.002), digest="e")
    current["measurement"]["region_exposure_mm"] = simple_region(xy=(.003, -.002), hole=True)
    original = registration_receipt(20)["original_recognition_chain"]
    before = deepcopy((current, original))
    anchor = native100_source_anchor(current, original, chip_rotation_evidence=rotation_evidence())
    assert anchor["frame"] == "native100_derived_centroid_chip_axes_v1"
    assert anchor["anchor_exposure_mm"] == [.003, -.002]
    assert anchor["boundary_uncertainty_mm"] is None and anchor["cross_scale_registration_uncertainty_mm"] is None
    assert anchor["registration"]["status"] == "unknown" and anchor["admission"] == "blocked"
    theta, old_xy = 81., (.04, -.08)
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    cx, cy = anchor["anchor_exposure_mm"]
    rebased_xy = (old_xy[0]+c*cx-s*cy, old_xy[1]+s*cx+c*cy)
    fixed = transform_region(current["measurement"]["region_exposure_mm"], xy=old_xy, rotation_deg=theta)
    rebased = transform_region(anchor["region_mm"], xy=rebased_xy, rotation_deg=theta+37.)
    assert _geos(fixed).hausdorff_distance(_geos(rebased)) < 1e-14
    assert len(as_region(anchor["region_mm"]).components[0].holes) == 1
    assert (current, original) == before


def test_native100_anchor_requires_original_rotation_events_and_same_image(monkeypatch):
    monkeypatch.setattr("qumus_simlab.perception.measurement.validate_measurement_receipt", deepcopy)
    monkeypatch.setattr("qumus_simlab.records.recognition.validate_recognition_chain", lambda chain, **kw: chain)
    current = registration_receipt(100, digest="e")
    original = registration_receipt(20)["original_recognition_chain"]
    for evidence in (None, {"rotation_deg": 0}, rotation_evidence(image_sha256="f"*64)):
        with pytest.raises(ValueError, match="rotation"):
            native100_source_anchor(current, original, chip_rotation_evidence=evidence)
