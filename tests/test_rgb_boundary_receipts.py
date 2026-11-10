"""Artifact/coordinate contracts; the raster estimator has separate image tests."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.regions import _geos, as_region
from qumus_simlab.perception.measurement import measure_recognition_boundary
from qumus_simlab.records.recognition import validate_recognition_chain


def source_chain(tmp_path):
    shape = (16, 24)
    selected = np.zeros(shape, dtype=bool)
    selected[5:10, 7:14] = True
    chain = {"version": "v1-rgb-mask-chain-v1", "backend": "v1_lab_rgb", "material": "Gr",
        "algorithm_sha256": "a"*64, "material_policy_sha256": "b"*64,
        "material_policy": {"material": "Gr", "objective": 20}, "selected_label_ids": [1],
        "calibration": {"image_size_px": [24, 16], "mm_per_pixel": [.000182, .000182],
                        "u_axis": "+world_x", "v_axis": "-world_y"},
        "derivation": {"policy": "center-priority-original-component-v1",
                       "operation": "select_original_label_only", "morphology": "none",
                       "selected_label_id": 1}}
    for key, array in (("image", np.full((*shape, 3), 100, dtype=np.uint8)),
                       ("original_mask", selected.astype(np.uint16)),
                       ("geometry_mask", selected.astype(np.uint8)*255)):
        path = tmp_path/(key+".png")
        Image.fromarray(array).save(path)
        chain[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    chain["image"].update(objective=20, size_px=[24, 16])
    return chain, selected


def analytic_result(seed):
    """Explicit raster test double, never a qualified measurement profile."""
    return {"status": "measured", "physical_qualification": "unqualified",
        "derived_mask": seed.copy(), "uncertain_mask": np.zeros_like(seed),
        "edge_mask": np.zeros_like(seed), "method_id": "analytic-receipt-test-double",
        "algorithm_sha256": "c"*64, "provenance": {}, "diagnostics": {}, "reason": None}


def test_receipt_preserves_originals_and_corrects_only_new_coordinates(tmp_path, monkeypatch):
    chain, seed = source_chain(tmp_path)
    original_chain = json.dumps(chain, sort_keys=True)
    original_bytes = {key: open(chain[key]["path"], "rb").read()
                      for key in ("image", "original_mask", "geometry_mask")}
    def candidate(rgb, mask, calibration, **kwargs):
        assert not rgb.flags.writeable and not mask.flags.writeable
        assert np.array_equal(mask, seed)
        assert set(kwargs) == {"material", "objective", "recognition_policy"}
        return analytic_result(mask)
    monkeypatch.setattr("qumus_simlab.perception.boundary.refine_rgb_boundary", candidate)
    receipt = measure_recognition_boundary(chain, output_dir=tmp_path/"measured")
    assert json.dumps(chain, sort_keys=True) == original_chain
    for key, before in original_bytes.items():
        assert open(chain[key]["path"], "rb").read() == before
    validate_recognition_chain(chain, image_path=chain["image"]["path"])
    region = _geos(as_region(receipt["measurement"]["region_exposure_mm"]))
    pitch = .000182
    assert region.bounds == pytest.approx(((7-12)*pitch, (8-10)*pitch,
                                          (14-12)*pitch, (8-5)*pitch))
    assert receipt["qualification"]["status"] == "unqualified"
    assert receipt["measurement"]["physical_qualification"] == "unqualified"
    assert all(a["path"] not in [chain[k]["path"] for k in original_bytes]
               for a in receipt["artifacts"].values())
    saved = receipt["receipt_artifact"]
    assert hashlib.sha256(open(saved["path"], "rb").read()).hexdigest() == saved["sha256"]
    again = measure_recognition_boundary(chain, output_dir=tmp_path/"measured")
    assert again == receipt


def test_changed_original_is_refused_before_measurement(tmp_path, monkeypatch):
    chain, _ = source_chain(tmp_path)
    chain["image"]["sha256"] = "0"*64
    monkeypatch.setattr("qumus_simlab.perception.boundary.refine_rgb_boundary",
                        lambda *args, **kwargs: pytest.fail("Changed source reached estimator"))
    with pytest.raises(ValueError, match="bytes changed"):
        measure_recognition_boundary(chain, output_dir=tmp_path/"measured")
    assert not (tmp_path/"measured").exists()


@pytest.mark.parametrize("failure", ["physical_claim", "malformed_map"])
def test_estimator_cannot_publish_invalid_evidence(tmp_path, monkeypatch, failure):
    chain, seed = source_chain(tmp_path)
    result = analytic_result(seed)
    if failure == "physical_claim":
        result["physical_qualification"] = "qualified_conditional_sim"
    else:
        result["derived_mask"] = result["derived_mask"].astype(np.uint8)
    monkeypatch.setattr("qumus_simlab.perception.boundary.refine_rgb_boundary", lambda *a, **k: result)
    with pytest.raises(ValueError):
        measure_recognition_boundary(chain, output_dir=tmp_path/"measured")
    assert not (tmp_path/"measured").exists()
