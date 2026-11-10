"""Pure declared RGB/mask parity and topology; no World, jobs or camera model."""
from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from qumus_simlab.devices.base import Frame
from qumus_simlab.perception._v1_rgb_core import Segmenter
from qumus_simlab.perception.rgb_material import (
    RGBMaterialAnalyzer,
    center_priority_candidate,
    component_geometry,
    mask_cell_rings,
    material_policy,
    segment_rgb,
)
from qumus_simlab.records.recognition import validate_recognition_chain

FIXTURE = json.loads((Path(__file__).parent/"fixtures/v1_rgb_parity.json").read_text(encoding="utf-8"))
CAL = {"source": "declared_unit_calibration", "u_axis": "+world_x", "v_axis": "-world_y",
       "mm_per_pixel": [.002, .003], "image_size_px": [480, 320]}


def pixels(case):
    def rgb(lab):
        return cv2.cvtColor(np.array([[[round(lab[0]*255/100), round(lab[1]+128), round(lab[2]+128)]]],
                                    dtype=np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
    width, height = case["size_px"]
    result = np.tile(rgb(case["background_lab"]), (height, width, 1))
    x0, y0, x1, y1 = case["rectangle_px"]
    result[y0:y1, x0:x1] = rgb(case["foreground_lab"])
    return result


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_numerical_port_matches_untouched_frozen_v1_fixture(case):
    rgb = pixels(case)
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == case["rgb_sha256"]
    material, _ = material_policy(case["material"], case["objective"])
    core = Segmenter(rgb, material, magnification=case["objective"], surround_bg=True,
                     seed=int(case["rgb_sha256"][:8], 16))
    if not case["background_support_pixels"]:
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"), np.errstate(invalid="ignore"):
            core.process_frame()
    else:
        core.process_frame()
    assert hashlib.sha256(np.asarray(core.masks, dtype="<u2").tobytes()).hexdigest() == case["raw_label_sha256"]
    assert int(core.bg_mask.sum()) == case["background_support_pixels"]
    if case["background_support_pixels"]:
        result = segment_rgb(rgb, case["material"], objective=case["objective"])
        assert result.candidates == tuple(case["candidates"])
        assert not result.raw_labels.flags.writeable
    else:
        with pytest.raises(ValueError, match="background support"):
            segment_rgb(rgb, case["material"], objective=case["objective"])


def test_port_does_not_modify_frozen_source_or_global_rng():
    # The frozen v1 donor copy lives in qumus-agent/qumus-v2 (v1compare/), not here.
    donor = Path(__file__).resolve().parents[1]/"v1compare/v1/agents/processing_agent/segmenter"
    if donor.exists():
        for name, digest in FIXTURE["donor_sha256"].items():
            assert hashlib.sha256((donor/name).read_bytes()).hexdigest() == digest
    before = np.random.get_state()
    segment_rgb(pixels(FIXTURE["cases"][0]), "Gr", objective=20)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_exact_pixel_cells_use_signed_non_square_scale_without_hull():
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[3:7, 4:9] = 1
    result = component_geometry(mask, CAL)
    assert result["status"] == "supported"
    assert result["centroid_px"] == [6., 4.5]
    assert result["area_um2"] == pytest.approx(120.)
    assert result["dx_mm"] == 0.
    assert result["dy_mm"] == pytest.approx(.0015)
    assert set(map(tuple, result["outline_mm"])) == {(-.005, -.006), (.005, -.006), (.005, .006), (-.005, .006)}


@pytest.mark.parametrize("kind", ["hole", "concave", "clipped", "diagonal_touch"])
def test_exact_region_topology_retains_actual_mask_and_boundary(kind):
    mask = np.zeros((14, 16), dtype=np.uint8)
    mask[3:11, 4:12] = 1
    if kind == "hole":
        mask[5:9, 6:10] = 0
    elif kind == "concave":
        mask[3:7, 8:12] = 0
    elif kind == "clipped":
        mask[:4, 4:12] = 1
    else:
        mask[2, 3] = 1
    original = mask.copy()
    result = component_geometry(mask, CAL)
    if kind == "clipped":
        assert result["status"] == "unsupported" and "region_mm" not in result
        assert result["unsupported_reason"] == "clipped_component"
    else:
        from qumus_simlab.core.regions import region_area_um2

        assert result["status"] == "supported"
        assert region_area_um2(result["region_mm"]) == pytest.approx(mask.sum()*6.)
        assert "outline_mm" not in result
    assert np.array_equal(mask, original)
    if kind == "hole":
        assert len(result["rings_px"]) == 2
        assert len(result["region_mm"]["components"][0]["holes"]) == 1
    if kind == "concave":
        assert len(mask_cell_rings(mask)[0]) == 6
    if kind == "diagonal_touch":
        assert len(result["region_mm"]["components"]) == 2


def test_center_priority_keeps_small_center_target_over_large_far_neighbour():
    centered = {"label_id": 2, "area_px": 300, "centroid_px": [240., 160.]}
    large = {"label_id": 3, "area_px": 20000, "centroid_px": [340., 160.]}
    assert center_priority_candidate([large, centered], [480, 320]) == centered
    with pytest.raises(ValueError, match="Ambiguous"):
        center_priority_candidate([centered, {**centered, "label_id": 4}], [480, 320])
    with pytest.raises(ValueError, match="No RGB"):
        center_priority_candidate([], [480, 320])


def rgb_frame(tmp_path):
    rgb = pixels(FIXTURE["cases"][0])
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    path = tmp_path/"original20.png"
    path.write_bytes(buffer.getvalue())
    return Frame("microscope", 123., path=str(path), data=buffer.getvalue(), meta={"objective": 20,
        "calibration": deepcopy(CAL), "flakes": [{"material": "untrusted", "outline_mm": [[999, 999]]}]})


def test_real_rgb_adapter_ignores_simulator_annotations_and_keeps_two_masks(tmp_path):
    frame = rgb_frame(tmp_path)
    analyzer = RGBMaterialAnalyzer()
    detections = analyzer.find_flakes(frame, "Gr")
    assert detections
    raw_before = bytes(frame.data)
    original = detections[0].model_dump(mode="json")
    frame.meta["flakes"] = [{"material": "Gr", "area_um2": 1e12}]
    assert analyzer.find_flakes(frame, "Gr")[0].model_dump(mode="json") == original
    target = analyzer.revisit_target(frame, "Gr")
    chain = analyzer.persist_detection_masks(frame, target)
    assert chain["derivation"]["ambiguity_status"] == "unique"
    assert chain["derivation"]["selection_context"] == "v1_center_priority_revisit"
    assert chain["material"] == "Gr"
    assert chain["original_mask"]["path"] != chain["geometry_mask"]["path"] != frame.path
    labels = np.asarray(Image.open(chain["original_mask"]["path"]))
    geometry = np.asarray(Image.open(chain["geometry_mask"]["path"]))
    assert np.array_equal(geometry != 0, labels == chain["selected_label_ids"][0])
    assert frame.data == raw_before
    assert validate_recognition_chain(chain, image_path=frame.path) == chain
    assert analyzer.persist_detection_masks(frame, target) == chain
    geometry = geometry.copy()
    geometry[1, 1] = 255
    Image.fromarray(geometry).save(chain["geometry_mask"]["path"])
    with pytest.raises(ValueError, match="changed"):
        validate_recognition_chain(chain, image_path=frame.path)


@pytest.mark.parametrize("bad", ["WTe2", "unknown"])
def test_unported_material_is_explicitly_unsupported(bad):
    with pytest.raises(ValueError, match="Unsupported"):
        segment_rgb(pixels(FIXTURE["cases"][0]), bad, objective=20)
