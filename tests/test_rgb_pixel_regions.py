"""Exact raster/region topology controls; no classifier, stock, flow or model use."""
import copy
import time

import numpy as np
import pytest
from shapely import contains_xy, prepare
from shapely.geometry import MultiPolygon, Polygon

from qumus_simlab.core.regions import as_region, region_area_um2, region_dict
from qumus_simlab.perception.rgb_material import component_geometry, mask_cell_rings

CALIBRATION = {"u_axis": "+world_x", "v_axis": "-world_y",
               "mm_per_pixel": [.000182, .000273]}


def rasterize_region(region, shape, centroid, calibration=CALIBRATION):
    """Independent full pixel-center sampling of the calibrated final region."""
    parsed = as_region(region)
    geometry = MultiPolygon([Polygon(part.exterior, part.holes) for part in parsed.components])
    prepare(geometry)
    sx, sy = calibration["mm_per_pixel"]
    xs = (np.arange(shape[1])-centroid[0])*sx
    raster = np.zeros(shape, dtype=bool)
    for first in range(0, shape[0], 128):
        ys = -(np.arange(first, min(first+128, shape[0]))-centroid[1])*sy
        raster[first:first+len(ys)] = contains_xy(geometry, xs[None, :], ys[:, None])
    return raster


def assert_exact_roundtrip(mask):
    original = mask.copy()
    geometry = component_geometry(mask, CALIBRATION)
    assert geometry["status"] == "supported", geometry["unsupported_reason"]
    parsed = as_region(geometry["region_mm"])
    expected_area = int(mask.sum())*np.prod(CALIBRATION["mm_per_pixel"])*1e6
    assert region_area_um2(parsed) == pytest.approx(expected_area, rel=1e-11)
    rebuilt = rasterize_region(parsed, mask.shape, geometry["centroid_px"])
    assert np.array_equal(mask != 0, rebuilt)
    assert np.array_equal(mask, original)
    assert all(len(set(map(tuple, ring))) == len(ring) for ring in geometry["rings_px"])
    return geometry


def test_every_three_by_three_pixel_topology_roundtrips_exactly():
    # Every local binary junction, including point-pinched holes and diagonal
    # islands, is validated without special-case ring-walk repairs.
    for pattern in range(1, 1 << 9):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = np.array([(pattern >> bit) & 1 for bit in range(9)]).reshape(3, 3)
        assert_exact_roundtrip(mask)


@pytest.mark.parametrize("kind", ["hole", "disconnected", "diagonal_corner", "pinched_hole"])
def test_topology_changes_are_detected_by_independent_raster_roundtrip(kind):
    mask = np.zeros((18, 20), dtype=bool)
    if kind == "diagonal_corner":
        mask[7, 7] = mask[8, 8] = True
    else:
        mask[3:15, 4:16] = True
        if kind == "hole":
            mask[7:10, 8:11] = False
        elif kind == "disconnected":
            mask[1, 1] = True
        else:
            mask[3, 4] = mask[4, 5] = False
    geometry = assert_exact_roundtrip(mask)
    changed = copy.deepcopy(geometry["region_mm"])
    if kind in ("hole", "pinched_hole"):
        assert sum(len(p["holes"]) for p in changed["components"]) == 1
        changed["components"][0]["holes"] = []
    elif kind == "disconnected":
        changed["components"] = [max(changed["components"], key=lambda p: Polygon(p["exterior"]).area)]
    else:
        vertices = np.array([point for p in changed["components"] for point in p["exterior"]])
        x0, y0 = vertices.min(axis=0)
        x1, y1 = vertices.max(axis=0)
        changed = region_dict({"version": "polygon-region-v1", "components": [{
            "exterior": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": []}]})
    assert not np.array_equal(mask, rasterize_region(changed, mask.shape, geometry["centroid_px"]))


def test_native_sized_sparse_noisy_region_roundtrip_is_bounded():
    mask = np.zeros((2360, 3560), dtype=bool)
    mask[1050:1350, 1550:1850] = True
    # Preserve an exterior notch, small internal holes, point-pinched holes and
    # a disconnected island; these are declared topology controls, not stock.
    mask[1100:1190, 1780:1850] = False
    for y, x in [(1060, 1560), (1061, 1561), (1280, 1700), (1281, 1701)]:
        mask[y, x] = False
    mask[1050, 1550] = mask[1051, 1551] = False
    mask[1000:1002, 1900:1902] = True
    started = time.monotonic()
    geometry = assert_exact_roundtrip(mask)
    assert time.monotonic()-started < 15
    assert len(geometry["region_mm"]["components"]) == 2
    assert sum(len(part["holes"]) for part in geometry["region_mm"]["components"]) >= 5


def test_pixel_ring_codec_keeps_prior_simple_rectangle_and_hole_conventions():
    mask = np.zeros((12, 14), dtype=bool)
    mask[3:9, 4:10] = True
    assert mask_cell_rings(mask) == [[[3.5, 2.5], [9.5, 2.5], [9.5, 8.5], [3.5, 8.5]]]
    mask[5:7, 6:8] = False
    rings = mask_cell_rings(mask)
    assert len(rings) == 2
    assert rings[1] == [[5.5, 4.5], [5.5, 6.5], [7.5, 6.5], [7.5, 4.5]]


def test_excessive_exact_run_input_refuses_without_dropping_pixels():
    mask = np.zeros((2360, 3560), dtype=bool)
    mask[2:600:2, 2:-2:2] = True
    original = mask.copy()
    result = component_geometry(mask, CALIBRATION)
    assert result["status"] == "unsupported" and "exact-run profile" in result["unsupported_reason"]
    assert "region_mm" not in result and np.array_equal(mask, original)
