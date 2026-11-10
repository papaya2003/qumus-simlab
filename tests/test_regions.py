"""Exact topology and uncertainty controls; all shapes are analytic unit data."""
import copy
import math

import pytest

from qumus_simlab.core.regions import (
    PolygonRegion,
    as_region,
    quantize_region,
    region_area_um2,
    region_bidirectional_within,
    region_clearance,
    region_compatible_with_bound,
    region_covers,
    region_covers_with_clearance,
    region_dict,
    region_difference,
    region_intersection,
    region_vertices,
    transform_region,
)
from qumus_simlab.records.geometry import pickup_coverage


def region(exterior, holes=(), extra=()):
    return as_region({"version": "polygon-region-v1", "components": [
        {"exterior": exterior, "holes": holes}, *extra]})


def square(a, b, c, d):
    return ((a, b), (c, b), (c, d), (a, d))


def test_concave_notch_is_not_filled_and_every_boundary_vertex_survives():
    points = ((0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3))
    material = region(points)
    assert region_area_um2(material) == pytest.approx(7e6)
    assert len(region_vertices(material)) == len(points)
    assert not region_covers(material, region(square(.5, 1.5, 2.5, 2.5)))
    flipped = region(tuple(reversed(points)))
    assert region_dict(material) == region_dict(flipped)


def test_hole_is_not_hidden_by_all_vertices_being_inside_exterior():
    material = region(square(0, 0, 10, 10), [square(4, 4, 6, 6)])
    crossing = region(square(3, 3, 7, 7))
    assert not region_covers(material, crossing)
    assert region_clearance(material, crossing) is None
    result = pickup_coverage(material, crossing, frame="unit", binding={"id": "analytic"},
        required_bound={"frame": "unit", "binding": {"id": "analytic"},
                        "required_clearance_mm": .1, "provenance": "unit_bound"})
    assert result["state"] == "bound_failed" and result["nominal_contains"] is False
    assert result["nominal_clearance_mm"] is None and result["headroom_mm"] is None


def test_holes_control_clearance_and_disconnected_components_are_not_bridged():
    material = region(square(0, 0, 10, 10), [square(4, 4, 6, 6)])
    inner = region(square(2, 4, 3, 5))
    assert region_clearance(material, inner) == pytest.approx(1.)
    assert region_covers_with_clearance(material, inner, 1.)
    assert not region_covers_with_clearance(material, inner, 1.00001)
    islands = region(square(0, 0, 1, 1), extra=[{"exterior": square(2, 0, 3, 1)}])
    assert not region_covers(islands, region(square(.5, .2, 2.5, .8)))


def test_boolean_operations_preserve_holes_and_disconnected_parts():
    outer = region(square(0, 0, 3, 3))
    cut = region(square(1, 1, 2, 2))
    annulus = region_difference(outer, cut)
    assert len(annulus.components[0].holes) == 1
    assert region_area_um2(annulus) == pytest.approx(8e6)
    split = region_intersection(annulus, region(square(0, 1.25, 3, 1.75)))
    assert len(split.components) == 2 and region_area_um2(split) == pytest.approx(1e6)
    moved = transform_region(annulus, xy=(10, 20), rotation_deg=37)
    assert region_area_um2(moved) == pytest.approx(region_area_um2(annulus))
    assert len(moved.components[0].holes) == 1


@pytest.mark.parametrize("shape", [
    {"exterior": [(0, 0), (2, 2), (0, 2), (2, 0)]},
    {"exterior": square(0, 0, 1, 1), "holes": [square(2, 2, 3, 3)]},
    {"exterior": [(0, 0), (1, 0), (1, float("nan"))]},
    {"exterior": [(0, 0), (True, 0), (1, 1)]},
    {"exterior": [(0, 0), ("1", 0), (1, 1)]},
])
def test_invalid_geometry_is_refused_without_repair(shape):
    with pytest.raises(ValueError):
        as_region({"version": "polygon-region-v1", "components": [shape]})


def test_overlapping_components_refuse_but_point_touching_cells_are_retained():
    with pytest.raises(ValueError):
        region(square(0, 0, 2, 2), extra=[{"exterior": square(1, 1, 3, 3)}])
    touching = region(square(0, 0, 1, 1), extra=[{"exterior": square(1, 1, 2, 2)}])
    assert len(touching.components) == 2


def test_quantization_has_separate_bound_and_refuses_collapsed_hole():
    material = region(square(.0000002, .0000002, .01, .01), [square(.004, .004, .006, .006)])
    original = copy.deepcopy(region_dict(material))
    quantized, receipt = quantize_region(material)
    assert region_dict(material) == original and len(quantized.components[0].holes) == 1
    assert receipt["boundary_displacement_bound_mm"] == pytest.approx(math.sqrt(2)*.5e-6)
    tiny_hole = region(square(0, 0, .01, .01), [square(.0040001, .0040001, .0040002, .0040002)])
    with pytest.raises(ValueError):
        quantize_region(tiny_hole)
    assert PolygonRegion.model_validate(region_dict(material)) == material
    almost_touching = region(square(0, 0, .001, .001),
        extra=[{"exterior": square(.0010001, .0010001, .002, .002)}])
    with pytest.raises(ValueError, match="contact topology"):
        quantize_region(almost_touching)


def test_association_is_bidirectional_bounded_and_does_not_change_original_holes():
    material = region(square(0, 0, 10, 10), [square(4, 4, 6, 6)])
    original = copy.deepcopy(region_dict(material))
    shifted = transform_region(material, xy=(.01, 0))
    assert not region_bidirectional_within(material, shifted, 0)
    assert region_compatible_with_bound(material, shifted, .011)
    assert not region_bidirectional_within(material, shifted, .009)
    assert region_bidirectional_within(material, material, 0)
    filled = region(square(0, 0, 10, 10))
    assert not region_bidirectional_within(material, filled, .1)
    # A one-direction cover must not hide an added remote material component.
    extra = region(square(0, 0, 10, 10), [square(4, 4, 6, 6)],
                   [{"exterior": square(20, 20, 21, 21)}])
    assert not region_bidirectional_within(material, extra, .1)
    assert region_dict(material) == original


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), True, "0.1"])
def test_association_refuses_invalid_uncertainty(budget):
    material = region(square(0, 0, 1, 1))
    with pytest.raises(ValueError):
        region_bidirectional_within(material, material, budget)
