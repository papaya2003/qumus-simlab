"""Complete image-derived body containment; unchanged interior pockets stay unknown."""
from PIL import Image, ImageDraw

from qumus_simlab.perception.tracking import (
    PixelCalibration,
    v1_pickup_coverage,
    v1_pickup_outline_coverage,
)


def frames(pocket=False):
    reference = Image.new("RGB", (100, 100), "white")
    current = reference.copy()
    draw = ImageDraw.Draw(current)
    draw.rectangle((10, 10, 90, 90), fill=(100, 100, 100))
    if pocket:
        draw.rectangle((45, 45, 55, 55), fill="white")
    return reference, current


def observe(a, b, points):
    return v1_pickup_outline_coverage(a, b, PixelCalibration(.001, -.001), points,
        margin_mm=.01, min_area_mm2=.00001, blur_support_mm=0)


def test_complete_body_margin_has_calibrated_pixel_evidence():
    a, b = frames()
    result = observe(a, b, [(35, 35), (65, 35), (65, 65), (35, 65)])
    assert result["covered"] and result["holes_filled"] is False
    assert result["boundary_segments_checked"] > 256  # Full proof exceeds display sample.
    assert result["minimum_outline_clearance_mm"] > .01


def test_interior_uncovered_pocket_cannot_be_filled_for_whole_source():
    a, b = frames(pocket=True)
    old = v1_pickup_coverage(a, b, PixelCalibration(.001, -.001), (50, 50),
        margin_mm=.01, min_area_mm2=.00001, blur_support_mm=0)
    assert old["covered"]  # Historical external-contour semantics are preserved.
    assert not observe(a, b, [(35, 35), (65, 35), (65, 65), (35, 65)])["covered"]


def test_long_offcenter_source_point_passes_but_edge_margin_fails():
    a, b = frames()
    assert not observe(a, b, [(20, 45), (85, 45), (85, 55), (20, 55)])["covered"]


def test_clipped_source_never_inherits_point_coverage():
    a, b = frames()
    result = observe(a, b, [(-1, 35), (65, 35), (65, 65), (-1, 65)])
    assert not result["covered"] and "clipped" in result["reason"]
