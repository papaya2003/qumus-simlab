"""The OpenCV v1 difference (#204) decides exactly like the per-pixel original."""

from __future__ import annotations

import hashlib
import math
from collections import deque

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from qumus_simlab.perception.tracking import (
    PixelCalibration,
    _closed_boundary_polygon,
    _v1_difference,
    _whole_outline_cell_coverage,
    load_pixels,
)


def reference_v1_difference(reference: object, current: object, calibration: PixelCalibration,
                   *, threshold: int, aspect: tuple[float, float], min_area_mm2: float,
                   blur_support_mm: float = 0., target_px: tuple[float, float] | None = None,
                   margin_mm: float = 0., target_outline_px=None) -> dict:
    """Fixed-reference v1 observation, with 4-connected regions in actual pixels.

    The Gaussian's six-sigma support is an explicit raster adaptation of the
    historical nine-pixel kernel. It does not recover subpixel information.
    Distances use external cell edges in the calibrated image plane; holes are
    filled for the external-contour test, as in v1 RETR_EXTERNAL. No metadata,
    palette, simulator state, registration or motion estimate enters this test.
    """
    sx, sy = abs(calibration.mm_per_pixel_x), abs(calibration.mm_per_pixel_y)
    if any(not math.isfinite(v) or v <= 0 for v in (sx, sy, min_area_mm2)):
        raise ValueError("Difference observation requires finite positive pixel scales and area")
    a, b = load_pixels(reference).convert("L"), load_pixels(current).convert("L")
    if a.size != b.size:
        raise ValueError("Difference images have different native sizes")
    if blur_support_mm:
        blur = ImageFilter.GaussianBlur((blur_support_mm / (6*sx), blur_support_mm / (6*sy)))
        a, b = a.filter(blur), b.filter(blur)
    difference = ImageChops.difference(a, b)
    lo, hi = difference.getextrema()
    result = {"found": False, "threshold": threshold, "connectivity": 4,
              "minimum_area_mm2": min_area_mm2, "blur_support_mm": blur_support_mm,
              "difference_range": [lo, hi], "covered": False}
    if hi == lo:
        return result
    w, h = a.size
    mask = bytearray(1 if (v-lo)*255/(hi-lo) > threshold else 0
                     for v in difference.tobytes())
    selected: list[int] = []
    for start in range(w*h):
        if not mask[start]:
            continue
        mask[start] = 0
        todo, component = deque([start]), []
        while todo:
            i = todo.popleft()
            component.append(i)
            x, y = i % w, i // w
            for j in (i-1 if x else -1, i+1 if x+1 < w else -1,
                      i-w if y else -1, i+w if y+1 < h else -1):
                if j >= 0 and mask[j]:
                    mask[j] = 0
                    todo.append(j)
        if len(component) <= len(selected) or len(component)*sx*sy <= min_area_mm2:
            continue
        xs, ys = [i % w for i in component], [i // w for i in component]
        ratio = (max(xs)-min(xs)+1)*sx / ((max(ys)-min(ys)+1)*sy)
        if aspect[0] <= ratio <= aspect[1]:
            selected = component
    if not selected:
        return result
    cells = set(selected)
    # Flood only the outside; enclosed holes are absent from the external contour.
    outside, todo = set(), deque()
    for i in [*range(w), *range((h-1)*w, h*w), *range(0, w*h, w), *range(w-1, w*h, w)]:
        if i not in cells and i not in outside:
            outside.add(i)
            todo.append(i)
    while todo:
        i = todo.popleft()
        x, y = i % w, i // w
        for j in (i-1 if x else -1, i+1 if x+1 < w else -1,
                  i-w if y else -1, i+w if y+1 < h else -1):
            if j >= 0 and j not in cells and j not in outside:
                outside.add(j)
                todo.append(j)
    filled = set(range(w*h))-outside
    boundary = []
    for i in filled:
        x, y = i % w, i // w
        # Each side runs in one fixed rotational sense around its cell, so
        # the union's external edges form a balanced closed walk.
        if x == 0 or i-1 in outside:
            boundary.append((x-.5, y-.5, x-.5, y+.5))
        if x == w-1 or i+1 in outside:
            boundary.append((x+.5, y+.5, x+.5, y-.5))
        if y == 0 or i-w in outside:
            boundary.append((x+.5, y-.5, x-.5, y-.5))
        if y == h-1 or i+w in outside:
            boundary.append((x-.5, y+.5, x+.5, y+.5))
    xs, ys = [i % w for i in selected], [i // w for i in selected]
    region_bytes = bytes(255 if i in cells else 0 for i in range(w*h))
    result.update(found=True, centroid_px=[sum(xs)/len(xs), sum(ys)/len(ys)],
                  bbox_px=[min(xs), min(ys), max(xs)+1, max(ys)+1],
                  component_pixels=len(cells), area_mm2=len(cells)*sx*sy,
                  region_mask_sha256=hashlib.sha256(region_bytes).hexdigest(),
                  external_boundary_segments_px=_closed_boundary_polygon(boundary, 256),
                  boundary_display_sampled=len(boundary) > 256)
    if target_outline_px is not None:
        result.update(_whole_outline_cell_coverage(cells, w, h, target_outline_px, sx, sy, margin_mm))
    if target_px is not None:
        tx, ty = target_px
        if any(not math.isfinite(v) for v in target_px) or not (0 <= tx < w and 0 <= ty < h):
            raise ValueError("Coverage target lies outside the captured image")
        inside = int(ty+.5)*w+int(tx+.5) in filled
        distances = [math.hypot((tx-min(max(tx, x1), x2))*sx,
                                (ty-min(max(ty, y1), y2))*sy)
                     for x1, y1, x2, y2 in boundary]
        distance = min(distances)*(1 if inside else -1)
        result.update(target_px=list(target_px), target_inside=inside,
                      edge_distance_mm=distance, required_margin_mm=margin_mm,
                      covered=inside and distance > margin_mm)
    return result




CAL = PixelCalibration(0.005416666666666667, -0.005375)


def _pair(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 130, (120, 160, 3), dtype=np.uint8)
    a = Image.fromarray(base)
    b = a.copy()
    d = ImageDraw.Draw(b)
    for _ in range(int(rng.integers(1, 5))):
        x, y = rng.integers(10, 150), rng.integers(10, 110)
        r = rng.integers(3, 30)
        fill = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < .5:
            d.ellipse((x-r, y-r*rng.uniform(.3, 1.7), x+r, y+r), fill=fill)
        else:
            d.rectangle((x-r, y-r//2, x+r, y+r), fill=fill)
        if rng.random() < .4:  # a hole inside
            d.ellipse((x-r//3, y-r//3, x+r//3, y+r//3), fill=tuple(int(v) for v in base[int(y), int(x)]))
    return a, b


def _without_display(result):
    return {k: v for k, v in result.items() if k != "external_boundary_segments_px"}


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("kind", ["first", "pickup", "outline"])
def test_opencv_port_matches_the_per_pixel_original(seed, kind):
    a, b = _pair(seed)
    kw = {"first": dict(threshold=30, aspect=(.8, 1.25), min_area_mm2=.002),
          "pickup": dict(threshold=40, aspect=(.1, 10.), min_area_mm2=.002, blur_support_mm=.00657,
                         target_px=(80., 60.), margin_mm=.02),
          "outline": dict(threshold=40, aspect=(.1, 10.), min_area_mm2=.002, blur_support_mm=.00657,
                          target_outline_px=[(70., 50.), (90., 50.), (90., 70.), (70., 70.)],
                          margin_mm=.01)}[kind]
    new, old = _v1_difference(a, b, CAL, **kw), reference_v1_difference(a, b, CAL, **kw)
    assert _without_display(new) == _without_display(old)
    if new["found"] and not new["boundary_display_sampled"]:
        vertices = {seg[:2] for seg in new["external_boundary_segments_px"]}
        assert vertices == {seg[:2] for seg in old["external_boundary_segments_px"]}


def test_no_change_and_size_mismatch_behave_as_before():
    a, _ = _pair(1)
    assert _v1_difference(a, a, CAL, threshold=30, aspect=(.8, 1.25), min_area_mm2=.002) == \
        reference_v1_difference(a, a, CAL, threshold=30, aspect=(.8, 1.25), min_area_mm2=.002)
    with pytest.raises(ValueError):
        _v1_difference(a, a.resize((80, 60)), CAL, threshold=30, aspect=(.8, 1.25), min_area_mm2=.002)
