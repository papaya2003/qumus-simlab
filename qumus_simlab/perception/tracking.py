"""Deterministic raster measurements; no simulator state or device commands.

Registration measures motion in image axes (right/down, positive Pillow rotation
counterclockwise). Calibration, focus-plane identity and evidence provenance are
supplied by callers. Quality scores and uncertainty are conservative heuristics,
not calibrated confidence intervals or laboratory rate measurements.
"""

from __future__ import annotations

import hashlib
import io
import math
import statistics
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

_SPECIMEN_BLUR_RADIUS_PX = .65


@dataclass(frozen=True)
class PixelROI:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True)
class PixelCalibration:
    """Signed mm/pixel for an already established optical plane."""

    mm_per_pixel_x: float
    mm_per_pixel_y: float


def _closed_boundary_polygon(edges, limit):
    """One closed boundary polygon of at most `limit` distinct vertices.

    The pixel-edge boundary is walked as one closed circuit (a region that
    pinches at a pixel corner visits that corner twice), then every k-th
    vertex is kept, skipping repeats, so the published segments stay one
    closed loop with degree-two vertices. A strided slice of unordered
    edges, as before, was open whenever there were more than `limit` edges.
    """
    out = {}
    for ax, ay, bx, by in edges:
        out.setdefault((ax, ay), []).append((bx, by))
    if not out:
        return []
    stack, walk = [next(iter(out))], []
    while stack:
        if out.get(stack[-1]):
            stack.append(out[stack[-1]].pop())
        else:
            walk.append(stack.pop())
    walk = walk[::-1][:-1]
    step = max(1, math.ceil(len(walk)/limit))
    kept, seen = [], set()
    for vertex in walk[::step]:
        if vertex not in seen:
            seen.add(vertex)
            kept.append(vertex)
    if len(kept) < 3:
        return []
    return [(*a, *b) for a, b in zip(kept, kept[1:]+kept[:1], strict=True)]


def _v1_difference(reference: object, current: object, calibration: PixelCalibration,
                   *, threshold: int, aspect: tuple[float, float], min_area_mm2: float,
                   blur_support_mm: float = 0., target_px: tuple[float, float] | None = None,
                   margin_mm: float = 0., target_outline_px=None) -> dict:
    """Fixed-reference v1 observation, with 4-connected regions in actual pixels.

    The Gaussian's six-sigma support is an explicit raster adaptation of the
    historical nine-pixel kernel. It does not recover subpixel information.
    Distances use external cell edges in the calibrated image plane; holes are
    filled for the external-contour test, as in v1 RETR_EXTERNAL. No metadata,
    palette, simulator state, registration or motion estimate enters this test.
    Regions are labelled with OpenCV (v1's own library); the selection keeps
    the first largest qualifying component in raster order of its first pixel.
    """
    import cv2
    import numpy as np
    from scipy import ndimage

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
    values = np.asarray(difference, dtype=np.int64)
    mask = (((values-lo)*255)/(hi-lo) > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    flat = labels.reshape(-1)
    _, first = np.unique(flat, return_index=True)
    best, best_size = 0, 0
    for label in sorted(range(1, count), key=lambda k: first[k]):
        size = int(stats[label, cv2.CC_STAT_AREA])
        if size <= best_size or size*sx*sy <= min_area_mm2:
            continue
        ratio = int(stats[label, cv2.CC_STAT_WIDTH])*sx / (int(stats[label, cv2.CC_STAT_HEIGHT])*sy)
        if aspect[0] <= ratio <= aspect[1]:
            best, best_size = label, size
    if not best:
        return result
    cells = labels == best
    # Only the outside is flooded (4-connected); enclosed holes are absent
    # from the external contour.
    filled = ndimage.binary_fill_holes(cells)
    pad = np.pad(filled, 1, constant_values=False)
    ys, xs = np.nonzero(filled)
    boundary = []
    for dx, dy, edge in ((-1, 0, lambda x, y: (x-.5, y-.5, x-.5, y+.5)),
                         (1, 0, lambda x, y: (x+.5, y+.5, x+.5, y-.5)),
                         (0, -1, lambda x, y: (x+.5, y-.5, x-.5, y-.5)),
                         (0, 1, lambda x, y: (x-.5, y+.5, x+.5, y+.5))):
        open_side = ~pad[ys+1+dy, xs+1+dx]
        boundary.extend(edge(float(x), float(y)) for x, y in zip(xs[open_side], ys[open_side], strict=True))
    cy, cx = np.nonzero(cells)
    region_bytes = (cells.astype(np.uint8)*255).tobytes()
    result.update(found=True, centroid_px=[float(cx.sum())/len(cx), float(cy.sum())/len(cy)],
                  bbox_px=[int(cx.min()), int(cy.min()), int(cx.max())+1, int(cy.max())+1],
                  component_pixels=best_size, area_mm2=best_size*sx*sy,
                  region_mask_sha256=hashlib.sha256(region_bytes).hexdigest(),
                  external_boundary_segments_px=_closed_boundary_polygon(boundary, 256),
                  boundary_display_sampled=len(boundary) > 256)
    if target_outline_px is not None:
        result.update(_whole_outline_cell_coverage(set(np.flatnonzero(cells).tolist()), w, h,
                                                   target_outline_px, sx, sy, margin_mm))
    if target_px is not None:
        tx, ty = target_px
        if any(not math.isfinite(v) for v in target_px) or not (0 <= tx < w and 0 <= ty < h):
            raise ValueError("Coverage target lies outside the captured image")
        inside = bool(filled[int(ty+.5), int(tx+.5)])
        edges = np.asarray(boundary, dtype=float)
        # Same clamp as the original per-edge loop (segment endpoints in walk order).
        x1, y1, x2, y2 = edges.T
        distances = np.hypot((tx-np.minimum(np.maximum(tx, x1), x2))*sx,
                             (ty-np.minimum(np.maximum(ty, y1), y2))*sy)
        distance = float(distances.min())*(1 if inside else -1)
        result.update(target_px=list(target_px), target_inside=inside,
                      edge_distance_mm=distance, required_margin_mm=margin_mm,
                      covered=inside and distance > margin_mm)
    return result


def _whole_outline_cell_coverage(cells, width, height, outline_px, sx, sy, margin_mm):
    """Strict convex body+margin inside observed cells, retaining all interior holes."""
    from qumus_simlab.core.polygons import convex_polygon, cross

    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("whole-outline coverage needs a finite nonnegative physical margin")
    points = tuple((float(x), float(y)) for x, y in outline_px)
    polygon = convex_polygon((x*sx, y*sy) for x, y in points)
    base = {"target_outline_px": [list(p) for p in points], "required_margin_mm": margin_mm,
            "coverage_rule": "complete_convex_outline_in_unfilled_observed_cells_v1", "holes_filled": False,
            "covered": False}
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        return {**base, "reason": "selected source outline is clipped"}
    if any(int(y+.5)*width+int(x+.5) not in cells for x, y in points):
        return {**base, "reason": "selected source has an uncovered vertex"}

    def inside(p):
        return all(cross(a, b, p) >= -1e-14 for a, b in zip(polygon, polygon[1:]+polygon[:1], strict=True))

    def point_distance(p, a, b):
        length = (b[0]-a[0])**2+(b[1]-a[1])**2
        t = max(0., min(1., ((p[0]-a[0])*(b[0]-a[0])+(p[1]-a[1])*(b[1]-a[1]))/length))
        return math.dist(p, (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))

    def distance(a, b):
        if inside(a) or inside(b):
            return 0.
        best = math.inf
        for c, d in zip(polygon, polygon[1:]+polygon[:1], strict=True):
            if cross(a, b, c)*cross(a, b, d) <= 0 and cross(c, d, a)*cross(c, d, b) <= 0:
                if (max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
                        and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))):
                    return 0.
            best = min(best, point_distance(a, c, d), point_distance(b, c, d),
                       point_distance(c, a, b), point_distance(d, a, b))
        return best

    clearance, count = math.inf, 0
    for i in cells:
        x, y = i % width, i // width
        segments = []
        if x == 0 or i-1 not in cells:
            segments.append((x-.5, y-.5, x-.5, y+.5))
        if x == width-1 or i+1 not in cells:
            segments.append((x+.5, y-.5, x+.5, y+.5))
        if y == 0 or i-width not in cells:
            segments.append((x-.5, y-.5, x+.5, y-.5))
        if y == height-1 or i+width not in cells:
            segments.append((x-.5, y+.5, x+.5, y+.5))
        for x1, y1, x2, y2 in segments:
            count += 1
            clearance = min(clearance, distance((x1*sx, y1*sy), (x2*sx, y2*sy)))
    return {**base, "minimum_outline_clearance_mm": clearance, "boundary_segments_checked": count,
            "covered": clearance > margin_mm,
            "reason": None if clearance > margin_mm else "source outline or required margin contains an uncovered boundary/pocket"}


def _whole_region_cell_coverage(cells, width, height, region_px, sx, sy, margin_mm):
    """Exact material components/holes inside the union of observed pixel cells."""
    import numpy as np
    import shapely
    from shapely import affinity

    from qumus_simlab.core.regions import _geos, as_region, region_dict

    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("Region coverage needs a finite nonnegative physical margin")
    region = as_region(region_px)
    material = affinity.scale(_geos(region), xfact=sx, yfact=sy, origin=(0, 0))
    base = {"target_region_px": region_dict(region), "required_margin_mm": margin_mm,
            "coverage_rule": "complete_region_in_unfilled_observed_cells_v1", "holes_filled": False,
            "covered": False}
    if material.is_empty or not cells:
        return {**base, "reason": "No complete material or observed contact cells"}
    indices = np.array(sorted(cells), dtype=np.int64)
    if indices[0] < 0 or indices[-1] >= width*height:
        raise ValueError("Observed contact cell lies outside the image")
    breaks = np.flatnonzero((np.diff(indices) != 1) | (indices[:-1]//width != indices[1:]//width))+1
    starts, ends = indices[np.r_[0, breaks]], indices[np.r_[breaks-1, len(indices)-1]]
    rows = starts//width
    boxes = shapely.box((starts % width-.5)*sx, (rows-.5)*sy,
                        (ends % width+.5)*sx, (rows+.5)*sy)
    observed = shapely.union_all(boxes)
    contained = bool(observed.covers(material))
    clearance = float(observed.boundary.distance(material)) if contained else 0.
    passed = contained and clearance > margin_mm
    return {**base, "minimum_outline_clearance_mm": clearance,
            "covered": passed, "observed_runs_checked": len(starts),
            "reason": None if passed else "material or required margin crosses an unobserved contact cell"}


def v1_first_contact(reference: object, current: object, calibration: PixelCalibration,
                     *, min_area_mm2: float = .0026645) -> dict:
    """Largest near-round changed component; an optical candidate, not apex/contact truth."""
    return _v1_difference(reference, current, calibration, threshold=30,
                          aspect=(.8, 1.25), min_area_mm2=min_area_mm2)


def v1_pickup_coverage(reference: object, current: object, calibration: PixelCalibration,
                       target_px: tuple[float, float], *, margin_mm: float = .1752,
                       min_area_mm2: float = .0026645, blur_support_mm: float = .00657) -> dict:
    """Largest changed region's strict target-point interior margin; not transfer proof."""
    return _v1_difference(reference, current, calibration, threshold=40,
                          aspect=(.1, 10.), min_area_mm2=min_area_mm2,
                          blur_support_mm=blur_support_mm, target_px=target_px, margin_mm=margin_mm)


def v1_pickup_outline_coverage(reference: object, current: object, calibration: PixelCalibration,
                              outline_px, *, margin_mm: float, min_area_mm2: float = .0026645,
                              blur_support_mm: float = .00657) -> dict:
    """New-design whole-body evidence; unlike v1 external contours, holes remain unknown."""
    return _v1_difference(reference, current, calibration, threshold=40, aspect=(.1, 10.),
        min_area_mm2=min_area_mm2, blur_support_mm=blur_support_mm,
        target_outline_px=outline_px, margin_mm=margin_mm)


def _ring_finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False

def _ring_scales(values):
    if (not isinstance(values, (tuple, list)) or len(values) != 2
            or any(not _ring_finite(value) or value <= 0 for value in values)):
        raise ValueError("Two finite positive physical pixel spacings are required")
    return tuple(values)

def _ring_point(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not _ring_finite(number) for number in value)):
        raise ValueError("A finite native image point is required")
    return tuple(value)

def _ring_consecutive_runs(bands, channel):
    """Missing radii and zero/ambiguous phases break a channel's evidence run."""
    runs, run, previous = [], [], None
    for index, band in enumerate(bands):
        radius, phase = band["radius_native_px"], band["phase_derivative"][channel]
        if not phase or previous is None or radius != previous+1:
            if run:
                runs.append(run)
            run = []
        if phase:
            run.append(index)
        previous = radius if phase else None
    if run:
        runs.append(run)
    return runs

def _ring_components(points):
    remaining = set(points)
    while remaining:
        seed = remaining.pop()
        todo, component = deque([seed]), [seed]
        while todo:
            x, y = todo.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = x+dx, y+dy
                    if other in remaining:
                        remaining.remove(other)
                        todo.append(other)
                        component.append(other)
        yield [points[key] for key in component]

def _ring_families(reference, current, scales):
    sx, sy = _ring_scales(scales)
    before, after = load_pixels(reference), load_pixels(current)
    if before.size != after.size or min(before.size) < 64:
        raise ValueError("Identical native rasters with an observable interior are required")
    pixel = max(sx, sy)
    width, height = before.size
    points = {}
    for index, (a, b) in enumerate(zip(before.get_flattened_data(), after.get_flattened_data(), strict=True)):
        x, y = index % width, index // width
        if not (8 <= x < width-8 and 30 <= y < height-32):
            continue
        change = [b[j]-a[j] for j in range(3)]
        if max(change)-min(change) > 4:
            points[x, y] = (x, y, change[0]-change[1], change[1]-change[2])
    candidates, largest = [], 0
    for component in _ring_components(points):
        largest = max(largest, len(component))
        if len(component) < 24:
            continue
        center = [statistics.mean(p[j] for p in component) for j in (0, 1)]
        bins = {}
        for x, y, rg, gb in component:
            dx, dy = (x-center[0])*sx/pixel, (y-center[1])*sy/pixel
            radius = round(math.hypot(dx, dy))
            sector = int((math.atan2(dy, dx)+math.pi)/(math.pi/4)) % 8
            bins.setdefault(radius, {}).setdefault(sector, []).append((x, y, rg, gb))
        means = {radius: {sector: [statistics.mean(p[j] for p in values) for j in range(4)]
                         for sector, values in sectors.items()} for radius, sectors in bins.items()}
        bands = []
        for radius, sectors in sorted(means.items()):
            previous = means.get(radius-1, {})
            common = sectors.keys() & previous.keys()
            if len(common) < 6:
                continue
            phases = []
            for channel in (2, 3):
                changes = [sectors[s][channel]-previous[s][channel] for s in common]
                positive, negative = sum(v > 1 for v in changes), sum(v < -1 for v in changes)
                phases.append(1 if positive >= 6 else -1 if negative >= 6 else 0)
            if not any(phases):
                continue
            pairs = [s for s in range(4) if s in common and s+4 in common]
            if len(pairs) < 2:
                continue
            midpoints = [[(sectors[sector][j]+sectors[sector+4][j])/2 for j in (0, 1)] for sector in pairs]
            bands.append({"radius_native_px": radius, "sectors": sorted(common),
                          "opposing_pairs": pairs, "phase_derivative": phases, "center_midpoints_px": midpoints})
        runs, alternations, qualified = [], [], set()
        for channel in (0, 1):
            channel_runs, maximum = [], 0
            for indices in _ring_consecutive_runs(bands, channel):
                phases = [bands[index]["phase_derivative"][channel] for index in indices]
                changes = sum(a != b for a, b in zip(phases, phases[1:], strict=False))
                maximum = max(maximum, changes)
                channel_runs.append({"band_indices": indices, "changes": changes})
                if changes >= 2:
                    qualified.update(indices)
            runs.append(channel_runs)
            alternations.append(maximum)
        midpoints = [point for index in qualified for point in bands[index]["center_midpoints_px"]]
        spread = max((math.hypot((p[0]-center[0])*sx/pixel, (p[1]-center[1])*sy/pixel)
                      for p in midpoints), default=None)
        uncertainty = max(1., spread) if spread is not None else None
        inner_radius = min((bands[index]["radius_native_px"] for index in qualified), default=0)
        resolved = bool(qualified) and uncertainty < inner_radius
        candidates.append({"family_found": resolved, "component_pixels": len(component), "centre_px": center,
            "bands": bands, "consecutive_runs": runs, "qualified_band_indices": sorted(qualified),
            "coherent_derivative_changes": alternations,
            "band_center_spread_native_px": spread, "uncertainty_radius_native_px": uncertainty,
            "uncertainty_radius_mm": uncertainty*pixel if uncertainty is not None else None,
            "minimum_supported_radius_native_px": inner_radius,
            "scope": "Local optical candidate only; not physical contact, gap, apex or metrology"})
    families = [candidate for candidate in candidates if candidate["family_found"]]
    return {"found": len(families) == 1, "unique_family": len(families) == 1,
            "reason": None if len(families) == 1 else "absent, unresolved or multiple local families",
            "largest_component_pixels": largest, "candidates": candidates,
            "selected": families[0] if len(families) == 1 else None}

def _ring_candidate_boundary(candidate, scales):
    sx, sy = _ring_scales(scales)
    pixel = max(sx, sy)
    if not isinstance(candidate, dict):
        raise ValueError("An observed optical candidate is required")
    centroid = _ring_point(candidate.get("centroid_px"))
    segments = candidate.get("external_boundary_segments_px")
    if not isinstance(segments, (list, tuple)) or len(segments) < 3:
        raise ValueError("A nonempty closed local candidate boundary is required")
    degrees, neighbours = Counter(), {}
    for segment in segments:
        if (not isinstance(segment, (list, tuple)) or len(segment) != 4
                or any(not _ring_finite(value) for value in segment)):
            raise ValueError("Candidate boundary segments must be finite image coordinates")
        a, b = tuple(segment[:2]), tuple(segment[2:])
        if a == b:
            raise ValueError("Zero-length candidate boundary segment")
        degrees.update((a, b))
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    if any(degree != 2 for degree in degrees.values()):
        raise ValueError("Candidate boundary is open or has duplicated vertices")
    remaining = set(neighbours)
    todo = [remaining.pop()]
    while todo:
        for other in neighbours[todo.pop()]:
            if other in remaining:
                remaining.remove(other)
                todo.append(other)
    if remaining:
        raise ValueError("Multiple disconnected candidate boundaries are ambiguous")
    points = [((a+c)*sx/(2*pixel), (b+d)*sy/(2*pixel)) for a, b, c, d in segments]
    mx, my = [statistics.mean(p[j] for p in points) for j in (0, 1)]
    u = [x-mx for x, y in points]
    v = [y-my for x, y in points]
    xx, yy, xy = sum(x*x for x in u), sum(y*y for y in v), sum(x*y for x, y in zip(u, v, strict=True))
    determinant = xx*yy-xy*xy
    if not math.isfinite(determinant) or determinant <= 1e-9:
        raise ValueError("Candidate boundary does not identify a two-dimensional centre")
    bx = sum(x*(x*x+y*y) for x, y in zip(u, v, strict=True))/2
    by = sum(y*(x*x+y*y) for x, y in zip(u, v, strict=True))/2
    center = [mx+(bx*yy-by*xy)/determinant, my+(by*xx-bx*xy)/determinant]
    radii = [math.dist(p, center) for p in points]
    radius = statistics.mean(radii)
    residual = max(abs(r-radius) for r in radii)
    if any(not _ring_finite(value) for value in (*center, radius, residual)) or radius <= 0:
        raise ValueError("Candidate boundary fit has no finite positive radius/centre")
    centroid_metric = centroid[0]*sx/pixel, centroid[1]*sy/pixel
    uncertainty = max(1., residual, math.dist(center, centroid_metric))
    return {"boundary_center_px": [center[0]*pixel/sx, center[1]*pixel/sy], "centroid_px": centroid,
            "mean_boundary_radius_px": radius,
            "boundary_radial_residual_px": residual, "uncertainty_radius_native_px": uncertainty,
            "uncertainty_radius_mm": uncertainty*pixel, "well_conditioned": uncertainty < radius}

def _ring_candidate_geometry(family, candidate, scales):
    """Qualitative co-location only; caller must bind the SAME captured pixels.

    The existing v1 external boundary is a closed optical region. Its centre
    may be transparent, so containment here never means solid physical contact.
    """
    sx, sy = _ring_scales(scales)
    if not isinstance(family, dict) or family.get("family_found") is not True:
        return {"compatible": False, "reason": "No resolved local family"}
    center = _ring_point(family.get("centre_px"))
    radius = family.get("uncertainty_radius_mm")
    if not _ring_finite(radius) or radius < max(sx, sy):
        raise ValueError("Family uncertainty must include its native-pixel floor")
    try:
        boundary = _ring_candidate_boundary(candidate, scales)
    except ValueError as exc:
        # A degenerate observed boundary is a refusal of this exposure, not a crash.
        return {"compatible": False, "reason": f"Unresolved candidate boundary: {exc}"}
    inside, clearance = False, math.inf
    for ax, ay, bx, by in candidate["external_boundary_segments_px"]:
        if (ay > center[1]) != (by > center[1]):
            crossing = ax+(center[1]-ay)*(bx-ax)/(by-ay)
            if center[0] < crossing:
                inside = not inside
        dx, dy = (bx-ax)*sx, (by-ay)*sy
        px, py = (center[0]-ax)*sx, (center[1]-ay)*sy
        t = max(0., min(1., (px*dx+py*dy)/(dx*dx+dy*dy)))
        clearance = min(clearance, math.hypot(px-t*dx, py-t*dy))
    distance = math.hypot((center[0]-candidate["centroid_px"][0])*sx,
                          (center[1]-candidate["centroid_px"][1])*sy)
    combined = family["uncertainty_radius_mm"]+boundary["uncertainty_radius_mm"]
    compatible = (family["family_found"] and boundary["well_conditioned"] and inside
                  and family["uncertainty_radius_mm"] < clearance and distance <= combined)
    return {"compatible": compatible, "center_distance_mm": distance,
            "interval_radius_sum_mm": combined, "closed_optical_boundary_clearance_mm": clearance,
            "uncertainty_disk_within_closed_optical_boundary": inside and family["uncertainty_radius_mm"] < clearance,
            "candidate_boundary": boundary,
            "scope": "Native-pixel optical co-location; no absolute pose accuracy, contact or intact-material proof"}

def ring_candidate_binding(cue, candidate, scales):
    """Frame-record equality is necessary; production must recheck capture events."""
    if cue.get("current") != candidate.get("current") or cue.get("reference") != candidate.get("reference"):
        return {"compatible": False, "reason": "Cue and candidate do not use the same exposures"}
    for key in ("current", "reference"):
        record = cue.get(key)
        if (not isinstance(record, dict) or not record.get("sha256")
                or not _ring_finite(record.get("captured_at")) or not isinstance(record.get("meta"), dict)):
            raise ValueError("A complete immutable captured frame record is required")
    a, b = cue["reference"]["meta"], cue["current"]["meta"]
    if any(a.get(key) != b.get(key) for key in
           ("calibration", "objective", "focus_z", "focus_plane", "stage_kinematics")):
        return {"compatible": False, "reason": "Optical capture context changed"}
    observation = cue.get("observation", {})
    if observation.get("found") is not True or observation.get("unique_family") is not True:
        return {"compatible": False, "reason": "No unique resolved local family"}
    return _ring_candidate_geometry(observation["selected"], candidate, scales)

def precontact_ring_evidence(reference: object, current: object, calibration: PixelCalibration) -> dict:
    """One local coherent ring family from pixels; no gap or contact inference."""
    signed = (calibration.mm_per_pixel_x, calibration.mm_per_pixel_y)
    if any(not _ring_finite(value) or value == 0 for value in signed):
        raise ValueError("Ring evidence requires finite nonzero signed pixel calibration")
    result = _ring_families(reference, current, tuple(abs(value) for value in signed))
    result.update(method="synthetic-local-coherent-ring-v2",
                  scope="Native-pixel optical family; not contact, gap, apex or metrology")
    if result["found"]:
        center = result["selected"]["centre_px"]
        image = load_pixels(reference)
        result.update(centre_px=center,
            centre_offset_mm=[(center[0]-image.width/2)*signed[0],
                              (center[1]-image.height/2)*signed[1]],
            radial_sign_changes=max(result["selected"]["coherent_derivative_changes"]))
    return result


@dataclass(frozen=True)
class RegistrationConfig:
    max_translation_px: float = 48.0  # on the working raster (longest side 480 px)
    max_rotation_deg: float = 6.0
    rotation_step_deg: float = 2.0
    min_quality: float = 0.78
    min_ambiguity: float = 0.035
    nominal_rotation_deg: float = 0.0
    pixel_aspect_ratio: float = 1.0  # abs(mm/pixel_y) / abs(mm/pixel_x)


@dataclass(frozen=True)
class RegistrationResult:
    ok: bool
    dx_px: float = 0.0
    dy_px: float = 0.0
    rotation_deg: float = 0.0
    quality: float = 0.0
    ambiguity: float = 0.0
    uncertainty_px: float = math.inf
    fov_margin_px: float = 0.0
    loss_reason: str | None = None
    roi: PixelROI | None = None
    photometric_calibration: str | None = None
    photometric_noise_gain: float = 1.
    pose_sensitivity_gain: float = 1.
    pose_noise_floor_px: float = 0.
    reference_mask_offset_px: tuple[float, float] = (0., 0.)
    photometric_estimator: str | None = None
    # Retain historical diagnostic fields for the public result/JSON schema.
    photometric_interface_uncertainty_px: float = 0.
    photometric_interface_exclusion_px: float = 0.
    photometric_boundary_pairs: tuple[int, int] | None = None
    photometric_front_method: str | None = None
    photometric_front_diagnostics: tuple[float, float, float, float, float, float] | None = None


def load_pixels(source: object) -> Image.Image:
    """Read encoded pixels only, including lazy Frame saves; never inspect meta.

    A Frame's save callback must itself be a pure save of its captured exposure,
    as required by the Frame contract. Loading never calls capture or preview.
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return image.convert("RGB")
    data = getattr(source, "data", None)
    if data:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    path = getattr(source, "path", None)
    if not path:
        saver = getattr(source, "saved_path", None)
        path = saver() if callable(saver) else None
    if not path:
        raise ValueError("Captured frame has no raster pixels")
    return load_pixels(path)


def _valid_roi(image: Image.Image, roi: PixelROI) -> bool:
    return (0 <= roi.left < roi.right <= image.width
            and 0 <= roi.top < roi.bottom <= image.height
            and roi.right - roi.left >= 12 and roi.bottom - roi.top >= 12)


def raster_focus_score(source: object, roi: PixelROI | None = None, *, color: bool = False) -> float:
    """Noise-suppressed gradient energy; compare within one optical plane.

    Preblur and a two-level gradient deadband reject the renderer's small
    camera-fixed sensor noise and slow illumination falloff. Zero denotes no
    observable optical texture, not successful focus. A flat score curve cannot
    establish a focus plane; callers must retain/reacquire a textured reference.
    Scores are dimensionless heuristics rather than laboratory calibration.
    """
    image = load_pixels(source).convert("RGB" if color else "L")
    if roi is not None:
        if not _valid_roi(image, roi):
            raise ValueError("Invalid focus ROI")
        image = image.crop(roi.box)
    elif min(image.size) >= 24:
        # Microscope objective and scale legends live in the border. Their
        # constant sharp edges cannot establish focus of either specimen plane.
        image = image.crop((round(image.width*.06), round(image.height*.09),
                            round(image.width*.94), round(image.height*.88)))
    if min(image.size) < 5:
        return 0.0
    # Colour interference can have strong channel contrast but almost equal
    # luminance. Preserve that observed signal without changing the spatial
    # filter, noise deadband or peak-selection policy.
    channels = image.split() if color else (image,)
    energy = 0
    for channel in channels:
        smooth = channel.filter(ImageFilter.GaussianBlur(1.2))
        origin = smooth.crop((0, 0, smooth.width-1, smooth.height-1))
        gradients = [ImageChops.difference(origin, smooth.crop(box)) for box in
                     ((1, 0, smooth.width, smooth.height-1),
                      (0, 1, smooth.width-1, smooth.height))]
        energy += sum(sum(max(0, value-2)**2 * count
                          for value, count in enumerate(gradient.histogram()))
                      for gradient in gradients)
    score = energy/(origin.width*origin.height*255**2*len(channels))
    return score if score >= 1e-8 else 0.0


def polygon_roi(
    polygon_px: list[tuple[float, float]], *, padding: int = 4,
    image_size: tuple[int, int] | None = None,
) -> PixelROI:
    """Tight padded bounds for an observed optical outline, in image pixels."""
    if len(polygon_px) < 3 or any(not math.isfinite(v) for p in polygon_px for v in p):
        raise ValueError("Specimen registration requires a finite observed polygon")
    xs, ys = zip(*polygon_px, strict=True)
    left, top = math.floor(min(xs))-padding, math.floor(min(ys))-padding
    right, bottom = math.ceil(max(xs))+padding+1, math.ceil(max(ys))+padding+1
    if right-left < 12:
        left, right = math.floor((left+right)/2)-6, math.floor((left+right)/2)+6
    if bottom-top < 12:
        top, bottom = math.floor((top+bottom)/2)-6, math.floor((top+bottom)/2)+6
    if image_size:
        left, top = max(0, left), max(0, top)
        right, bottom = min(image_size[0], right), min(image_size[1], bottom)
    return PixelROI(left, top, right, bottom)


def register_specimen(
    reference: object, current: object, *, polygon_px: list[tuple[float, float]],
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    """Locate an observed specimen using polygon-supported optical contrast.

    Unlike background registration, only pixels inside the specimen contribute
    to the matched signature. A surrounding annulus estimates illumination and
    background colour, and positive signed contrast support is required at the
    candidate. Stationary textured substrate cannot substitute for a deleted
    specimen. The polygon must come from a prior optical observation; identity,
    provenance and plane calibration remain the caller's responsibility.

    """
    config = config or RegistrationConfig()
    try:
        ref, cur = load_pixels(reference), load_pixels(current)
        roi = polygon_roi(polygon_px, image_size=ref.size)
    except (OSError, ValueError):
        return RegistrationResult(False, loss_reason="invalid_specimen_reference")
    if ref.size != cur.size:
        return RegistrationResult(False, loss_reason="image_size_changed", roi=roi)
    if (config.max_translation_px <= 0 or config.max_rotation_deg < 0 or config.pixel_aspect_ratio <= 0
            or config.rotation_step_deg <= 0 or not all(math.isfinite(v) for v in
                (config.max_translation_px, config.max_rotation_deg, config.rotation_step_deg,
                 config.nominal_rotation_deg, config.pixel_aspect_ratio))):
        raise ValueError("Invalid specimen registration bounds")
    scale = min(1., 480/max(ref.size))
    size = tuple(round(value*scale) for value in ref.size)
    ref = ref.resize(size).filter(ImageFilter.GaussianBlur(_SPECIMEN_BLUR_RADIUS_PX))
    cur = cur.resize(size).filter(ImageFilter.GaussianBlur(_SPECIMEN_BLUR_RADIUS_PX))
    polygon = [(x*scale, y*scale) for x, y in polygon_px]
    xs, ys = zip(*polygon, strict=True)
    if min(xs) < 4 or min(ys) < 4 or max(xs) >= size[0]-4 or max(ys) >= size[1]-4:
        return RegistrationResult(False, loss_reason="specimen_outline_clipped", roi=roi)
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    inside_mask, expanded = mask.filter(ImageFilter.MinFilter(3)), mask.filter(ImageFilter.MaxFilter(7))
    box = polygon_roi(polygon, padding=4, image_size=size)
    inside_pixels, expanded_pixels = inside_mask.load(), expanded.load()
    mask_pixels = mask.load()
    inside, outside = [], []
    for y in range(box.top, box.bottom):
        for x in range(box.left, box.right):
            if inside_pixels[x, y]:
                inside.append((x, y))
            elif expanded_pixels[x, y] and not mask_pixels[x, y]:
                outside.append((x, y))
    if len(inside) < 4 or len(outside) < 12:
        return RegistrationResult(False, loss_reason="specimen_too_small", roi=roi)

    def subsample(points: list[tuple[int, int]], maximum: int) -> list[tuple[int, int]]:
        count = min(len(points), maximum)
        return [points[min(len(points)-1, int((index+.5)*len(points)/count))]
                for index in range(count)]

    inside = subsample(inside, 96)
    outside = subsample(outside, 48)
    ref_pixels = ref.load()
    current_bytes = iter(cur.tobytes())
    current_pixels = tuple(zip(current_bytes, current_bytes, current_bytes, strict=True))
    background = tuple(statistics.mean(ref_pixels[p][channel] for p in outside) for channel in range(3))
    signature = [tuple(ref_pixels[p][channel]-background[channel] for channel in range(3))
                 for p in inside]
    energy = sum(value*value for point in signature for value in point)
    if energy < len(signature)*3*9:
        return RegistrationResult(False, loss_reason="no_specimen_contrast", roi=roi)
    center = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2)
    contrast_mean = tuple(statistics.mean(p[c] for p in signature) for c in range(3))
    norm = math.sqrt(sum(value*value for value in contrast_mean))
    if norm < 3:
        return RegistrationResult(False, loss_reason="no_specimen_boundary_contrast", roi=roi)
    unit = tuple(value/norm for value in contrast_mean)
    contour = []
    for y in range(box.top+1, box.bottom-1):
        for x in range(box.left+1, box.right-1):
            if not mask_pixels[x, y] or all(mask_pixels[x+u, y+v]
                                               for u, v in ((-1,0),(1,0),(0,-1),(0,1))):
                continue
            vx, vy = x-center[0], y-center[1]
            distance = math.hypot(vx, vy)
            if distance < 1:
                continue
            vx, vy = vx/distance, vy/distance
            inner = round(x-vx), round(y-vy)
            outer = round(x+2*vx), round(y+2*vy)
            amplitude = sum((ref_pixels[inner][c]-ref_pixels[outer][c])*unit[c]
                            for c in range(3))
            if amplitude >= 1.5:
                quadrant = (0 if vx > 0 else 1) if abs(vx) >= abs(vy) else (2 if vy > 0 else 3)
                contour.append((inner, outer, amplitude, quadrant))
    if len(contour) < 8:
        return RegistrationResult(False, loss_reason="specimen_boundary_unobservable", roi=roi)
    # Thin the boundary in angular order. Raster order alternates each row's
    # left and right edge, so an even stride kept one side only and a thin
    # sliver (a518 seed 2 Gr) lost a whole direction even against itself.
    contour.sort(key=lambda item: math.atan2(item[0][1]-center[1], item[0][0]-center[0]))
    contour = [contour[index] for index in range(0, len(contour), max(1, len(contour)//32))][:48]
    rotated = {}
    noise_gains: dict[tuple[int, int, float], float] = {}
    contour_residuals: dict[tuple[int, int, float], float] = {}


    def measure(dx: int, dy: int, angle: float) -> float:
        if angle not in rotated:
            total = angle+config.nominal_rotation_deg
            sine, cosine = math.sin(math.radians(total)), math.cos(math.radians(total))

            def turn(points):
                return [(round(center[0]+(x-center[0])*cosine+(y-center[1])*sine*config.pixel_aspect_ratio),
                         round(center[1]-(x-center[0])*sine/config.pixel_aspect_ratio+(y-center[1])*cosine))
                        for x, y in points]

            rotated[angle] = (turn(inside), turn(outside), turn([p[0] for p in contour]),
                              turn([p[1] for p in contour]))
        ip, op, ci, co = rotated[angle]
        points = [(x+dx, y+dy) for x, y in ip+op+ci+co]
        if any(x < 0 or y < 0 or x >= size[0] or y >= size[1] for x, y in points):
            return -1.
        values = [current_pixels[y*size[0]+x] for x, y in points]
        boundary_start = len(ip)+len(op)
        current_background = tuple(sum(p[c] for p in values[len(ip):boundary_start])/len(op)
                                   for c in range(3))
        gains = tuple(max(.5, min(2., background[c]/max(1., current_background[c]))) for c in range(3))
        supported, counts = [0]*4, [0]*4
        contour_error = 0.
        for index, (_, _, amplitude, quadrant) in enumerate(contour):
            inner, outer = values[boundary_start+index], values[boundary_start+len(ci)+index]
            observed = sum((inner[c]-outer[c])*gains[c]*unit[c] for c in range(3))
            counts[quadrant] += 1
            supported[quadrant] += int(observed >= max(.7, amplitude*.2))
            contour_error += (observed-amplitude)**2
        visible_directions = sum(count >= 2 for count in counts)
        if (visible_directions < 4 or sum(counts) < 8
                or sum(supported)/sum(counts) < .72
                or any(count > 0 and support/count < .5
                       for support, count in zip(supported, counts, strict=True))):
            return -1.
        error, support = 0., 0.
        for wanted, observed in zip(signature, values[:len(ip)], strict=True):
            for channel in range(3):
                contrast = (observed[channel]-current_background[channel])*gains[channel]
                error += (contrast-wanted[channel])**2
                support += contrast*wanted[channel]
        if support/energy < .45:
            return -1.
        noise_gains[dx, dy, angle] = 1.
        contour_residuals[dx, dy, angle] = contour_error/len(contour)
        return max(-1., 1-error/(energy+len(signature)*3*12))

    measurements: dict[tuple[int, int, float], float] = {}

    def evaluate(dx: int, dy: int, angle: float) -> float:
        key = dx, dy, angle
        if key not in measurements:
            measurements[key] = measure(dx, dy, angle)
        return measurements[key]

    # The translation bound is declared on the working raster (longest side
    # 480 px, the field the bound was set for), so it keeps its distance on a
    # full-sensor v1 capture of the same field: 40 px is ~58 um at 20X on
    # either raster, not 7 um on the 4K one (seed 3 r5, #204).
    limit = max(1, math.floor(config.max_translation_px))
    step = 2 if limit >= 8 else 1
    grid = sorted(set(range(-limit, limit+1, step)) | {0, limit})
    coarse = sorted(((evaluate(x, y, 0.), x, y) for y in grid for x in grid), reverse=True)
    seeds = []
    for candidate in coarse:
        if all(math.hypot(candidate[1]-s[1], candidate[2]-s[2]) > 2 for s in seeds):
            seeds.append(candidate)
        if len(seeds) == 5:
            break
    count = math.ceil(config.max_rotation_deg/config.rotation_step_deg)
    angles = {max(-config.max_rotation_deg, min(config.max_rotation_deg,
                  index*config.rotation_step_deg)) for index in range(-count, count+1)}
    refined = {}
    for _, x, y in seeds:
        for dy in range(max(-limit, y-step), min(limit, y+step)+1):
            for dx in range(max(-limit, x-step), min(limit, x+step)+1):
                for angle in sorted(angles):
                    refined[dx, dy, angle] = evaluate(dx, dy, angle)
    ranked = sorted(((quality, -abs(angle), -math.hypot(x, y), x, y, angle)
                     for (x, y, angle), quality in refined.items()), reverse=True)
    # Weak contrast plus pixel quantization may change a score by a few
    # hundredths without resolving rotation. Preserve the reference angle
    # among score-equivalent matches; this is not an orientation certificate.
    # Callers requiring translation only should explicitly set rotation to 0.
    equivalent = [item for item in ranked if item[0] >= ranked[0][0]-.015]
    # A broad, uniform specimen interior may score equally well across several
    # translations. Localize its observed boundary among those equivalent
    # identity matches, rather than preferring a stationary interior patch.
    chosen = max(equivalent, key=lambda item: (
        item[1], -contour_residuals.get((item[3], item[4], item[5]), math.inf), item[0], item[2]))
    quality, _, _, dx, dy, angle = chosen
    separation = max(3, min(max(xs)-min(xs), max(ys)-min(ys))*.3)
    alternatives = [q for q, _, _, x, y, _ in ranked if math.hypot(x-dx, y-dy) >= separation]
    ambiguity = quality-max(alternatives, default=-1.)
    # Renderer/camera rasterization also quantizes at the native image grid.
    # Native and working-grid geometric reserves are independent of the
    # photometric penalty; changing contrast does not change either grid pitch.
    noise_gain = noise_gains.get((dx, dy, angle), 1.)
    pose_gain, pose_floor = 1., 0.
    uncertainty = 1./scale + .5 + max(
        (2*(1-max(0., quality))+.05/max(.01, ambiguity))/scale*noise_gain*pose_gain, pose_floor)
    margin = min(min(xs)+dx, min(ys)+dy, size[0]-max(xs)-dx, size[1]-max(ys)-dy)/scale
    if config.nominal_rotation_deg or config.pixel_aspect_ratio != 1:
        total = math.radians(angle+config.nominal_rotation_deg)
        rotated_polygon = [(center[0]+(x-center[0])*math.cos(total)+(y-center[1])*math.sin(total)*config.pixel_aspect_ratio+dx,
                            center[1]-(x-center[0])*math.sin(total)/config.pixel_aspect_ratio+(y-center[1])*math.cos(total)+dy)
                           for x, y in polygon]
        margin = min(min(x for x, _ in rotated_polygon), min(y for _, y in rotated_polygon),
                     size[0]-max(x for x, _ in rotated_polygon), size[1]-max(y for _, y in rotated_polygon))/scale
    reason = None
    if quality < config.min_quality:
        reason = "specimen_not_present"
    elif ambiguity < config.min_ambiguity:
        reason = "ambiguous_specimen"
    elif abs(dx) >= limit or abs(dy) >= limit:
        reason = "translation_search_limit"
    elif config.max_rotation_deg and abs(angle) >= config.max_rotation_deg:
        reason = "rotation_search_limit"
    elif margin <= uncertainty:
        reason = "specimen_at_fov_edge"
    return RegistrationResult(
        reason is None, dx/scale, dy/scale, angle+config.nominal_rotation_deg, quality, ambiguity, uncertainty,
        max(0., margin), reason, roi,
    )


@dataclass(frozen=True)
class RasterContactEvidence:
    has_contact: bool
    centroid_px: tuple[float, float] | None = None
    radius_px: float | None = None
    angular_coverage: float = 0.0
    rings: int = 0
    film_intact: bool | None = None
    quality: float = 0.0
    failure_reason: str | None = None
    outer_radius_px: float | None = None
    interface_uncertainty_px: float = 0.
    boundary_method: str = "palette_annulus"
    boundary_diagnostics: tuple[float, float, float, float, float, float] | None = None


def _annulus_boundaries(
    coordinates: set[tuple[int, int]], width: int, height: int,
    provisional: tuple[float, float],
    pixels: list[tuple[int, int, int]],
) -> tuple[float, float, float, float, float] | None:
    """Fit one centre and separate inner/outer radii to actual observed arcs.

    Thick-band area moments bias a clipped annulus toward its visible area.
    Group-centred normal equations eliminate each boundary's radius intercept,
    leaving a shared centre. Image-border cut edges never enter the fit.
    The final retained arc residual is a pixel-derived exclusion heuristic; it
    does not change measured contact radii or certify a statistical bound.
    """
    groups: list[list[tuple[int, int]]] = [[], []]
    for x, y in sorted(coordinates):
        if x < 2 or y < 2 or x >= width-2 or y >= height-2:
            continue
        neighbours = [(u, v) for u, v in ((-1,0),(1,0),(0,-1),(0,1))
                      if (x+u, y+v) not in coordinates]
        if not neighbours:
            continue
        nx, ny = sum(p[0] for p in neighbours), sum(p[1] for p in neighbours)
        vx, vy = x-provisional[0], y-provisional[1]
        direction = vx*nx+vy*ny
        if abs(direction) > .25*math.hypot(vx, vy):
            length = math.hypot(nx, ny)
            xx, yy = round(x+2*nx/length), round(y+2*ny/length)
            # Palette membership alone acquires spurious edges where camera
            # vignette/quantization nudges substrate pixels across the mask.
            # Require an actual brightening beyond the synthetic camera's
            # small noise/quantization steps in the observed outward normal.
            if (0 <= xx < width and 0 <= yy < height
                    and pixels[yy*width+xx][1]-pixels[y*width+x][1] > 8):
                groups[int(direction > 0)].append((x, y))
    if min(map(len, groups)) < 6:
        return None
    fitted = None
    for _ in range(3):
        suu = svv = suv = bu = bv = 0.
        for group in groups:
            mx = statistics.mean(p[0] for p in group)
            my = statistics.mean(p[1] for p in group)
            mq = statistics.mean(x*x+y*y for x, y in group)
            for x, y in group:
                u, v, q = x-mx, y-my, x*x+y*y-mq
                suu += u*u
                svv += v*v
                suv += u*v
                bu += u*q/2
                bv += v*q/2
        determinant = suu*svv-suv*suv
        if determinant <= 1e-6:
            return None
        cx, cy = (bu*svv-bv*suv)/determinant, (bv*suu-bu*suv)/determinant
        radii = [statistics.mean(math.hypot(x-cx, y-cy) for x, y in group)
                 for group in groups]
        if radii[0] >= radii[1]:
            return None
        fitted = cx, cy, *radii
        trimmed = []
        for group, radius in zip(groups, radii, strict=True):
            residuals = [abs(math.hypot(x-cx, y-cy)-radius) for x, y in group]
            cutoff = max(1.25, 3*statistics.median(residuals))
            trimmed.append([point for point, error in zip(group, residuals, strict=True)
                            if error <= cutoff])
        if min(map(len, trimmed)) < 6:
            return None
        groups = trimmed
    return (*fitted, max(abs(math.hypot(x-cx, y-cy)-radius)
                         for group, radius in zip(groups, radii, strict=True) for x, y in group))


def _component_annulus(
    coordinates: set[tuple[int, int]], width: int, height: int,
    pixels: list[tuple[int, int, int]],
) -> tuple[float, float, float, float, float] | None:
    """Seed boundary classification from area moments, then fit actual arcs."""
    mx = statistics.mean(x for x, y in coordinates)
    my = statistics.mean(y for x, y in coordinates)
    us, vs = [x-mx for x, y in coordinates], [y-my for x, y in coordinates]
    suu, svv = sum(u*u for u in us), sum(v*v for v in vs)
    suv = sum(u*v for u, v in zip(us, vs, strict=True))
    determinant = suu*svv-suv*suv
    if determinant <= 1e-6:
        return None
    bu = sum(u*(u*u+v*v) for u, v in zip(us, vs, strict=True))/2
    bv = sum(v*(u*u+v*v) for u, v in zip(us, vs, strict=True))/2
    center = mx+(bu*svv-bv*suv)/determinant, my+(bv*suu-bu*suv)/determinant
    return _annulus_boundaries(coordinates, width, height, center, pixels)


def contact_front_evidence(source: object) -> RasterContactEvidence:
    """Prefer palette evidence; bound one background-relative annulus check."""
    original = _contact_front_evidence(source)
    if original.has_contact:
        return original
    contrast = _contact_front_evidence(source, local_contrast=True)
    return contrast if contrast.has_contact else original


def contact_front_domain(source: object):
    """Original observed front pixel-cell bounds; no fitted circle is filled."""
    original, bounds = _contact_front_evidence(source, with_domain=True)
    if original.has_contact:
        return original, bounds
    contrast, other = _contact_front_evidence(source, local_contrast=True, with_domain=True)
    return (contrast, other) if contrast.has_contact else (original, bounds)


def _contact_front_evidence(source: object, *, local_contrast: bool = False, with_domain: bool = False):
    """Conservative palette/shape evidence from captured RGB pixels only.

    The renderer's strong purple front plus brighter filled interior differs
    from its faint alternating precontact rings. Thresholds describe that raster
    appearance; they are not a real microscope classifier. An unbroken visible
    boundary cannot prove film integrity, so film_intact stays None unless a
    dark internal discontinuity is observed (False).
    """
    image = load_pixels(source)
    # Preserve the renderer's two-pixel first front; aggressive downsampling
    # can erase that boundary and confuse actual contact with faint fringes.
    scale = min(1.0, 800/max(image.size))
    image = image.resize(tuple(max(1, round(v*scale)) for v in image.size))
    channel_bytes = iter(image.tobytes())
    pixels = list(zip(channel_bytes, channel_bytes, channel_bytes, strict=True))
    strong = {i for i, (r, g, b) in enumerate(pixels)
              if (50 < r < 105 and 38 < g < 84 and 85 < b < 149 and b-r > 25
                  and r-g > 8 and b-r > 2.16*(r-g))}
    if local_contrast:
        local_green = image.getchannel("G").filter(ImageFilter.MaxFilter(7)).tobytes()
        dark_ridge = {i for i, (r,g,b) in enumerate(pixels)
                      if b-r > r-g > 0 and local_green[i]-g > 8}
    weak = sum(1 for r, g, b in pixels if 82 <= g < 98 and b-r > 28 and r-g > 10)
    components = []
    while strong:
        seed = min(strong)
        strong.remove(seed)
        component, pending = [seed], [seed]
        while pending:
            i = pending.pop()
            x, y = i % image.width, i // image.width
            for xx, yy in ((x-1,y), (x+1,y), (x,y-1), (x,y+1),
                           (x-1,y-1), (x+1,y-1), (x-1,y+1), (x+1,y+1)):
                other = yy*image.width+xx
                if 0 <= xx < image.width and 0 <= yy < image.height and other in strong:
                    strong.remove(other)
                    component.append(other)
                    pending.append(other)
        if len(component) >= 12:
            components.append(component)
    # A specimen may interrupt a clipped front into separate observed arcs.
    # Associate only mutually consistent independently fitted inner/outer arcs;
    # chains of gradually different centres/radii cannot join unrelated circles.
    associated: list[list[tuple[list[int], tuple[float, float, float, float, float]]]] = []
    reserve = 3*scale
    ordered = sorted(components, key=len, reverse=True)
    for component in ordered[:12]:
        coordinates = {(index % image.width, index // image.width) for index in component}
        fitted = _component_annulus(coordinates, image.width, image.height, pixels)
        if fitted is None:
            continue
        for group in associated:
            if all(math.dist(fitted[:2], previous[:2]) <= reserve
                   and abs(fitted[2]-previous[2]) <= reserve
                   and abs(fitted[3]-previous[3]) <= reserve for _, previous in group):
                group.append((component, fitted))
                break
        else:
            associated.append([(component, fitted)])
    components = [[index for component, _ in group for index in component] for group in associated]
    # The association search is bounded; smaller components retain their
    # original independent path so a two-pixel first front is not discarded
    # merely because larger residue patches were present elsewhere.
    components.extend(ordered[12:])
    best: RasterContactEvidence | None = None
    best_bounds = None
    for component in sorted(components, key=len, reverse=True):
        xs, ys = [i % image.width for i in component], [i // image.width for i in component]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        fitted = _component_annulus(set(zip(xs, ys, strict=True)), image.width, image.height, pixels)
        if fitted is None:
            continue
        cx, cy, inner_radius, outer_radius, fit_error = fitted
        seed_radius = (inner_radius+outer_radius)/2
        seed_coverage = len({int((math.atan2(y-cy,x-cx)+math.pi)*36/(2*math.pi)) % 36
                             for x,y in zip(xs,ys,strict=True)
                             if .72*seed_radius < math.hypot(x-cx,y-cy) < 1.18*seed_radius})/36
        refit_change = (0.,0.,0.)
        if local_contrast:
            seed_clipped = x0 < 2 or y0 < 2 or x1 >= image.width-2 or y1 >= image.height-2
            # This extension repairs insufficient viewport-clipped arc coverage
            # only. A complete original arc that failed a contact/intactness
            # check cannot gain a different outcome from boundary extension.
            if not seed_clipped or seed_coverage >= (.6 if seed_radius < 6 else .45):
                continue
            # A partially observed original arc supplies only a bounded search
            # corridor. New boundary evidence must be independently dark in
            # local RGB and refit the same nested annulus; no physical ellipse
            # or renderer attenuation is supplied.
            expanded = {(i % image.width, i // image.width) for i in dark_ridge
                        if inner_radius-reserve <= math.hypot(i % image.width-cx, i // image.width-cy) <= outer_radius+reserve}
            recovered = _annulus_boundaries(expanded, image.width, image.height, (cx, cy), pixels)
            if (recovered is None or math.dist(recovered[:2], fitted[:2]) > reserve
                    or abs(recovered[2]-inner_radius) > reserve or abs(recovered[3]-outer_radius) > reserve):
                continue
            refit_change = (math.dist(recovered[:2],fitted[:2])/scale,
                            (recovered[2]-inner_radius)/scale,(recovered[3]-outer_radius)/scale)
            cx, cy, inner_radius, outer_radius, fit_error = recovered
            xs, ys = [x for x,y in expanded], [y for x,y in expanded]
            x0, x1, y0, y1 = min(xs),max(xs),min(ys),max(ys)
        radius = (inner_radius+outer_radius)/2
        radial = [math.hypot(x-cx, y-cy) for x, y in zip(xs, ys, strict=True)]
        if radius < 2.5 or statistics.pstdev(radial)/radius > 0.19:
            continue
        bins = {int((math.atan2(y-cy, x-cx)+math.pi)*36/(2*math.pi)) % 36
                for x, y in zip(xs, ys, strict=True)
                if 0.72*radius < math.hypot(x-cx, y-cy) < 1.18*radius}
        coverage = len(bins)/36
        clipped = x0 < 2 or y0 < 2 or x1 >= image.width-2 or y1 >= image.height-2
        if coverage < (0.6 if radius < 6 else 0.45 if clipped else 0.72):
            continue
        inner, outer, cracks = [], [], 0
        for y in range(max(0, y0-5), min(image.height, y1+6)):
            for x in range(max(0, x0-5), min(image.width, x1+6)):
                distance = math.hypot(x-cx, y-cy)/radius
                r, g, b = pixels[y*image.width+x]
                if distance < 0.65:
                    inner.append(g)
                    cracks += int(max(r, g, b) < 65)
                elif 1.08 < distance < 1.3:
                    outer.append(g)
        if not inner or not outer or statistics.median(inner)-statistics.median(outer) < 4:
            continue
        broken = cracks >= max(4, len(inner)*0.006)
        evidence = RasterContactEvidence(
            # The light interior is physical contact. The dark outer annulus
            # has display thickness and cannot enlarge the covered region.
            True, (cx/scale, cy/scale), inner_radius/scale, coverage, 0,
            False if broken else None, coverage,
            "boundary_clipped" if clipped else "film_discontinuity" if broken else None,
            outer_radius_px=outer_radius/scale, interface_uncertainty_px=fit_error/scale,
            boundary_method="observed_local_contrast_annulus" if local_contrast else "palette_annulus",
            boundary_diagnostics=(seed_coverage,coverage,*refit_change,reserve/scale if local_contrast else 0.),
        )
        if best is None or evidence.quality > best.quality:
            best = evidence
            best_bounds = (min(xs)-.5, min(ys)-.5, max(xs)+.5, max(ys)+.5)
    if best is not None:
        return (best, best_bounds) if with_domain else best
    result = RasterContactEvidence(False, rings=1 if weak > 30 else 0,
                                   failure_reason="fringes_only" if weak > 30 else "no_contact_front")
    return (result, None) if with_domain else result
