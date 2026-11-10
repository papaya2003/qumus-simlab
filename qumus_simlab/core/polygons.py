"""Small convex-polygon operations in mm; no physical or chemical assumptions.

Only convex, hole-free flakes are supported. Reject other geometry instead of
silently replacing it by a hull. All public model inputs validate this contract.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

Point = tuple[float, float]
Polygon = tuple[Point, ...]
EPS = 1e-14  # mm squared; far below a square micrometre


def cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def signed_area(p: Polygon) -> float:
    # Translate before summing to avoid cancellation for small, offset flakes.
    return sum(cross(p[0], p[i], p[i + 1]) for i in range(1, len(p) - 1)) / 2 if p else 0


def area_um2(p: Polygon) -> float:
    return abs(signed_area(p)) * 1e6


def convex_polygon(points: Iterable[Point]) -> Polygon:
    p = tuple((float(x), float(y)) for x, y in points)
    if not 3 <= len(p) <= 128 or not all(math.isfinite(v) for xy in p for v in xy):
        raise ValueError("a flake needs 3..128 finite vertices")
    area = signed_area(p)
    if not math.isfinite(area):
        raise ValueError("polygon area must be finite")
    if len(set(p)) != len(p) or abs(area) <= EPS:
        raise ValueError("a flake must have distinct vertices and positive area")
    if signed_area(p) < 0:
        p = tuple(reversed(p))
    if any(cross(a, b, q) < -EPS for a, b in zip(p, p[1:] + p[:1], strict=True) for q in p):
        raise ValueError("only convex, simple, hole-free flake outlines are supported")
    return p


def hull(points: Iterable[Point]) -> Polygon:
    """Convex hull used only to synthesize new shapes, never to repair inputs."""
    p = sorted(set(points))
    halves = []
    for order in (p, p[::-1]):
        half: list[Point] = []
        for q in order:
            while len(half) >= 2 and cross(half[-2], half[-1], q) <= 0:
                half.pop()
            half.append(q)
        halves.extend(half[:-1])
    return convex_polygon(halves)


def intersection(subject: Polygon, clip: Polygon) -> Polygon:
    """Sutherland-Hodgman clipping of validated counterclockwise polygons."""
    out = list(subject)
    for a, b in zip(clip, clip[1:] + clip[:1], strict=True):
        incoming, out = out, []
        if not incoming:
            break
        prev = incoming[-1]
        dp = cross(a, b, prev)
        for cur in incoming:
            dc = cross(a, b, cur)
            # Zero tolerance avoids adding a thin exterior strip to the overlap.
            if (dc >= 0) != (dp >= 0):
                t = dp / (dp - dc)
                out.append((prev[0] + t * (cur[0] - prev[0]),
                            prev[1] + t * (cur[1] - prev[1])))
            if dc >= 0:
                out.append(cur)
            prev, dp = cur, dc
    clean: list[Point] = []
    for q in out:
        if not clean or math.dist(q, clean[-1]) > 1e-12:
            clean.append(q)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) <= 1e-12:
        clean.pop()
    p = tuple(clean)
    return p if len(p) >= 3 and signed_area(p) > EPS else ()


def erode_convex(points: Iterable[Point], distance_mm: float) -> Polygon:
    """Convex polygon eroded by a Euclidean disk, via exact shifted halfplanes.

    No disk tessellation or hull repair is used. A collapsed core has zero area
    and returns empty; malformed input or nonfinite arithmetic is rejected.
    """
    if (isinstance(distance_mm, bool) or not isinstance(distance_mm, (int, float))
            or not math.isfinite(distance_mm) or distance_mm < 0):
        raise ValueError("erosion distance must be finite and nonnegative")
    polygon = convex_polygon(points)
    if distance_mm == 0:
        return polygon
    out = list(polygon)
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        length = math.dist(a, b)
        if not math.isfinite(length) or length <= 0:
            raise ValueError("erosion needs finite nonzero edges")
        nx, ny = -(b[1]-a[1])/length, (b[0]-a[0])/length

        def side(q, a=a, nx=nx, ny=ny):
            value = nx*(q[0]-a[0])+ny*(q[1]-a[1])-distance_mm
            if not math.isfinite(value):
                raise ValueError("nonfinite erosion halfplane")
            return value

        incoming, out = out, []
        if not incoming:
            return ()
        previous = incoming[-1]
        before = side(previous)
        for current in incoming:
            after = side(current)
            if (after >= 0) != (before >= 0):
                # Opposite signs make this denominator nonzero, including for
                # redundant parallel supporting edges.
                denominator = before-after
                if not math.isfinite(denominator) or denominator == 0:
                    raise ValueError("invalid erosion intersection denominator")
                fraction = before/denominator
                point = (previous[0]+fraction*(current[0]-previous[0]),
                         previous[1]+fraction*(current[1]-previous[1]))
                if not all(math.isfinite(value) for value in point):
                    raise ValueError("nonfinite erosion intersection")
                out.append(point)
            if after >= 0:
                out.append(current)
            previous, before = current, after
    clean: list[Point] = []
    for point in out:
        if not clean or math.dist(point, clean[-1]) > 1e-12:
            clean.append(point)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) <= 1e-12:
        clean.pop()
    core = tuple(clean)
    return convex_polygon(core) if len(core) >= 3 and signed_area(core) > EPS else ()


def transform(p: Polygon, *, xy: Point = (0, 0), rotation_deg: float = 0) -> Polygon:
    if not all(math.isfinite(v) for v in (*xy, rotation_deg)):
        raise ValueError("transform must be finite")
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    return tuple((xy[0] + c * x - s * y, xy[1] + s * x + c * y) for x, y in p)


def contract(p: Polygon, fraction: float, anchor: Point | None = None) -> Polygon:
    """A connected subset with the requested area fraction, not a growth law."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("area fraction must lie in [0, 1]")
    if not p or fraction == 0:
        return ()
    anchor = anchor or (sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p))
    if any(cross(a, b, anchor) < -EPS for a, b in zip(p, p[1:] + p[:1], strict=True)):
        raise ValueError("contraction anchor must be inside polygon")
    scale = math.sqrt(fraction)
    return tuple((anchor[0] + (x - anchor[0]) * scale,
                  anchor[1] + (y - anchor[1]) * scale) for x, y in p)


def rectangle(width: float, height: float, center: Point = (0, 0)) -> Polygon:
    if not all(math.isfinite(v) for v in (width, height, *center)) or min(width, height) <= 0:
        raise ValueError("rectangle dimensions must be finite and positive")
    x, y = center
    return convex_polygon(((x - width / 2, y - height / 2), (x + width / 2, y - height / 2),
                           (x + width / 2, y + height / 2), (x - width / 2, y + height / 2)))


def contains(outer: Polygon, inner: Polygon) -> bool:
    """Exact containment of validated convex polygons; no physical margin inferred."""
    return bool(outer and inner) and all(cross(a, b, q) >= -EPS
        for a, b in zip(outer, outer[1:] + outer[:1], strict=True) for q in inner)


def containment_clearance(outer: Polygon, inner: Polygon) -> float:
    """Signed minimum distance to outer edges, in mm, for convex polygons.

    Positive clearance ranks nominal fits; it is not a calibrated positioning
    tolerance and does not replace checking the actual transformed outlines.
    """
    return min(cross(a, b, point) / math.dist(a, b)
               for a, b in zip(outer, outer[1:] + outer[:1], strict=True) for point in inner)


def rigid_polygon_error(expected: Polygon, actual: Polygon) -> dict:
    """Proper rotation/translation fit only: no scale, reflection, hull or crop.

    Cyclic vertex order is immaterial. Symmetric bodies can have geometrically
    indistinguishable orientations; this is not crystallographic twist metrology.
    """
    expected, actual = convex_polygon(expected), convex_polygon(actual)
    if len(expected) != len(actual):
        return {"shape_residual_mm": None, "rotation_deg": None, "translation_mm": None}
    a = tuple(sum(p[i] for p in actual)/len(actual) for i in (0, 1))
    b = tuple(sum(p[i] for p in expected)/len(expected) for i in (0, 1))
    centered_actual = tuple((x-a[0], y-a[1]) for x, y in actual)
    best = None
    for shift in range(len(expected)):
        ordered = expected[shift:]+expected[:shift]
        centered_expected = tuple((x-b[0], y-b[1]) for x, y in ordered)
        dot = sum(x*u+y*v for (x, y), (u, v) in zip(centered_expected, centered_actual, strict=True))
        cross_sum = sum(x*v-y*u for (x, y), (u, v) in zip(centered_expected, centered_actual, strict=True))
        angle = math.atan2(cross_sum, dot)
        rotated = transform(centered_expected, rotation_deg=math.degrees(angle))
        error = max(math.dist(p, q) for p, q in zip(rotated, centered_actual, strict=True))
        candidate = (error, abs(angle), math.degrees(angle))
        if (best is None or error < best[0]-1e-12
                or abs(error-best[0]) <= 1e-12 and abs(angle) < best[1]):
            best = candidate
    return {"shape_residual_mm": best[0], "rotation_deg": best[2],
            "translation_mm": [a[0]-b[0], a[1]-b[1]]}

def cap_surface_point(apex, cap_center, half_size, radius):
    """Closest point on rectangle INTERSECT physical circular cap, or None.

    All inputs are in the chip's local axes. Curvature is deliberately absent.
    """
    values = (*apex, *cap_center, *half_size, radius)
    if (len(apex) != 2 or len(cap_center) != 2 or len(half_size) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or radius <= 0 or min(half_size) <= 0):
        raise ValueError("Cap support needs finite positive rectangle/circle geometry")
    x, y = apex
    cx, cy = cap_center
    hw, hh = half_size
    candidates = []
    def add(px, py):
        if (-hw-1e-12 <= px <= hw+1e-12 and -hh-1e-12 <= py <= hh+1e-12
                and math.hypot(px-cx, py-cy) <= radius+1e-12):
            candidates.append((px, py))
    add(max(-hw, min(hw, x)), max(-hh, min(hh, y)))
    distance = math.hypot(x-cx, y-cy)
    if distance:
        add(cx+radius*(x-cx)/distance, cy+radius*(y-cy)/distance)
    for px in (-hw, hw):
        for py in (-hh, hh):
            add(px, py)
        q = radius*radius-(px-cx)**2
        if q >= 0:
            add(px, cy+math.sqrt(q))
            add(px, cy-math.sqrt(q))
    for py in (-hh, hh):
        q = radius*radius-(py-cy)**2
        if q >= 0:
            add(cx+math.sqrt(q), py)
            add(cx-math.sqrt(q), py)
    return min(candidates, key=lambda p: math.dist(apex, p)) if candidates else None
