"""Validated polygon regions in mm, retaining every component and hole.

GEOS supplies topology/Boolean predicates. Original regions are never repaired,
filled, simplified or snapped. Strict material containment uses covers/distance.
Only bounded association may use positive inscribed buffer approximations; it
does not mutate evidence or certify a strict transfer margin. Legacy convex
helpers remain in ``polygons`` for their original callers.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, field_validator, model_validator
from shapely import STRtree, affinity
from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.geometry import Polygon as GEOSPolygon
from shapely.validation import explain_validity

from qumus_simlab.core.polygons import Point, convex_polygon

Ring = tuple[tuple[StrictFloat, StrictFloat], ...]
MAX_REGION_VERTICES = 2_000_000
MAX_REGION_COMPONENTS = 65_536
ASSOCIATION_QUAD_SEGS = 32


def _ring(points: Iterable[Point], *, positive: bool) -> Ring:
    ring = tuple((float(x), float(y)) for x, y in points)
    if (len(ring) < 3 or len(set(ring)) != len(ring)
            or any(not math.isfinite(v) for point in ring for v in point)):
        raise ValueError("Region rings require distinct finite vertices without a closing duplicate")
    area = GEOSPolygon(ring).area
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Region ring has no finite positive area")
    signed = sum((ring[i][0]-ring[0][0])*(ring[(i+1)%len(ring)][1]-ring[0][1])
                 -(ring[(i+1)%len(ring)][0]-ring[0][0])*(ring[i][1]-ring[0][1])
                 for i in range(len(ring)))
    if (signed > 0) != positive:
        ring = tuple(reversed(ring))
    start = min(range(len(ring)), key=ring.__getitem__)
    return ring[start:]+ring[:start]


class RegionComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exterior: Ring
    holes: tuple[Ring, ...] = ()

    @field_validator("exterior")
    @classmethod
    def _exterior(cls, value):
        return _ring(value, positive=True)

    @field_validator("holes")
    @classmethod
    def _holes(cls, value):
        return tuple(sorted(_ring(ring, positive=False) for ring in value))

    @model_validator(mode="after")
    def _valid(self):
        geometry = GEOSPolygon(self.exterior, self.holes)
        if not geometry.is_valid:
            raise ValueError("Invalid region component: "+explain_validity(geometry))
        return self


class PolygonRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    version: Literal["polygon-region-v1"] = "polygon-region-v1"
    components: tuple[RegionComponent, ...]

    @field_validator("components")
    @classmethod
    def _ordered(cls, value):
        return tuple(sorted(value, key=lambda c: (c.exterior, c.holes)))

    @model_validator(mode="after")
    def _valid(self):
        count = sum(len(c.exterior)+sum(map(len, c.holes)) for c in self.components)
        if len(self.components) > MAX_REGION_COMPONENTS or count > MAX_REGION_VERTICES:
            raise ValueError("Region exceeds the declared geometry resource bound")
        geometry = _geos(self)
        if not geometry.is_valid:
            raise ValueError("Invalid region topology: "+explain_validity(geometry))
        return self


def _geos(value: PolygonRegion):
    parts = [GEOSPolygon(c.exterior, c.holes) for c in value.components]
    return MultiPolygon(parts) if parts else MultiPolygon()


def as_region(value) -> PolygonRegion:
    if isinstance(value, PolygonRegion):
        return value
    if isinstance(value, dict):
        return PolygonRegion.model_validate(value)
    # A bare ring retains its historical convex contract. New general geometry
    # must carry the explicit versioned region shape rather than reinterpret it.
    points = convex_polygon(value)
    return PolygonRegion(components=(RegionComponent(exterior=points),))


def region_dict(value) -> dict:
    return as_region(value).model_dump(mode="json")


def region_vertices(value) -> tuple[Point, ...]:
    return tuple(point for c in as_region(value).components for ring in (c.exterior, *c.holes)
                 for point in ring)


def _from_geos(geometry) -> PolygonRegion:
    if geometry.is_empty:
        return PolygonRegion(components=())
    if isinstance(geometry, GEOSPolygon):
        return PolygonRegion(components=(RegionComponent(
            exterior=tuple(geometry.exterior.coords)[:-1],
            holes=tuple(tuple(r.coords)[:-1] for r in geometry.interiors)),))
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        parts = [c for g in geometry.geoms for c in _from_geos(g).components]
        return PolygonRegion(components=tuple(parts))
    # A zero-area point/line intersection is no two-dimensional material.
    return PolygonRegion(components=())


def transform_region(value, *, xy: Point = (0, 0), rotation_deg: float = 0) -> PolygonRegion:
    if (len(xy) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                            for v in (*xy, rotation_deg))):
        raise ValueError("Region transform must be finite")
    region = as_region(value)
    if rotation_deg == 0 and xy == (0, 0):
        return region
    geometry = affinity.rotate(_geos(region), rotation_deg, origin=(0, 0))
    return _from_geos(affinity.translate(geometry, xoff=xy[0], yoff=xy[1]))


def region_area_um2(value) -> float:
    return float(_geos(as_region(value)).area)*1e6


def region_covers(outer, inner) -> bool:
    a, b = _geos(as_region(outer)), _geos(as_region(inner))
    return not a.is_empty and not b.is_empty and bool(a.covers(b))


def region_intersection(first, second) -> PolygonRegion:
    return _from_geos(_geos(as_region(first)).intersection(_geos(as_region(second))))


def region_difference(first, second) -> PolygonRegion:
    return _from_geos(_geos(as_region(first)).difference(_geos(as_region(second))))


def region_clearance(outer, inner) -> float | None:
    """Distance of the entire contained material to all outer boundary rings.

    ``None`` means not contained, never a fabricated negative distance or zero.
    Holes and disconnected pieces participate in the same GEOS predicate.
    """
    a, b = _geos(as_region(outer)), _geos(as_region(inner))
    if a.is_empty or b.is_empty or not a.covers(b):
        return None
    return float(a.boundary.distance(b))


def region_covers_with_clearance(outer, inner, required_mm: float) -> bool:
    if type(required_mm) not in (int, float) or not math.isfinite(required_mm) or required_mm < 0:
        raise ValueError("Region clearance bound must be finite and nonnegative")
    clearance = region_clearance(outer, inner)
    return clearance is not None and clearance >= required_mm


def region_bidirectional_within(first, second, uncertainty_mm: float) -> bool:
    """Conservative whole-region association, never replacement source geometry.

    Positive round GEOS buffers use inscribed polygonal circle approximations,
    so their coverage is sufficient for the declared Euclidean displacement
    bound. They can conservatively refuse a true match near an arc chord. Both
    directions include holes and every disconnected component. A zero budget
    requires exact set equality through covers in both directions and never
    invokes ``buffer(0)``. No physical/calibration budget is inferred here.
    """
    if (type(uncertainty_mm) not in (int, float) or not math.isfinite(uncertainty_mm)
            or uncertainty_mm < 0):
        raise ValueError("Region association uncertainty must be finite and nonnegative")
    a, b = _geos(as_region(first)), _geos(as_region(second))
    if a.is_empty or b.is_empty:
        return False
    if uncertainty_mm == 0:
        return bool(a.covers(b) and b.covers(a))
    return bool(a.buffer(uncertainty_mm, quad_segs=ASSOCIATION_QUAD_SEGS, join_style="round").covers(b)
                and b.buffer(uncertainty_mm, quad_segs=ASSOCIATION_QUAD_SEGS, join_style="round").covers(a))


def region_compatible_with_bound(expected, observed, bound_mm: float) -> bool:
    """Named compatibility alias for source/observed association callers."""
    return region_bidirectional_within(expected, observed, bound_mm)


def quantize_region(value, *, dbu_mm: float = 1e-6) -> tuple[PolygonRegion, dict]:
    """Independent rounded derivation; refuse topology changes and lost rings.

    Original evidence is untouched. Both directed boundary distances are bounded
    by half a cell diagonal. Admission must account for this bound separately.
    """
    if type(dbu_mm) not in (int, float) or not math.isfinite(dbu_mm) or dbu_mm <= 0:
        raise ValueError("DBU size must be finite and positive")
    original = as_region(value)
    def rounded(ring):
        return tuple((round(x/dbu_mm)*dbu_mm, round(y/dbu_mm)*dbu_mm) for x, y in ring)
    parts = tuple(RegionComponent(exterior=rounded(c.exterior),
        holes=tuple(rounded(ring) for ring in c.holes)) for c in original.components)
    quantized = PolygonRegion(components=parts)
    before, after = _geos(original), _geos(quantized)
    bound = math.sqrt(2)*dbu_mm/2
    if before.boundary.hausdorff_distance(after.boundary) > bound+1e-12:
        raise ValueError("DBU quantization exceeds its declared boundary displacement")
    if (len(original.components) != len(quantized.components)
            or sorted(len(c.holes) for c in original.components) != sorted(len(c.holes) for c in quantized.components)):
        raise ValueError("DBU quantization changes region topology")
    def contacts(geometries):
        tree = STRtree(geometries)
        pairs = set()
        for i, geometry in enumerate(geometries):
            pairs.update((i, int(j)) for j in tree.query(geometry, predicate="intersects") if i < j)
            if len(pairs) > MAX_REGION_VERTICES:
                raise ValueError("Region boundary contacts exceed the declared resource bound")
        return pairs
    if contacts([GEOSPolygon(c.exterior, c.holes) for c in original.components]) != contacts(
            [GEOSPolygon(c.exterior, c.holes) for c in parts]):
        raise ValueError("DBU quantization changes component contact topology")
    for before_part, after_part in zip(original.components, parts, strict=True):
        if contacts([GEOSPolygon(ring).exterior for ring in (before_part.exterior, *before_part.holes)]) != contacts(
                [GEOSPolygon(ring).exterior for ring in (after_part.exterior, *after_part.holes)]):
            raise ValueError("DBU quantization changes hole contact topology")
    return quantized, {"version": "region-dbu-rounding-v1", "dbu_mm": dbu_mm,
                       "boundary_displacement_bound_mm": bound,
                       "operation": "round_each_original_vertex_no_repair"}
