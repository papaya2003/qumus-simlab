"""Persistent synthetic cleavage candidates and explicit transfer placement.

Parameters are illustrative distribution controls, not measured material yields.
No observation samples random state. Geometry uses chip-local millimetres.
"""

from __future__ import annotations

import hashlib
import json
import math
import random

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qumus_simlab.core.polygons import (
    Point,
    Polygon,
    area_um2,
    convex_polygon,
    hull,
    intersection,
    rectangle,
    transform,
)

MODEL_VERSION = "qualitative-material-v1"


def stable_key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Flake(FrozenModel):
    id: str = Field(min_length=1)
    material: str = Field(min_length=1)
    layers: int = Field(gt=0, strict=True)
    source_chip_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    outline_mm: Polygon  # absolute coordinates in the source chip frame

    @field_validator("outline_mm")
    @classmethod
    def valid_outline(cls, value: Polygon) -> Polygon:
        return convex_polygon(value)

    @property
    def area_um2(self) -> float:
        return area_um2(self.outline_mm)

    @property
    def center_mm(self) -> Point:
        return (sum(x for x, _ in self.outline_mm) / len(self.outline_mm),
                sum(y for _, y in self.outline_mm) / len(self.outline_mm))


class CleavageProfile(FrozenModel):
    median_area_um2: float = Field(250, gt=0)
    log_area_sigma: float = Field(0.75, ge=0, le=3)
    min_area_um2: float = Field(12, gt=0)
    max_area_um2: float = Field(12000, gt=0)
    max_aspect_ratio: float = Field(3.0, ge=1, le=10)
    layer_values: tuple[int, ...] = (1, 2, 3, 6, 20, 60)
    layer_weights: tuple[float, ...] = (0.12, 0.10, 0.08, 0.30, 0.25, 0.15)
    thick_area_multiplier: float = Field(2.5, ge=1, le=10)
    cluster_fraction: float = Field(0.65, ge=0, le=1)
    cluster_radius_mm: float = Field(0.08, gt=0)
    cluster_count: int = Field(4, ge=1, le=100)

    @model_validator(mode="after")
    def coherent(self) -> CleavageProfile:
        if not self.min_area_um2 <= self.median_area_um2 <= self.max_area_um2:
            raise ValueError("area limits must bracket the median")
        if (not self.layer_values or len(self.layer_values) != len(self.layer_weights)
                or any(x < 1 for x in self.layer_values)
                or any(x < 0 for x in self.layer_weights) or sum(self.layer_weights) <= 0):
            raise ValueError("layer classes need positive counts and nonnegative, nonzero weights")
        return self


def flake_outline(key: str, area: float, max_aspect_ratio: float = 3.0) -> Polygon:
    """Seeded angular outline, centered at the origin with exact geometric area."""
    if not math.isfinite(area) or area <= 0 or not 1 <= max_aspect_ratio <= 10:
        raise ValueError("finite positive area and aspect ratio in [1,10] required")
    rng = random.Random(key)
    aspect = rng.uniform(1, max_aspect_ratio)
    points = []
    sides = rng.randint(5, 9)
    for i in range(sides):
        # Angular convex fragments; not a claim about any material's lattice.
        angle = i * math.tau / sides + rng.uniform(-0.12, 0.12)
        radius = rng.uniform(0.7, 1.2)
        points.append((radius * math.cos(angle) * math.sqrt(aspect),
                       radius * math.sin(angle) / math.sqrt(aspect)))
    p = hull(points)
    cx, cy = sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p)
    scale = math.sqrt(area / area_um2(p))
    p = tuple(((x - cx) * scale, (y - cy) * scale) for x, y in p)
    return convex_polygon(transform(p, rotation_deg=rng.uniform(0, 360)))


def cleave(*, seed: int, chip_id: str, batch_id: str, material: str, count: int,
           chip_size_mm: Point = (10, 10), profile: CleavageProfile | None = None,
           contact_regions: tuple[Polygon, ...] | None = None) -> tuple[Flake, ...]:
    """Generate once per physical deposition, never per image.

    Count is a requested candidate count, not yield. Empty/no contact returns
    none. A finite rejection budget can return fewer candidates when a region
    cannot contain their full outline. No clipping or edge-clamping pileups.
    """
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 10000:
        raise ValueError("count must be an integer in [0,10000]")
    if not chip_id or not batch_id or not material:
        raise ValueError("chip, batch and material identities are required")
    p = profile or CleavageProfile()
    chip = rectangle(*chip_size_mm)
    regions = (chip,) if contact_regions is None else tuple(
        intersection(convex_polygon(r), chip) for r in contact_regions)
    regions = tuple(r for r in regions if r)
    if not regions:
        return ()
    rng = random.Random(stable_key(MODEL_VERSION, seed, chip_id, batch_id, material))
    weights = [area_um2(r) for r in regions]

    def uniform_point() -> Point:
        region = rng.choices(regions, weights=weights)[0]
        # Sample a triangulated convex polygon uniformly by triangle area.
        triangles = [(region[0], region[i], region[i + 1]) for i in range(1, len(region) - 1)]
        a, b, c = rng.choices(triangles, weights=[area_um2(t) for t in triangles])[0]
        u, v = math.sqrt(rng.random()), rng.random()
        return ((1 - u) * a[0] + u * (1 - v) * b[0] + u * v * c[0],
                (1 - u) * a[1] + u * (1 - v) * b[1] + u * v * c[1])

    centers = [uniform_point() for _ in range(p.cluster_count)]
    result = []
    for i in range(count):
        layers = rng.choices(p.layer_values, weights=p.layer_weights)[0]
        area = rng.lognormvariate(math.log(p.median_area_um2), p.log_area_sigma)
        area *= p.thick_area_multiplier if layers >= 10 else 1
        area = min(p.max_area_um2, max(p.min_area_um2, area))
        fid = "synthetic_flake_" + stable_key(seed, chip_id, batch_id, material, i)
        outline = flake_outline(fid, area, p.max_aspect_ratio)
        for _ in range(80):
            if rng.random() < p.cluster_fraction:
                cx, cy = rng.choice(centers)
                xy = (rng.gauss(cx, p.cluster_radius_mm), rng.gauss(cy, p.cluster_radius_mm))
            else:
                xy = uniform_point()
            placed = transform(outline, xy=xy)
            if any(abs(area_um2(intersection(placed, r)) - area) <= area * 1e-8 for r in regions):
                result.append(Flake(id=fid, material=material, layers=layers, source_chip_id=chip_id,
                                    batch_id=batch_id, outline_mm=placed))
                break
    return tuple(result)


class Placement(FrozenModel):
    flake: Flake  # original source geometry retained in full
    target_center_mm: Point
    rotation_deg: float = 0

    @property
    def outline_mm(self) -> Polygon:
        cx, cy = self.flake.center_mm
        centered = tuple((x - cx, y - cy) for x, y in self.flake.outline_mm)
        return transform(centered, xy=self.target_center_mm, rotation_deg=self.rotation_deg)


class PairAssembly(FrozenModel):
    id: str = Field(min_length=1)
    target_chip_id: str = Field(min_length=1)
    chip_size_mm: Point = (10, 10)
    inputs: tuple[Placement, Placement]  # exactly two, ordered bottom then top

    @model_validator(mode="after")
    def coherent(self) -> PairAssembly:
        if self.inputs[0].flake.id == self.inputs[1].flake.id:
            raise ValueError("a reaction requires two distinct flake identities")
        chip = rectangle(*self.chip_size_mm)
        for placed in self.inputs:
            area = placed.flake.area_um2
            if abs(area_um2(intersection(placed.outline_mm, chip)) - area) > area * 1e-8:
                raise ValueError("the entire transferred flake must fit on the target chip")
        return self

    @property
    def overlap_mm(self) -> Polygon:
        return intersection(*(p.outline_mm for p in self.inputs))


def project_outline(outline: Polygon, *, center_mm: Point, fov_mm: Point,
                    image_size: tuple[int, int] = (640, 420)) -> dict:
    """Projection of fixed geometry; full and in-field areas stay distinct."""
    viewport = rectangle(*fov_mm, center=center_mm)
    if min(image_size) <= 0:
        raise ValueError("image dimensions must be positive")
    clipped = intersection(outline, viewport)
    fw, fh = fov_mm
    width, height = image_size
    pixels = tuple(((x - center_mm[0]) / fw * width + width / 2,
                    height / 2 - (y - center_mm[1]) / fh * height) for x, y in outline)
    fully_visible = bool(clipped) and math.isclose(area_um2(clipped), area_um2(outline), rel_tol=1e-8)
    return {"outline_px": pixels, "visible": bool(clipped), "fully_visible": fully_visible,
            "area_um2": area_um2(outline),
            "visible_area_um2": area_um2(clipped)}
