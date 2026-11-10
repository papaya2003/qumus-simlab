"""Immutable eight-contact hBN/Gr design binding, separate from legacy rectangles.

Legacy convex sources retain their original schema. New region sources preserve
every exterior, hole and disconnected component with an explicit boundary budget.
Imported metal primitives remain exact original parts. A design describes whole
flakes, never an etched shape; geometry does not certify electrical contact.
"""
from __future__ import annotations

import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from qumus_simlab.core.materials import normalize_material_code
from qumus_simlab.core.polygons import (
    area_um2,
    containment_clearance,
    contains,
    convex_polygon,
    erode_convex,
    intersection,
    rectangle,
    transform,
)
from qumus_simlab.core.regions import (
    PolygonRegion,
    as_region,
    quantize_region,
    region_area_um2,
    region_clearance,
    region_covers,
    region_covers_with_clearance,
    region_intersection,
    region_vertices,
    transform_region,
)
from qumus_simlab.core.run_paths import resolve_run_path
from qumus_simlab.records.geometry import (
    _pickup_outline,
    _pickup_region,
    geometry_digest,
    selection_source_snapshot,
)

CONTACT_ROLES = frozenset({"source", "drain", "voltage_left_1", "voltage_left_2", "voltage_left_3",
                           "voltage_right_1", "voltage_right_2", "voltage_right_3"})
SHA = r"^[a-f0-9]{64}$"


class DesignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class NamedElectrode(DesignModel):
    id: str = Field(min_length=1, max_length=100)
    role: Literal["source", "drain", "voltage_left_1", "voltage_left_2", "voltage_left_3",
                  "voltage_right_1", "voltage_right_2", "voltage_right_3"]
    points_mm: tuple[tuple[float, float], ...]

    @field_validator("points_mm")
    @classmethod
    def _polygon(cls, value):
        return convex_polygon(value)


class HallBarTarget(DesignModel):
    version: Literal["hallbar-target-v1"] = "hallbar-target-v1"
    source: Literal["simulated_design", "design", "calibrated_image"]
    reference: str = Field(min_length=1, max_length=400)
    units: Literal["mm"] = "mm"
    frame: Literal["chip_bottom_left_axes"] = "chip_bottom_left_axes"
    size_mm: tuple[float, float]
    contacts: tuple[NamedElectrode, ...] = Field(min_length=8, max_length=8)
    receiving_layers_top_first: tuple[()] = ()

    @model_validator(mode="after")
    def _target(self):
        import math

        if any(not math.isfinite(v) or v <= 0 for v in self.size_mm):
            raise ValueError("target chip dimensions must be finite and positive")
        if len({c.id for c in self.contacts}) != 8 or {c.role for c in self.contacts} != CONTACT_ROLES:
            raise ValueError("eight distinct electrode IDs and all eight contact roles are required")
        centers = {c.role: (sum(x for x, _ in c.points_mm) / len(c.points_mm),
                            sum(y for _, y in c.points_mm) / len(c.points_mm)) for c in self.contacts}
        sx, sy = centers["source"]
        dx, dy = centers["drain"]
        axis_x = (sx + dx) / 2
        if sy <= dy or any(not dy < y < sy or (x >= axis_x if "left" in role else x <= axis_x)
                          for role, (x, y) in centers.items() if role.startswith("voltage_")):
            raise ValueError("source/drain must be top/bottom with three voltage contacts on each side between them")
        for i, contact in enumerate(self.contacts):
            if any(not (0 <= x <= self.size_mm[0] and 0 <= y <= self.size_mm[1])
                   for x, y in contact.points_mm):
                raise ValueError("electrodes must lie within the declared target chip")
            if any(intersection(contact.points_mm, c.points_mm) for c in self.contacts[:i]):
                raise ValueError("distinct electrodes must not overlap")
        return self


FULL_CONTACT_POLICY = "complete_contact_region_containment"
LEGACY_CONTACT_POLICY = "intersection_area_um2 >= contact_overlap_um ** 2"
# Owner-approved R019 source indices, independent of any reference/selected flake.
_IMPORTED_ROLES = (("source", 33), ("drain", 32), ("voltage_left_3", 22),
    ("voltage_left_2", 21), ("voltage_left_1", 23), ("voltage_right_3", 19),
    ("voltage_right_2", 18), ("voltage_right_1", 20))


def _canonical_polygon(points):
    polygon = convex_polygon(points)
    start = min(range(len(polygon)), key=polygon.__getitem__)
    return polygon[start:]+polygon[:start]


class GDSMetalPrimitive(DesignModel):
    """Original integer primitive semantics AND its exact converted polygon."""
    index: StrictInt = Field(ge=0, le=33)
    kind: Literal["polygon", "path", "box"]
    outline_dbu: tuple[tuple[StrictInt, StrictInt], ...]
    points_dbu: tuple[tuple[StrictInt, StrictInt], ...] = ()
    width_dbu: StrictInt | None = None
    bgn_ext: StrictInt | None = None
    end_ext: StrictInt | None = None
    box_dbu: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None = None

    @model_validator(mode="after")
    def _shape(self):
        _canonical_polygon([(x/1e6, y/1e6) for x, y in self.outline_dbu])
        if self.kind == "path":
            if (len(self.points_dbu) != 2 or self.points_dbu[0] == self.points_dbu[1]
                    or type(self.width_dbu) is not int or self.width_dbu <= 0
                    or self.bgn_ext != 0 or self.end_ext != 0 or self.box_dbu is not None):
                raise ValueError("R019 supports original straight square-ended integer paths only")
        elif self.kind == "box":
            if self.box_dbu is None or self.points_dbu or any(v is not None for v in
                    (self.width_dbu, self.bgn_ext, self.end_ext)):
                raise ValueError("Original terminal box semantics are incomplete")
            x0, y0, x1, y1 = self.box_dbu
            if x0 >= x1 or y0 >= y1 or set(self.outline_dbu) != {(x0,y0),(x1,y0),(x1,y1),(x0,y1)}:
                raise ValueError("Converted box differs from its original integer bounds")
        elif (not self.points_dbu or set(self.points_dbu) != set(self.outline_dbu)
                or self.box_dbu is not None or any(v is not None for v in
                    (self.width_dbu, self.bgn_ext, self.end_ext))):
            raise ValueError("Converted polygon differs from its original integer vertices")
        return self

    def points_mm(self):
        return _canonical_polygon([(x/1e6, y/1e6) for x, y in self.outline_dbu])


class ImportedContactRegion(NamedElectrode):
    source_primitive_index: StrictInt = Field(ge=0, le=33)
    definition: Literal["complete_drawn_terminal_box", "first_2um_of_original_1um_inner_path"]

    @field_validator("points_mm")
    @classmethod
    def _canonical(cls, value):
        return _canonical_polygon(value)


class TargetGDSImportReceipt(DesignModel):
    version: Literal["target-gds-import-v1"] = "target-gds-import-v1"
    source_sha256: str = Field(pattern=SHA)
    source_cell: Literal["QUMUS_DEVICE"] = "QUMUS_DEVICE"
    source_layer: tuple[Literal[3], Literal[0]] = (3, 0)
    dbu_um: Literal[0.001] = 0.001
    selection_rule: Literal["r019-original-boxes-and-first-2um-paths-v1"] = "r019-original-boxes-and-first-2um-paths-v1"
    excluded_layers: tuple[StrictInt, ...] = (1, 2, 101, 102, 103, 104, 105, 106, 107)
    owner_instruction_id: str = Field(min_length=1, max_length=400)
    owner_instruction_sha256: str = Field(pattern=SHA)
    geometry_sha256: str = Field(pattern=SHA)

    @model_validator(mode="after")
    def _excluded(self):
        if self.excluded_layers != (1, 2, 101, 102, 103, 104, 105, 106, 107):
            raise ValueError("Reference flake/obsolete contact layers cannot become imported metal")
        return self


def _derived_import_contacts(primitives):
    """Recompute fixed ends from ORIGINAL path centreline, never a supplied tip."""
    by_index = {part.index: part for part in primitives}
    contacts = []
    for role, index in _IMPORTED_ROLES:
        part = by_index[index]
        polygon = [[float(x), float(y)] for x, y in part.outline_dbu]
        if role in ("source", "drain"):
            if part.kind != "box":
                raise ValueError("Owner-defined source/drain must be entire original terminal boxes")
            definition = "complete_drawn_terminal_box"
        else:
            if part.kind != "path" or part.width_dbu != 1000:
                raise ValueError("Owner-defined voltage region must retain its original 1um path width")
            a, b = part.points_dbu
            length = math.dist(a, b)
            if length < 2000:
                raise ValueError("Original voltage path is shorter than the fixed 2um region")
            direction = [(b[j]-a[j])/length for j in (0, 1)]
            # Clip only the contact REGION at exactly 2um along the original
            # first endpoint. The full metal primitive itself is never cropped.
            clipped = []
            for first, second in zip(polygon, polygon[1:]+polygon[:1], strict=True):
                da = sum((first[j]-a[j])*direction[j] for j in (0, 1))-2000
                db = sum((second[j]-a[j])*direction[j] for j in (0, 1))-2000
                if da <= 0:
                    clipped.append(first)
                if (da <= 0) != (db <= 0):
                    ratio = da/(da-db)
                    clipped.append([first[j]+ratio*(second[j]-first[j]) for j in (0, 1)])
            polygon = clipped
            definition = "first_2um_of_original_1um_inner_path"
        region = ImportedContactRegion(id=role, role=role, source_primitive_index=index,
            definition=definition, points_mm=[(x/1e6, y/1e6) for x, y in polygon])
        if not contains(part.points_mm(), region.points_mm):
            raise ValueError("Derived contact region leaves its exact original metal primitive")
        contacts.append(region)
    return tuple(contacts)


def _import_geometry(primitives, contacts, size_mm):
    return {"size_mm": list(size_mm), "metal_primitives": [p.model_dump(mode="json") for p in primitives],
            "contacts": [c.model_dump(mode="json") for c in contacts]}


class ImportedHallBarTarget(HallBarTarget):
    """Additive target variant; old target defaults and dumps are untouched."""
    version: Literal["hallbar-target-gds-v2"] = "hallbar-target-gds-v2"
    source: Literal["design"] = "design"
    contacts: tuple[ImportedContactRegion, ...] = Field(min_length=8, max_length=8)
    metal_primitives: tuple[GDSMetalPrimitive, ...] = Field(min_length=34, max_length=34)
    import_receipt: TargetGDSImportReceipt
    coverage_policy: Literal["complete_contact_region_containment"] = FULL_CONTACT_POLICY

    @model_validator(mode="after")
    def _imported(self):
        if tuple(p.index for p in self.metal_primitives) != tuple(range(34)):
            raise ValueError("All 34 original metal primitives must retain their original indices")
        contacts = _derived_import_contacts(self.metal_primitives)
        if self.contacts != contacts:
            raise ValueError("Contact ends/roles differ from the fixed original-primitive selection")
        if any(not (0 <= x <= self.size_mm[0] and 0 <= y <= self.size_mm[1])
               for part in self.metal_primitives for x, y in part.points_mm()):
            raise ValueError("The complete imported metal must fit its declared chip")
        receipt = self.import_receipt
        if (receipt.geometry_sha256 != geometry_digest(_import_geometry(self.metal_primitives, contacts, self.size_mm))
                or self.reference != "saved-gds-import:"+geometry_digest(receipt.model_dump(mode="json"))):
            raise ValueError("Imported target geometry/provenance receipt changed")
        return self


def parse_hallbar_target(value):
    if isinstance(value, (HallBarTarget, ImportedHallBarTarget)):
        value = value.model_dump(mode="json")
    if isinstance(value, dict) and value.get("version") == "hallbar-target-gds-v2":
        return ImportedHallBarTarget.model_validate(value)
    return HallBarTarget.model_validate(value)


def design_anchor_mm(target) -> tuple[float, float]:
    """Chip-local (bottom-left axes) point the stack is aimed at: the centre of
    the eight contact ends, on a 1 um grid. It is the chip centre for a pattern
    in the middle of its chip (the 10 x 10 design windows) and the electrode
    site for a pattern off the centre (#206: v1's 17.3 x 26.5 mm electrode chip
    with the GDS metal 5 mm from its bottom-left corner; v1 likewise goes to
    the electrode's recorded offset from the chip's bottom-left corner)."""
    target = parse_hallbar_target(target)
    xs = [x for contact in target.contacts for x, _ in contact.points_mm]
    ys = [y for contact in target.contacts for _, y in contact.points_mm]
    return (round((min(xs)+max(xs))/2, 3), round((min(ys)+max(ys))/2, 3))


def target_metal_primitives(target):
    target = parse_hallbar_target(target)
    if isinstance(target, ImportedHallBarTarget):
        return [{"id": f"metal_{p.index:02d}", "source_primitive_index": p.index,
                 "points_mm": p.points_mm()} for p in target.metal_primitives]
    return [{"id": c.id, "points_mm": c.points_mm} for c in target.contacts]


def target_import_binding(target):
    target = parse_hallbar_target(target)
    if not isinstance(target, ImportedHallBarTarget):
        return None
    receipt = target.import_receipt
    return {"source_sha256": receipt.source_sha256, "target_sha256": geometry_digest(target.model_dump(mode="json")),
        "source": "simulation_config.initial_samples", "target_version": target.version,
        "owner_instruction_id": receipt.owner_instruction_id, "owner_instruction_sha256": receipt.owner_instruction_sha256,
        "import_receipt_sha256": geometry_digest(receipt.model_dump(mode="json"))}


def target_contact_coverage(target, gr_outline, *, minimum_overlap_um=1.):
    target = parse_hallbar_target(target)
    general = isinstance(gr_outline, (PolygonRegion, dict))
    polygon = as_region(gr_outline) if general else convex_polygon(gr_outline)
    strict = isinstance(target, ImportedHallBarTarget)
    if type(minimum_overlap_um) not in (int, float) or not math.isfinite(minimum_overlap_um) or minimum_overlap_um <= 0:
        raise ValueError("Contact overlap parameter must be finite and positive")
    contacts, failed = {}, []
    for contact in target.contacts:
        area = (region_area_um2(region_intersection(polygon, contact.points_mm)) if general
                else area_um2(intersection(polygon, contact.points_mm)))
        contained = region_covers(polygon, contact.points_mm) if general else contains(polygon, contact.points_mm)
        required = area_um2(contact.points_mm) if strict else minimum_overlap_um**2
        good = contained if strict else area >= required
        contacts[contact.id] = {"contained": contained, "intersection_area_um2": area,
            "required_area_um2": required, "clearance_mm": region_clearance(polygon, contact.points_mm) if general
            else containment_clearance(polygon, contact.points_mm)}
        if not good:
            failed.append(contact.id)
    return {"status": "fail" if failed else "pass", "policy": FULL_CONTACT_POLICY if strict else LEGACY_CONTACT_POLICY,
            "failed_contacts": failed, "contacts": contacts}


class FlakePlacement(DesignModel):
    """Rigid map: flake-centred chip axes -> target bottom-left axes, in mm."""
    flake_id: str = Field(min_length=1)
    translation_mm: tuple[float, float]
    rotation_deg: float = Field(allow_inf_nan=False)

    @field_validator("translation_mm")
    @classmethod
    def _translation(cls, value):
        import math

        if not all(math.isfinite(v) for v in value):
            raise ValueError("translation must be finite")
        return value


class DesignSource(DesignModel):
    flake_id: str
    material: Literal["hBN", "Gr"]
    source_chip_id: str
    outline_mm: tuple[tuple[float, float], ...]
    observation: dict[str, Any]
    evidence_event_ids: tuple[int, ...] = Field(min_length=1)
    image_sha256: str = Field(pattern=SHA)
    placement: FlakePlacement

    @field_validator("outline_mm")
    @classmethod
    def _outline(cls, value):
        return convex_polygon(value)

    @model_validator(mode="after")
    def _source(self):
        if self.placement.flake_id != self.flake_id:
            raise ValueError("placement identity differs from its source")
        return self

    def placed_outline(self):
        return transform(self.outline_mm, xy=self.placement.translation_mm,
                         rotation_deg=self.placement.rotation_deg)

    def placed_region(self):
        return as_region(self.placed_outline())


class RegionDesignSource(DesignModel):
    """Additive source shape; historical convex source dumps stay unchanged."""
    flake_id: str
    material: Literal["hBN", "Gr"]
    source_chip_id: str
    region_mm: PolygonRegion
    observation: dict[str, Any]
    evidence_event_ids: tuple[int, ...] = Field(min_length=1)
    image_sha256: str = Field(pattern=SHA)
    placement: FlakePlacement

    @model_validator(mode="after")
    def _source(self):
        if self.placement.flake_id != self.flake_id:
            raise ValueError("placement identity differs from its source")
        if (not self.region_mm.components or not isinstance(self.observation.get("region_mm"), dict)
                or as_region(self.observation["region_mm"]) != self.region_mm):
            raise ValueError("Region source differs from its complete original observation")
        source_boundary_uncertainty(self)
        return self

    def placed_region(self):
        return transform_region(self.region_mm, xy=self.placement.translation_mm,
                                rotation_deg=self.placement.rotation_deg)


def source_region(source: DesignSource | RegionDesignSource, *, placed: bool = True) -> PolygonRegion:
    if placed:
        return source.placed_region()
    return as_region(source.region_mm if isinstance(source, RegionDesignSource) else source.outline_mm)


def source_boundary_uncertainty(source: DesignSource | RegionDesignSource) -> float:
    """Shape-comparison allowance; owner-inclusive budgets remain nonzero here."""
    if isinstance(source, DesignSource):
        return 0.  # Existing legacy observations retain their historical contract.
    observation = source.observation
    if (observation.get("source") == "rgb_native100_measured_region"
            or observation.get("source") == "rgb_optical_outline"
            and observation.get("owner_error_budget") is not None):
        from qumus_simlab.perception.measurement import validate_owner_error_budget

        return validate_owner_error_budget(observation)
    if observation.get("source") == "rgb_optical_outline":
        from qumus_simlab.records.recognition import validate_boundary_uncertainty

        return validate_boundary_uncertainty(observation)
    bound = observation.get("boundary_uncertainty_mm")
    if (type(bound) not in (int, float) or not math.isfinite(bound) or bound <= 0
            or not isinstance(observation.get("boundary_uncertainty_provenance"), dict)
            or not observation["boundary_uncertainty_provenance"]):
        raise ValueError("New region source requires explicit positive boundary uncertainty/provenance")
    return float(bound)


def source_additional_boundary_uncertainty(source: DesignSource | RegionDesignSource) -> float:
    """Only the boundary allowance additional to the configured spatial budget.

    The native owner profile includes measurement, registration and positioning
    in its one-micrometre spatial allowance. Zero here means included already,
    never zero physical measurement error or a calibration certificate.
    """
    bound = source_boundary_uncertainty(source)
    if (source.observation.get("source") == "rgb_native100_measured_region"
            or source.observation.get("source") == "rgb_optical_outline"
            and source.observation.get("owner_error_budget") is not None):
        return 0.
    return bound


class DesignPoseTolerance(DesignModel):
    translation_um: float = Field(gt=0, allow_inf_nan=False)
    rotation_deg: float = Field(gt=0, allow_inf_nan=False)
    frame: Literal["chip_bottom_left_axes"] = "chip_bottom_left_axes"
    source: Literal["synthetic_verification_defaults"] = "synthetic_verification_defaults"


class DeviceDesignSpec(DesignModel):
    version: Literal["device_design_v1"] = "device_design_v1"
    design_id: str = Field(pattern=r"^design_[a-f0-9]{32}$")
    project_id: str
    revision: int = Field(ge=1)
    bench: Literal["sim", "real"]
    target_chip_id: str
    target: HallBarTarget | ImportedHallBarTarget
    guidance_sha256: str = Field(pattern=SHA)
    selection_sha256: str = Field(pattern=SHA)
    sources: tuple[DesignSource | RegionDesignSource, DesignSource | RegionDesignSource]
    pickup_order: tuple[str, str]
    final_layers_top_first: tuple[Literal["hBN"], Literal["Gr"], Literal["electrodes"], Literal["silicon"]] = ("hBN", "Gr", "electrodes", "silicon")
    contact_overlap_um: float = Field(gt=0, allow_inf_nan=False)
    pose_tolerance: DesignPoseTolerance
    overlap_rule: Literal["intersection_area_um2 >= contact_overlap_um ** 2", "complete_contact_region_containment"] = LEGACY_CONTACT_POLICY

    @model_validator(mode="after")
    def _geometry(self):
        if self.bench != "sim":
            raise ValueError("this design's pose tolerance is synthetic; a real executable design requires measured calibration")
        if [s.material for s in self.sources] != ["hBN", "Gr"]:
            raise ValueError("exactly hBN then Gr is required; no bottom hBN")
        ids = tuple(s.flake_id for s in self.sources)
        if len(set(ids)) != 2 or self.pickup_order != ids:
            raise ValueError("pickup order must be the selected hBN then selected Gr")
        general = any(isinstance(s, RegionDesignSource) for s in self.sources)
        hbn, gr = (source_region(s) for s in self.sources) if general else (s.placed_outline() for s in self.sources)
        if not (region_covers(hbn, gr) if general else contains(hbn, gr)):
            raise ValueError("hBN must cover the complete Gr outline, not only a channel crop")
        if any(not (0 <= x <= self.target.size_mm[0] and 0 <= y <= self.target.size_mm[1])
               for source in self.sources for x, y in (region_vertices(source_region(source))
                   if general else source.placed_outline())):
            raise ValueError("complete placed flakes must fit within the target chip")
        coverage = target_contact_coverage(self.target, gr, minimum_overlap_um=self.contact_overlap_um)
        if self.overlap_rule != coverage["policy"]:
            raise ValueError("Design contact policy differs from its versioned target")
        if coverage["status"] != "pass":
            if isinstance(self.target, ImportedHallBarTarget):
                raise ValueError("Gr does not contain each complete imported contact end: "+", ".join(coverage["failed_contacts"]))
            raise ValueError(f"Gr does not meet the declared overlap area on contact {coverage['failed_contacts'][0]}")
        return self


class DesignArtifact(DesignModel):
    path: str
    sha256: str = Field(pattern=SHA)
    backend: Literal["klayout_batch"] = "klayout_batch"
    dbu_um: Literal[0.001] = 0.001
    geometry_sha256: str = Field(pattern=SHA)


class KLayoutMCPArtifact(DesignArtifact):
    """New transport binding; the historical batch artifact dumps stay unchanged."""
    backend: Literal["klayout_mcp"] = "klayout_mcp"
    receipt_sha256: str = Field(pattern=SHA)


class KLayoutMCPReceipt(DesignModel):
    version: Literal["klayout_mcp_receipt_v1"] = "klayout_mcp_receipt_v1"
    design_id: str = Field(pattern=r"^design_[a-f0-9]{32}$")
    nonce: str = Field(pattern=r"^[a-f0-9]{32}$")
    endpoint: str
    server_name: Literal["KlayoutClaw"] = "KlayoutClaw"
    server_version: str = Field(min_length=1)
    protocol: Literal["2025-03-26"] = "2025-03-26"
    atomic_profile: Literal["atomic-new-layout-v1"] = "atomic-new-layout-v1"
    geometry_sha256: str = Field(pattern=SHA)
    script_sha256: str = Field(pattern=SHA)
    gds_sha256: str = Field(pattern=SHA)
    image_sha256: str = Field(pattern=SHA)
    view_identity: str = Field(min_length=1)
    view_index: int = Field(ge=0, strict=True)
    dbu_um: Literal[0.001] = 0.001
    top_cell: Literal["QUMUS_DEVICE"] = "QUMUS_DEVICE"
    layer_names: tuple[str, ...]
    zoom_box_um: tuple[float, float, float, float]
    image_size: tuple[Literal[1280], Literal[960]] = (1280, 960)
    status: Literal["verified"] = "verified"

    @model_validator(mode="after")
    def _viewport(self):
        if (not all(math.isfinite(v) for v in self.zoom_box_um)
                or self.zoom_box_um[0] >= self.zoom_box_um[2]
                or self.zoom_box_um[1] >= self.zoom_box_um[3]):
            raise ValueError("native KLayout viewport must be finite and nonempty")
        return self


class KLayoutPythonArtifact(DesignArtifact):
    """Headless klayout.db save/reopen; explicitly selected, never a silent fallback."""
    backend: Literal["klayout_python"] = "klayout_python"
    receipt_sha256: str = Field(pattern=SHA)


class KLayoutPythonReceipt(DesignModel):
    version: Literal["klayout_python_receipt_v1"] = "klayout_python_receipt_v1"
    design_id: str = Field(pattern=r"^design_[a-f0-9]{32}$")
    nonce: str = Field(pattern=r"^[a-f0-9]{32}$")
    klayout_version: str = Field(min_length=1)
    selection: Literal["QUMUS_KLAYOUT_BACKEND=python"] = "QUMUS_KLAYOUT_BACKEND=python"
    geometry_sha256: str = Field(pattern=SHA)
    script_sha256: str = Field(pattern=SHA)
    gds_sha256: str = Field(pattern=SHA)
    image_sha256: str = Field(pattern=SHA)
    dbu_um: Literal[0.001] = 0.001
    top_cell: Literal["QUMUS_DEVICE"] = "QUMUS_DEVICE"
    layer_names: tuple[str, ...]
    zoom_box_um: tuple[float, float, float, float]
    image_size: tuple[Literal[1280], Literal[960]] = (1280, 960)
    status: Literal["verified"] = "verified"
    scope: Literal["Saved and reopened with the klayout Python module using the same generated checks as the MCP script; the preview is a rendering, not a native KLayout view"] = (
        "Saved and reopened with the klayout Python module using the same generated checks as the MCP script; the preview is a rendering, not a native KLayout view")

    @model_validator(mode="after")
    def _viewport(self):
        if (not all(math.isfinite(v) for v in self.zoom_box_um)
                or self.zoom_box_um[0] >= self.zoom_box_um[2]
                or self.zoom_box_um[1] >= self.zoom_box_um[3]):
            raise ValueError("preview viewport must be finite and nonempty")
        return self


class DeviceDesignV1(DeviceDesignSpec):
    # An MCP record cannot fall through to the legacy shape and lose its receipt.
    artifact: DesignArtifact | KLayoutMCPArtifact | KLayoutPythonArtifact
    canonical_digest: str = Field(pattern=SHA)

    @model_validator(mode="after")
    def _digest(self):
        if geometry_digest(portable(self.model_dump(mode="json", exclude={"canonical_digest"}))) != self.canonical_digest:
            raise ValueError("device design canonical digest changed")
        return self


class DesignPoseWindowPolicy(DesignModel):
    version: Literal["device_design_pose_window_policy_v1"] = "device_design_pose_window_policy_v1"
    source: Literal["verification.yaml"] = "verification.yaml"
    verification_sha256: str = Field(pattern=SHA)
    translation_um: float = Field(gt=0, allow_inf_nan=False)
    rotation_deg: float = Field(gt=0, le=180, allow_inf_nan=False)
    shape_residual_mm: Literal[1e-9] = 1e-9
    floating_guard_mm: Literal[1e-12] = 1e-12
    pivot: Literal["arithmetic_vertex_mean"] = "arithmetic_vertex_mean"
    frame: Literal["chip_bottom_left_axes"] = "chip_bottom_left_axes"


class DesignPoseWindowSource(DesignModel):
    flake_id: str
    material: Literal["hBN", "Gr"]
    pivot_mm: tuple[float, float]
    radius_mm: float = Field(ge=0, allow_inf_nan=False)
    error_bound_mm: float = Field(gt=0, allow_inf_nan=False)
    chip_clearance_mm: float = Field(allow_inf_nan=False)
    chip_headroom_mm: float = Field(allow_inf_nan=False)


class RegionPoseWindowSource(DesignPoseWindowSource):
    geometry: Literal["polygon-region-v1"] = "polygon-region-v1"
    pose_error_bound_mm: float = Field(gt=0, allow_inf_nan=False)
    boundary_uncertainty_mm: float = Field(ge=0, allow_inf_nan=False,
        description="Additional boundary allowance; zero only when the owner profile includes it in translation")
    boundary_uncertainty_provenance: dict[str, Any]


class DesignPoseWindowContact(DesignModel):
    contact_id: str
    nominal_area_um2: float = Field(ge=0, allow_inf_nan=False)
    certified_area_um2: float = Field(ge=0, allow_inf_nan=False)
    required_area_um2: float = Field(gt=0, allow_inf_nan=False)
    headroom_um2: float = Field(allow_inf_nan=False)


class DesignPoseWindowReport(DesignModel):
    version: Literal["device_design_pose_window_report_v1"] = "device_design_pose_window_report_v1"
    status: Literal["pass", "fail"]
    spec_sha256: str = Field(pattern=SHA)
    policy: DesignPoseWindowPolicy
    sources: tuple[RegionPoseWindowSource | DesignPoseWindowSource,
                   RegionPoseWindowSource | DesignPoseWindowSource]
    hbn_clearance_mm: float = Field(allow_inf_nan=False)
    hbn_required_clearance_mm: float = Field(gt=0, allow_inf_nan=False)
    hbn_headroom_mm: float = Field(allow_inf_nan=False)
    contacts: tuple[DesignPoseWindowContact, ...] = Field(min_length=8, max_length=8)
    failed_contacts: tuple[str, ...]
    failures: tuple[str, ...]
    scope: Literal["Declared endpoint pose-window geometry only; no physical registration, noise-confidence or optical certificate"] = (
        "Declared endpoint pose-window geometry only; no physical registration, noise-confidence or optical certificate")


class DesignPoseWindowAdmission(DesignModel):
    """Fresh-job sidecar; never part of the historical canonical design schema."""
    version: Literal["device_design_pose_window_admission_v1"] = "device_design_pose_window_admission_v1"
    canonical_design_digest: str = Field(pattern=SHA)
    report_sha256: str = Field(pattern=SHA)
    report: DesignPoseWindowReport

    @model_validator(mode="after")
    def _report_digest(self):
        if geometry_digest(self.report.model_dump(mode="json")) != self.report_sha256:
            raise ValueError("pose-window report digest changed")
        return self


class DesignPoseWindowError(ValueError):
    def __init__(self, report: DesignPoseWindowReport):
        self.report = report
        super().__init__("Design does not cover its declared endpoint pose window: " + ", ".join(report.failures))


def design_rotation_window_deg(verification: dict) -> float:
    """Per-flake rotation the design must survive (#204: v1 never turns a chip)."""
    window = verification.get("design_pose_window") or {}
    return window.get("rotation_deg", verification["vision_control"]["residual_rotation_deg"])


def design_pose_window_policy(verification: dict) -> DesignPoseWindowPolicy:
    """Pure public projection of the same trusted configuration used by screening."""
    try:
        translation = verification["refind_flake"]["centre_tolerance_um"]
        rotation = design_rotation_window_deg(verification)
        if any(type(value) not in (int, float) for value in (translation, rotation)):
            raise ValueError("pose-window limits must be numeric, not boolean")
        return DesignPoseWindowPolicy(verification_sha256=geometry_digest(verification),
            translation_um=translation, rotation_deg=rotation)
    except (KeyError, TypeError) as exc:
        raise ValueError("pose-window verification configuration is incomplete") from exc


def screen_design_pose_window(design: DeviceDesignSpec | dict, *, verification: dict | None = None) -> DesignPoseWindowReport:
    """Sufficient fresh screening for every pose accepted by the existing comparator.

    Historical model parsing and artifact validation intentionally do not call
    this policy. Each displacement ball uses the comparator's vertex-mean pivot
    and its unchanged nonzero shape-residual cap.
    """
    from qumus_simlab.records.geometry import pickup_coverage

    raw = design.model_dump(mode="json") if isinstance(design, DeviceDesignSpec) else design
    parsed = DeviceDesignV1.model_validate(raw) if "canonical_digest" in raw else DeviceDesignSpec.model_validate(raw)
    if verification is None:
        from qumus_simlab.config import load_yaml

        verification = load_yaml("verification.yaml")
    policy = design_pose_window_policy(verification)
    if (parsed.pose_tolerance.translation_um != policy.translation_um
            or parsed.pose_tolerance.rotation_deg != policy.rotation_deg):
        raise ValueError("frozen design pose window differs from the current verification configuration")
    if any(isinstance(source, RegionDesignSource) for source in parsed.sources):
        return _screen_region_pose_window(parsed, policy)
    spec_sha = geometry_digest(portable(parsed.model_dump(mode="json", exclude={"artifact", "canonical_digest"})))
    placed = [source.placed_outline() for source in parsed.sources]
    width, height = parsed.target.size_mm
    chip = rectangle(width, height, (width/2, height/2))
    sources = []
    for source, polygon in zip(parsed.sources, placed, strict=True):
        pivot = tuple(sum(point[axis] for point in polygon)/len(polygon) for axis in (0, 1))
        radius = max(math.dist(point, pivot) for point in polygon)
        error = (policy.translation_um/1000 + 2*radius*math.sin(math.radians(policy.rotation_deg)/2)
                 + policy.shape_residual_mm + policy.floating_guard_mm)
        clearance = containment_clearance(chip, polygon)
        sources.append(DesignPoseWindowSource(flake_id=source.flake_id, material=source.material,
            pivot_mm=pivot, radius_mm=radius, error_bound_mm=error,
            chip_clearance_mm=clearance, chip_headroom_mm=clearance-error))
    binding = {"spec_sha256": spec_sha, "policy_sha256": geometry_digest(policy.model_dump(mode="json"))}
    required = sum(source.error_bound_mm for source in sources)
    coverage = pickup_coverage(*placed, frame=policy.frame, binding=binding,
        required_bound={"frame": policy.frame, "binding": binding, "required_clearance_mm": required,
                        "provenance": policy.version})
    if coverage["state"] not in {"bound_satisfied", "bound_failed"}:
        raise ValueError("pose-window containment geometry is unresolved")
    core = erode_convex(placed[1], sources[1].error_bound_mm)
    contacts = []
    strict = isinstance(parsed.target, ImportedHallBarTarget)
    containment_failures = set()
    for contact in parsed.target.contacts:
        area = area_um2(intersection(core, contact.points_mm)) if core else 0.
        minimum = area_um2(contact.points_mm) if strict else parsed.contact_overlap_um**2
        if strict and not contains(core, contact.points_mm):
            containment_failures.add(contact.id)
        contacts.append(DesignPoseWindowContact(contact_id=contact.id,
            nominal_area_um2=area_um2(intersection(placed[1], contact.points_mm)),
            certified_area_um2=area, required_area_um2=minimum, headroom_um2=area-minimum))
    failed_contacts = tuple(contact.contact_id for contact in contacts
        if contact.contact_id in containment_failures or not strict and contact.headroom_um2 < 0)
    failures = (["whole_gr_coverage"] if coverage["state"] != "bound_satisfied" else [])
    failures += ["chip_fit:"+source.flake_id for source in sources if source.chip_headroom_mm < 0]
    failures += [("contact_containment:" if strict else "contact_overlap:")+contact for contact in failed_contacts]
    return DesignPoseWindowReport(status="fail" if failures else "pass", spec_sha256=spec_sha, policy=policy,
        sources=tuple(sources), hbn_clearance_mm=coverage["nominal_clearance_mm"],
        hbn_required_clearance_mm=required, hbn_headroom_mm=coverage["headroom_mm"],
        contacts=tuple(contacts), failed_contacts=failed_contacts, failures=tuple(failures))


def _region_pose_displacement(region, policy):
    vertices = region_vertices(region)
    pivot = tuple(sum(point[axis] for point in vertices)/len(vertices) for axis in (0, 1))
    radius = max(math.dist(point, pivot) for point in vertices)
    pose_error = (policy.translation_um/1000 + 2*radius*math.sin(math.radians(policy.rotation_deg)/2)
                  + getattr(policy, "shape_residual_mm", 1e-9)
                  + getattr(policy, "floating_guard_mm", 1e-12))
    return pivot, radius, pose_error


def _region_source_errors(source, region, policy):
    pivot, radius, pose_error = _region_pose_displacement(region, policy)
    boundary = source_additional_boundary_uncertainty(source)
    if (source.observation.get("source") == "rgb_native100_measured_region"
            or source.observation.get("source") == "rgb_optical_outline"
            and source.observation.get("owner_error_budget") is not None):
        owner = source.observation["owner_error_budget"]
        if (policy.translation_um/1000 < source_boundary_uncertainty(source)
                or policy.rotation_deg < owner["rotation_tolerance_deg"]):
            raise ValueError("Configured pose allowance is smaller than the inclusive owner budget")
    return pivot, radius, pose_error, boundary, pose_error+boundary


def _screen_region_pose_window(parsed, policy):
    """Covers + exact boundary distance, with no polygonized circle certificate."""
    placed = [source_region(source) for source in parsed.sources]
    width, height = parsed.target.size_mm
    chip = as_region(rectangle(width, height, (width/2, height/2)))
    sources = []
    for source, region in zip(parsed.sources, placed, strict=True):
        pivot, radius, pose_error, boundary, error = _region_source_errors(source, region, policy)
        clearance = region_clearance(chip, region)
        if clearance is None:
            raise ValueError("Nominal region source leaves the target chip")
        values = dict(flake_id=source.flake_id, material=source.material, pivot_mm=pivot,
            radius_mm=radius, error_bound_mm=error, chip_clearance_mm=clearance,
            chip_headroom_mm=clearance-error)
        if isinstance(source, RegionDesignSource):
            sources.append(RegionPoseWindowSource(**values, pose_error_bound_mm=pose_error,
                boundary_uncertainty_mm=boundary,
                boundary_uncertainty_provenance=source.observation["boundary_uncertainty_provenance"]))
        else:
            sources.append(DesignPoseWindowSource(**values))
    required = sum(source.error_bound_mm for source in sources)
    clearance = region_clearance(*placed)
    if clearance is None:
        raise ValueError("Nominal hBN region does not contain whole Gr")
    contacts = []
    failed = []
    for contact in parsed.target.contacts:
        nominal = region_area_um2(region_intersection(placed[1], contact.points_mm))
        good = region_covers_with_clearance(placed[1], contact.points_mm, sources[1].error_bound_mm)
        minimum = (area_um2(contact.points_mm) if isinstance(parsed.target, ImportedHallBarTarget)
                   else parsed.contact_overlap_um**2)
        # Full-region coverage is a conservative area certificate. A failing
        # margin has no certified positive area; no approximate erosion is used.
        certified = area_um2(contact.points_mm) if good else 0.
        contacts.append(DesignPoseWindowContact(contact_id=contact.id, nominal_area_um2=nominal,
            certified_area_um2=certified, required_area_um2=minimum, headroom_um2=certified-minimum))
        if not good:
            failed.append(contact.id)
    failures = ["whole_gr_coverage"] if clearance < required else []
    failures += ["chip_fit:"+source.flake_id for source in sources if source.chip_headroom_mm < 0]
    failures += ["contact_containment:"+name for name in failed]
    return DesignPoseWindowReport(status="fail" if failures else "pass",
        spec_sha256=geometry_digest(portable(parsed.model_dump(mode="json", exclude={"artifact", "canonical_digest"}))),
        policy=policy, sources=tuple(sources), hbn_clearance_mm=clearance,
        hbn_required_clearance_mm=required, hbn_headroom_mm=clearance-required,
        contacts=tuple(contacts), failed_contacts=tuple(failed), failures=tuple(failures))


def require_design_pose_window(design: DeviceDesignSpec | dict, *, verification: dict | None = None) -> DesignPoseWindowReport:
    report = screen_design_pose_window(design, verification=verification)
    if report.status != "pass":
        raise DesignPoseWindowError(report)
    return report


def _diagnostic_region_margins(target, regions, displacement_mm):
    """Exact containment distances; an outside body has no invented clearance."""
    chip = as_region(rectangle(*target.size_mm, (target.size_mm[0]/2, target.size_mm[1]/2)))
    constraints = []
    for index, region in enumerate(regions):
        clearance = region_clearance(chip, region)
        constraints.append({"id": "chip_fit:"+("hBN" if index == 0 else "Gr"),
            "clearance_um": None if clearance is None else clearance*1000,
            "required_clearance_um": displacement_mm[index]*1000})
    clearance = region_clearance(regions[0], regions[1])
    constraints.append({"id": "whole_gr_coverage", "clearance_um": None if clearance is None else clearance*1000,
                        "required_clearance_um": sum(displacement_mm)*1000})
    for contact in target.contacts:
        clearance = region_clearance(regions[1], contact.points_mm)
        constraints.append({"id": "contact_containment:"+contact.id,
            "clearance_um": None if clearance is None else clearance*1000,
            "required_clearance_um": displacement_mm[1]*1000,
            "intersection_area_um2": region_area_um2(region_intersection(regions[1], contact.points_mm)),
            "required_area_um2": area_um2(contact.points_mm), "source_primitive_index": contact.source_primitive_index})
    for item in constraints:
        clearance = item["clearance_um"]
        item["contained"] = clearance is not None
        item["headroom_um"] = None if clearance is None else clearance-item["required_clearance_um"]
        item["status"] = "pass" if item["headroom_um"] is not None and item["headroom_um"] >= 0 else "fail"
    return {"status": "pass" if all(c["status"] == "pass" for c in constraints) else "fail",
        "minimum_headroom_um": min(c["headroom_um"] for c in constraints)
            if all(c["contained"] for c in constraints) else None,
        "failed_constraints": [c["id"] for c in constraints if c["status"] != "pass"], "constraints": constraints}


def diagnose_measured_region_layout(target, sources: list[dict], *, verification: dict | None = None) -> dict:
    """Inspect explicit region placements without creating an executable design.

    Each source declares its coordinate frame and placement reference. These
    declarations are recorded, not promoted to source/registration evidence.
    Unknown observation error is never silently replaced by zero: the separate
    pose-only calculation deliberately excludes it and cannot grant admission.
    """
    target = parse_hallbar_target(target)
    if not isinstance(target, ImportedHallBarTarget):
        raise ValueError("Measured-region diagnostics require the full imported eight-end target")
    if len(sources) != 2 or [s.get("material") for s in sources] != ["hBN", "Gr"]:
        raise ValueError("Diagnostic sources must be exactly the selected hBN then Gr")
    inputs, placed, pose = [], [], []
    if verification is None:
        from qumus_simlab.config import load_yaml

        verification = load_yaml("verification.yaml")
    policy = design_pose_window_policy(verification)
    for source in sources:
        for key in ("source_frame", "placement_reference"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError("Diagnostic source needs an explicit "+key)
        placement = FlakePlacement.model_validate(source["placement"])
        if placement.flake_id != source.get("flake_id"):
            raise ValueError("Diagnostic placement and source identity differ")
        region = as_region(source["region_mm"])
        if not region.components:
            raise ValueError("Diagnostic measured region cannot be empty")
        record = {"flake_id": source["flake_id"], "material": source["material"],
            "source_frame": source["source_frame"], "placement_reference": source["placement_reference"],
            "region_mm": region.model_dump(mode="json"), "placement": placement.model_dump(mode="json"),
            "evidence": source.get("evidence", {})}
        inputs.append(record)
        body = transform_region(region, xy=placement.translation_mm, rotation_deg=placement.rotation_deg)
        placed.append(body)
        pivot, radius, error = _region_pose_displacement(body, policy)
        pose.append({"flake_id": source["flake_id"], "pivot_mm": list(pivot), "radius_mm": radius,
            "pose_displacement_mm": error, "measurement_bound_mm": None, "registration_bound_mm": None})
    if inputs[0]["flake_id"] == inputs[1]["flake_id"]:
        raise ValueError("Diagnostic sources must have distinct identities")
    rounded = [quantize_region(region) for region in placed]
    errors = [source["pose_displacement_mm"] for source in pose]
    layers = []
    for index, (source, (region, rounding)) in enumerate(zip(inputs, rounded, strict=True), 1):
        layers.append({"name": source["flake_id"], "layer": index,
            "region_components_dbu": [{"exterior": [[round(x*1e6), round(y*1e6)] for x, y in part.exterior],
                "holes": [[[round(x*1e6), round(y*1e6)] for x, y in ring] for ring in part.holes]}
                for part in region.components], "quantization": rounding})
    layers.append({"name": "full_imported_electrode_metal", "layer": 3,
        "polygons_dbu": [[list(point) for point in part.outline_dbu] for part in target.metal_primitives]})
    focus = [p for region in placed for p in region_vertices(region)]+[p for c in target.contacts for p in c.points_mm]
    xs, ys = ([p[axis]*1000 for p in focus] for axis in (0, 1))
    margin = max(max(xs)-min(xs), max(ys)-min(ys))*.12+1
    payload = {"version": "qumus-measured-region-diagnostic-v1", "diagnostic_only": True,
        "admission_status": "blocked", "dbu_um": .001, "top_cell": "QUMUS_DIAGNOSTIC_ONLY",
        "layers": layers, "source_target": {"import_receipt": target.import_receipt.model_dump(mode="json"),
            "primitives": [p.model_dump(mode="json") for p in target.metal_primitives]},
        "coverage_policy": FULL_CONTACT_POLICY,
        "contact_regions_mm": [{"id": c.id, "role": c.role, "points_mm": [list(p) for p in c.points_mm]}
                               for c in target.contacts],
        "viewport_box_um": [min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin]}
    return {"version": "measured-region-layout-diagnostic-v1", "admission_status": "blocked",
        "admission_reasons": ["diagnostic_only", "measurement_and_registration_qualification_not_validated"],
        "scope": "Exact supplied regions at declared placements; pose-only margins exclude unknown measurement/registration error",
        "target_binding": target_import_binding(target), "sources": inputs, "policy": policy.model_dump(mode="json"),
        "source_pose_errors": pose, "nominal": _diagnostic_region_margins(target, placed, [0., 0.]),
        "pose_only": _diagnostic_region_margins(target, placed, errors),
        "quantized_pose_only": _diagnostic_region_margins(target, [item[0] for item in rounded],
            [error+item[1]["boundary_displacement_bound_mm"] for error, item in zip(errors, rounded, strict=True)]),
        "native_overlay_payload": payload, "input_sha256": geometry_digest({"sources": inputs,
            "target": target.model_dump(mode="json"), "policy": policy.model_dump(mode="json")})}


def build_design_pose_window_admission(design: DeviceDesignV1 | dict, *, verification: dict | None = None) -> DesignPoseWindowAdmission:
    parsed = DeviceDesignV1.model_validate(design)
    report = require_design_pose_window(parsed, verification=verification)
    return DesignPoseWindowAdmission(canonical_design_digest=parsed.canonical_digest,
        report_sha256=geometry_digest(report.model_dump(mode="json")), report=report)


def validate_design_pose_window_admission(design: DeviceDesignV1 | dict, admission: DesignPoseWindowAdmission | dict,
                                         *, verification: dict | None = None) -> DesignPoseWindowAdmission:
    """Recompute a new-job claim; a well-formed caller sidecar never waives screening."""
    saved = DesignPoseWindowAdmission.model_validate(admission)
    expected = build_design_pose_window_admission(design, verification=verification)
    if saved != expected:
        raise ValueError("pose-window admission differs from the exact design/configuration recomputation")
    return expected


def design_directory(store, design_id: str) -> Path:
    """Generated private artifacts accompany the run database, never caller paths."""
    if store.path == ":memory:":
        raise ValueError("KLayout artifacts require a file-backed run database")
    if not design_id.startswith("design_") or len(design_id) != 39 or any(
            c not in "0123456789abcdef" for c in design_id[7:]):
        raise ValueError("invalid generated design ID")
    root = Path(store.path).resolve().parent / "designs"
    directory = root / design_id
    if root.is_symlink() or directory.is_symlink():
        raise ValueError("design artifact directories cannot be symlinks")
    return directory


def _source(store, placement: FlakePlacement) -> DesignSource:
    sample = store.get_sample(placement.flake_id)
    if not sample or sample["kind"] != "flake":
        raise ValueError("selected source is not a known flake")
    observation = sample["props"].get("optical_geometry") or {}
    general = "region_mm" in observation
    (_pickup_region if general else _pickup_outline)(store, sample)
    event = store._x("SELECT MAX(id) AS id FROM event WHERE run_id=? AND kind='job.status' "
                     "AND json_extract(payload_json,'$.status')='succeeded'",
                     (observation["job_id"],)).fetchone()
    cls = RegionDesignSource if general else DesignSource
    shape = {"region_mm": observation["region_mm"]} if general else {"outline_mm": observation["outline_mm"]}
    return cls(flake_id=sample["id"], material=normalize_material_code(sample["props"].get("material")),
                       source_chip_id=sample["location"], **shape,
                       observation=observation, evidence_event_ids=(event["id"],),
                       image_sha256=hashlib.sha256(resolve_run_path(observation["image"]).read_bytes()).hexdigest(),
                       placement=placement)


def build_design_spec(store, project_id: str, placements: list[FlakePlacement],
                      contact_overlap_um: float, *, design_id: str, revision: int) -> DeviceDesignSpec:
    state = (store.get_project(project_id) or {}).get("state", {})
    plan = state.get("device_plan") or {}
    selected = state.get("selected_flakes") or []
    if len(selected) != 2 or sorted(p.flake_id for p in placements) != sorted(selected):
        raise ValueError("design must use exactly the current two selected flakes")
    snapshot = selection_source_snapshot(store, selected)
    if portable(state.get("selection_source_evidence")) != portable(snapshot):
        raise ValueError("selected source evidence changed or was not bound by Picker")
    chip = store.get_sample(plan.get("target_chip_id", ""))
    if not chip or chip["kind"] != "chip":
        raise ValueError("accepted guidance must identify the target chip")
    target = parse_hallbar_target(chip["props"].get("device_design_target"))
    if isinstance(target, ImportedHallBarTarget):
        validate_imported_target_files(store, target)
    size = chip["props"].get("size_mm")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(v) not in (int, float) for v in size) or tuple(size) != target.size_mm):
        raise ValueError("registered design dimensions differ from the current declared target chip size")
    if target.source == "simulated_design" and store.bench != "sim":
        raise ValueError("simulated target geometry cannot authorize a real design")
    if chip["props"].get("layers_top_first") != []:
        raise ValueError("target must be explicitly fresh and unlayered")
    if any(f["location"] == chip["id"] for f in store.list_samples(kind="flake")):
        raise ValueError("target already carries a recorded flake")
    sources = tuple(sorted((_source(store, p) for p in placements), key=lambda s: s.material != "hBN"))
    from qumus_simlab.config import load_yaml

    verification = load_yaml("verification.yaml")
    return DeviceDesignSpec(design_id=design_id, project_id=project_id, revision=revision,
        bench=store.bench, target_chip_id=chip["id"], target=target,
        guidance_sha256=geometry_digest(portable(plan)), selection_sha256=geometry_digest(portable(snapshot)),
        sources=sources, pickup_order=tuple(s.flake_id for s in sources), contact_overlap_um=contact_overlap_um,
        overlap_rule=FULL_CONTACT_POLICY if isinstance(target, ImportedHallBarTarget) else LEGACY_CONTACT_POLICY,
        pose_tolerance=DesignPoseTolerance(translation_um=verification["refind_flake"]["centre_tolerance_um"],
            rotation_deg=design_rotation_window_deg(verification)))


def imported_target_directory(workspace: Path, target: ImportedHallBarTarget) -> Path:
    root = Path(workspace).resolve() / "targets"
    directory = root / target.import_receipt.source_sha256
    if root.is_symlink() or directory.is_symlink() or directory.resolve().parent != root.resolve():
        raise ValueError("Imported target must use its generated immutable workspace directory")
    return directory


def validate_imported_target_files(store, target: ImportedHallBarTarget) -> Path:
    """No caller artifact path authorizes import; copied source and typed target agree."""
    if store.path == ":memory:":
        raise ValueError("Imported target needs a file-backed workspace")
    directory = imported_target_directory(Path(store.path).resolve().parent, target)
    path, definition = directory / "electrode-layout.gds", directory / "target.json"
    if any(p.is_symlink() or p.resolve().parent != directory.resolve() for p in (path, definition)):
        raise ValueError("Imported target source escaped its generated directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != target.import_receipt.source_sha256:
        raise ValueError("Original imported GDS bytes changed")
    if parse_hallbar_target(json.loads(definition.read_text())) != target:
        raise ValueError("Immutable imported target definition changed")
    return path


def layout_payload(spec: DeviceDesignSpec) -> dict:
    """DBU-quantized complete source/contact polygons; no clipping or simplification."""
    if any(isinstance(source, RegionDesignSource) for source in spec.sources):
        return _region_layout_payload(spec)
    if isinstance(spec.target, ImportedHallBarTarget):
        return _imported_layout_payload(spec)
    layers = []
    for i, source in enumerate(spec.sources, 1):
        layers.append({"name": source.flake_id, "layer": i, "points_dbu":
                       [[round(x * 1_000_000), round(y * 1_000_000)] for x, y in source.placed_outline()]})
    for i, contact in enumerate(sorted(spec.target.contacts, key=lambda c: c.id), 100):
        layers.append({"name": contact.id, "layer": i, "points_dbu":
                       [[round(x * 1_000_000), round(y * 1_000_000)] for x, y in contact.points_mm]})
    quantized = [convex_polygon((x / 1_000_000, y / 1_000_000) for x, y in item["points_dbu"])
                 for item in layers]
    if not contains(quantized[0], quantized[1]):
        raise ValueError("GDS quantization loses whole-Gr hBN coverage")
    for i, contact in enumerate(quantized[2:]):
        if area_um2(intersection(quantized[1], contact)) < spec.contact_overlap_um ** 2:
            raise ValueError("GDS quantization loses a required contact overlap")
        if any(intersection(contact, other) for other in quantized[2:2+i]):
            raise ValueError("GDS quantization merges distinct electrode regions")
    return {"dbu_um": 0.001, "top_cell": "QUMUS_DEVICE", "layers": layers}


def _region_layout_payload(spec):
    originals = [source_region(source) for source in spec.sources]
    rounded = [quantize_region(region) for region in originals]
    quantized = [item[0] for item in rounded]
    errors = [_region_source_errors(source, region, spec.pose_tolerance)[-1]
              for source, region in zip(spec.sources, originals, strict=True)]
    rounding = [item[1]["boundary_displacement_bound_mm"] for item in rounded]
    width, height = spec.target.size_mm
    chip = as_region(rectangle(width, height, (width/2, height/2)))
    if any(not region_covers_with_clearance(chip, region, error+delta)
           for region, error, delta in zip(quantized, errors, rounding, strict=True)):
        raise ValueError("GDS quantization cannot certify complete region fit within the target chip")
    if not region_covers_with_clearance(*quantized, sum(errors)+sum(rounding)):
        raise ValueError("GDS quantization cannot certify whole-Gr region coverage and its explicit uncertainty")
    for contact in spec.target.contacts:
        if not region_covers_with_clearance(quantized[1], contact.points_mm, errors[1]+rounding[1]):
            raise ValueError("GDS quantization cannot certify complete contact region: "+contact.id)
    layers = []
    for i, (source, (region, receipt)) in enumerate(zip(spec.sources, rounded, strict=True), 1):
        layers.append({"name": source.flake_id, "layer": i,
            "region_components_dbu": [{"exterior": [[round(x*1e6), round(y*1e6)] for x, y in c.exterior],
                "holes": [[[round(x*1e6), round(y*1e6)] for x, y in ring] for ring in c.holes]}
                for c in region.components], "quantization": receipt})
    imported = isinstance(spec.target, ImportedHallBarTarget)
    if imported:
        layers.append({"name": "full_imported_electrode_metal", "layer": 3,
            "polygons_dbu": [[list(p) for p in part.outline_dbu] for part in spec.target.metal_primitives]})
    else:
        for i, contact in enumerate(spec.target.contacts, 100):
            quantized_contact, receipt = quantize_region(contact.points_mm)
            if not region_covers_with_clearance(quantized[1], quantized_contact,
                    errors[1]+rounding[1]+receipt["boundary_displacement_bound_mm"]):
                raise ValueError("GDS quantization cannot certify rounded contact region: "+contact.id)
            layers.append({"name": contact.id, "layer": i,
                "points_dbu": [[round(x*1e6), round(y*1e6)] for x, y in contact.points_mm]})
    focus = [point for region in quantized for point in region_vertices(region)]
    focus += [point for c in spec.target.contacts for point in c.points_mm]
    xs, ys = ([point[axis]*1000 for point in focus] for axis in (0, 1))
    margin = max(max(xs)-min(xs), max(ys)-min(ys))*.12+1
    result = {"version": "qumus-device-layout-v3", "dbu_um": .001, "top_cell": "QUMUS_DEVICE",
        "layers": layers, "coverage_policy": spec.overlap_rule,
        "contact_regions_mm": [{"id": c.id, "role": c.role, "points_mm": [list(p) for p in c.points_mm]}
                               for c in spec.target.contacts],
        "viewport_box_um": [min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin]}
    if imported:
        result["source_target"] = {"import_receipt": spec.target.import_receipt.model_dump(mode="json"),
            "primitives": [p.model_dump(mode="json") for p in spec.target.metal_primitives]}
    return result


def _imported_layout_payload(spec):
    """New-only grouped physical layers; preserve the historical payload byte form."""
    sources = [[[round(x*1e6), round(y*1e6)] for x, y in source.placed_outline()] for source in spec.sources]
    quantized = [convex_polygon((x/1e6, y/1e6) for x, y in polygon) for polygon in sources]
    if not contains(quantized[0], quantized[1]):
        raise ValueError("GDS quantization loses whole-Gr hBN coverage")
    coverage = target_contact_coverage(spec.target, quantized[1], minimum_overlap_um=spec.contact_overlap_um)
    if coverage["status"] != "pass":
        raise ValueError("GDS quantization loses complete original contact ends: "+", ".join(coverage["failed_contacts"]))
    target = spec.target
    # The 2um cuts are continuous; no rounded reference polygon can waive them.
    ends = [{"id": c.id, "role": c.role, "points_mm": [list(p) for p in c.points_mm]} for c in target.contacts]
    focus = [point for polygon in quantized for point in polygon]+[p for c in target.contacts for p in c.points_mm]
    xs, ys = ([p[i]*1000 for p in focus] for i in (0, 1))
    margin = max(max(xs)-min(xs), max(ys)-min(ys))*.12+1
    return {"version": "qumus-device-layout-v2", "dbu_um": .001, "top_cell": "QUMUS_DEVICE",
        "layers": [{"name": source.flake_id, "layer": i, "polygons_dbu": [polygon]}
                   for i, (source, polygon) in enumerate(zip(spec.sources, sources, strict=True), 1)]+
                  [{"name": "full_imported_electrode_metal", "layer": 3,
                    "polygons_dbu": [[list(p) for p in part.outline_dbu] for part in target.metal_primitives]}],
        "source_target": {"import_receipt": target.import_receipt.model_dump(mode="json"),
                          "primitives": [p.model_dump(mode="json") for p in target.metal_primitives]},
        "coverage_policy": FULL_CONTACT_POLICY, "contact_regions_mm": ends,
        "viewport_box_um": [min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin]}


#: folders whose files keep their name relative to the folder in every process:
#: the brain and the laboratory hold the same pictures, designs and targets in
#: their own run folders (contract v1: the brain mirrors the lab's records)
PORTABLE_FOLDERS = ("images", "designs", "targets")


def portable(value: Any) -> Any:
    """`value` with each absolute file path under one of PORTABLE_FOLDERS written
    as "<folder>:<path inside it>": digests and equality checks of a design then
    agree between the brain's and the laboratory's copies of the same files.
    The files themselves stay bound by their sha256 fields."""
    if isinstance(value, str):
        text = value.replace("\\", "/")
        if text.startswith("qumus-image:"):  # a picture the laboratory names by its id
            return "images:"+text[len("qumus-image:"):]
        if text.startswith("/") or text[1:3] == ":/":
            parts = text.split("/")
            for i in range(len(parts)-2, -1, -1):
                if parts[i] in PORTABLE_FOLDERS:
                    return parts[i]+":"+"/".join(parts[i+1:])
        return value
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [portable(item) for item in value]
    return value


def _adopt_design_directory(artifact_path: str, directory: Path) -> None:
    """Copy a design made by the other process on this computer into this side's
    own generated directory (files are then checked against their sha256)."""
    import shutil

    source = Path(artifact_path).parent
    if directory.exists() or not source.is_dir() or source.name != directory.name or source.is_symlink():
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    partial = directory.with_name(directory.name+".partial")
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(source, partial, symlinks=False)
    partial.rename(directory)


def bind_artifact(spec: DeviceDesignSpec,
                  artifact: DesignArtifact | KLayoutMCPArtifact | KLayoutPythonArtifact) -> DeviceDesignV1:
    body = {**spec.model_dump(mode="json"), "artifact": artifact.model_dump(mode="json")}
    return DeviceDesignV1(**body, canonical_digest=geometry_digest(portable(body)))


def validate_design_current(store, project_id: str, record: DeviceDesignV1 | dict | None = None,
                            *, verify_artifact: bool = True) -> DeviceDesignV1:
    """Resolve exact current accepted binding before dispatch/resume; no device access."""
    state = (store.get_project(project_id) or {}).get("state", {})
    saved = state.get("device_design")
    design = DeviceDesignV1.model_validate(record if record is not None else saved)
    if portable(saved) != portable(design.model_dump(mode="json")) or not state.get("design_accepted"):
        raise ValueError("design is not the current accepted revision")
    current = build_design_spec(store, project_id, [s.placement for s in design.sources],
        design.contact_overlap_um, design_id=design.design_id, revision=design.revision)
    if portable(current.model_dump(mode="json")) != portable(design.model_dump(mode="json", exclude={"artifact", "canonical_digest"})):
        raise ValueError("design sources, guidance, target or provenance changed")
    return validate_design_artifact(store, design, verify_bytes=verify_artifact)


def validate_design_artifact(store, record: DeviceDesignV1 | dict, *, verify_bytes: bool = True) -> DeviceDesignV1:
    """Validate a saved immutable design without refreshing source/target facts.

    Submission uses validate_design_current. A resumed Flow must additionally
    explain changed specimens through its own verified pickup/release history.
    This helper alone never grants motion or current-source admission.
    """
    design = DeviceDesignV1.model_validate(record)
    if design.bench != store.bench:
        raise ValueError("saved design laboratory provenance differs from this store")
    if isinstance(design.target, ImportedHallBarTarget):
        validate_imported_target_files(store, design.target)
    expected = design_directory(store, design.design_id) / "device.gds"
    if portable(design.artifact.path) != f"designs:{design.design_id}/device.gds":
        raise ValueError("artifact is outside its generated run design directory")
    _adopt_design_directory(design.artifact.path, expected.parent)
    if expected.is_symlink():
        raise ValueError("artifact is outside its generated run design directory")
    if design.artifact.geometry_sha256 != geometry_digest(layout_payload(design)):
        raise ValueError("artifact geometry digest changed")
    if verify_bytes and hashlib.sha256(expected.read_bytes()).hexdigest() != design.artifact.sha256:
        raise ValueError("saved KLayout artifact changed or is missing")
    if isinstance(design.artifact, KLayoutMCPArtifact):
        validate_klayout_mcp_receipt(expected.parent, design, design.artifact, verify_bytes=verify_bytes)
    elif isinstance(design.artifact, KLayoutPythonArtifact):
        validate_klayout_python_receipt(expected.parent, design, design.artifact, verify_bytes=verify_bytes)
    return design


def validate_klayout_python_receipt(directory: Path, spec: DeviceDesignSpec,
                                    artifact: KLayoutPythonArtifact, *, verify_bytes: bool = True) -> KLayoutPythonReceipt:
    """Same binding checks as the MCP receipt, for the headless backend."""
    from PIL import Image

    receipt_path, script, image = (directory / name for name in ("verified.json", "roundtrip.py", "preview.png"))
    if any(path.is_symlink() or path.resolve().parent != directory.resolve()
           for path in (receipt_path, script, image)):
        raise ValueError("KLayout Python evidence is outside its generated directory")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact.receipt_sha256:
        raise ValueError("KLayout Python receipt changed")
    receipt = KLayoutPythonReceipt.model_validate(json.loads(raw))
    payload = layout_payload(spec)
    if (receipt.design_id != spec.design_id or receipt.geometry_sha256 != geometry_digest(payload)
            or receipt.geometry_sha256 != artifact.geometry_sha256
            or receipt.gds_sha256 != artifact.sha256
            or receipt.layer_names != tuple(item["name"] for item in payload["layers"])):
        raise ValueError("KLayout Python receipt differs from the bound design")
    if verify_bytes:
        pixels = image.read_bytes()
        if (hashlib.sha256(script.read_bytes()).hexdigest() != receipt.script_sha256
                or hashlib.sha256(pixels).hexdigest() != receipt.image_sha256):
            raise ValueError("KLayout Python script or preview image changed")
        with Image.open(BytesIO(pixels)) as png:
            if png.format != "PNG" or png.size != receipt.image_size:
                raise ValueError("KLayout preview image format or dimensions changed")
            png.verify()
    return receipt


def validate_klayout_mcp_receipt(directory: Path, spec: DeviceDesignSpec,
                               artifact: KLayoutMCPArtifact, *, verify_bytes: bool = True) -> KLayoutMCPReceipt:
    """The same artifact authority checks new MCP receipts and retained native pixels."""
    from PIL import Image

    receipt_path, script, image = (directory / name for name in ("verified.json", "roundtrip.py", "native.png"))
    if any(path.is_symlink() or path.resolve().parent != directory.resolve()
           for path in (receipt_path, script, image)):
        raise ValueError("KLayout MCP evidence is outside its generated directory")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact.receipt_sha256:
        raise ValueError("KLayout MCP receipt changed")
    receipt = KLayoutMCPReceipt.model_validate(json.loads(raw))
    payload = layout_payload(spec)
    if (receipt.design_id != spec.design_id or receipt.geometry_sha256 != geometry_digest(payload)
            or receipt.geometry_sha256 != artifact.geometry_sha256
            or receipt.gds_sha256 != artifact.sha256
            or receipt.layer_names != tuple(item["name"] for item in payload["layers"])):
        raise ValueError("KLayout MCP receipt differs from the bound design")
    if verify_bytes:
        pixels = image.read_bytes()
        if (hashlib.sha256(script.read_bytes()).hexdigest() != receipt.script_sha256
                or hashlib.sha256(pixels).hexdigest() != receipt.image_sha256):
            raise ValueError("KLayout MCP script or native image changed")
        with Image.open(BytesIO(pixels)) as png:
            if png.format != "PNG" or png.size != receipt.image_size:
                raise ValueError("KLayout native image format or dimensions changed")
            png.verify()
    return receipt
