"""Target-electrode requirements from declared design/calibration data.

These describe a straight bridge between opposed contacts. Other geometries
remain unresolved; an inventory label or uncalibrated picture is not a size.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from qumus_simlab.runtime.store import RunStore


class ContactPolygon(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1, max_length=100)
    kind: Literal["contact"]
    points_mm: tuple[tuple[float, float], ...]

    @field_validator("points_mm")
    @classmethod
    def _polygon(cls, value):
        from qumus_simlab.core.polygons import convex_polygon

        return convex_polygon(value)


class ElectrodeContactRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    region_id: str = Field(min_length=1, max_length=100)
    units: Literal["mm"]
    frame: Literal["chip_center_axes"]
    anchor_mm: tuple[float, float]
    contact_pair: tuple[str, str]
    polygons: tuple[ContactPolygon, ...]

    @model_validator(mode="after")
    def _pair(self):
        ids = [p.id for p in self.polygons]
        if len(set(ids)) != len(ids) or len(set(self.contact_pair)) != 2 or set(ids) != set(self.contact_pair):
            raise ValueError("a region needs exactly two uniquely named contacts, excluding gates")
        if not all(math.isfinite(v) for v in self.anchor_mm):
            raise ValueError("the region anchor must be finite")
        return self


class ElectrodeGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["design", "calibrated_image", "simulated_design"]
    reference: str = Field(min_length=1, max_length=400)
    units: Literal["um"]
    contact_gap_um: float = Field(gt=0, allow_inf_nan=False)
    contact_span_um: float = Field(gt=0, allow_inf_nan=False)
    contact_depth_um: float = Field(gt=0, allow_inf_nan=False)
    um_per_pixel: float | None = Field(None, gt=0, allow_inf_nan=False)
    contact_regions: tuple[ElectrodeContactRegion, ...] | None = None

    @field_validator("contact_regions")
    @classmethod
    def _regions(cls, value):
        if value is not None and len({r.region_id for r in value}) != len(value):
            raise ValueError("contact region IDs must be unique")
        return value


def target_geometry(store: RunStore, chip_id: str) -> dict:
    chip = store.get_sample(chip_id)
    if not chip or chip["kind"] != "chip":
        return {"status": "unresolved", "reason": "target is not a known chip"}
    raw = chip["props"].get("electrode_geometry")
    try:
        g = ElectrodeGeometry.model_validate(raw)
        if g.source == "simulated_design" and store.bench != "sim":
            raise ValueError("synthetic geometry is not physical lab evidence")
        if g.source == "calibrated_image" and g.um_per_pixel is None:
            raise ValueError("image geometry requires pixel calibration")
    except (ValidationError, ValueError) as exc:
        return {"status": "unresolved", "target_chip_id": chip_id,
                "reason": "No usable electrode design/calibration data: " + str(exc)[:400],
                "next_action": "Obtain target design dimensions or calibrated image analysis before final requirements."}
    data = g.model_dump(mode="json")
    if g.contact_regions is None:
        data.pop("contact_regions")  # Preserve the dimensions-only legacy snapshot.
    return {"status": "resolved", "target_chip_id": chip_id, **data,
            "note": "Straight contact bridge; area alone does not establish clean continuous fit."}


def target_snapshot(store: RunStore, chip_id: str) -> dict:
    """The target facts a requirement/layout relies on, independent of prose or pose."""
    chip = store.get_sample(chip_id)
    if not chip or chip["kind"] != "chip":
        raise ValueError("Device Guidance requires a known target_chip_id")
    props = chip["props"]
    patterned = bool(props.get("electrodes") or props.get("electrode_geometry")
                     or chip["status"] == "patterned")
    geometry = target_geometry(store, chip_id) if patterned else None
    if geometry and geometry["status"] != "resolved":
        raise ValueError("Target electrode geometry is unresolved; obtain trusted design/calibration data")
    return {"chip_id": chip_id, "prepatterned": patterned, "geometry": geometry,
            **({"size_mm": props.get("size_mm")} if geometry and geometry.get("contact_regions") else {})}


def guidance_binding(store: RunStore, plan: dict) -> dict:
    """Reject stale target facts and bind a layout to the complete accepted guidance."""
    target = target_snapshot(store, plan.get("target_chip_id") or "")
    if plan.get("target_snapshot") != target:
        raise ValueError("Target geometry changed or this guidance predates target binding; revalidate Device Guidance")
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"target_chip_id": target["chip_id"],
            "guidance_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


class BridgeRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)
    contact_overlap_um: float = Field(gt=0, allow_inf_nan=False)
    channel_width_um: float = Field(gt=0, allow_inf_nan=False)
    placement_margin_um: float = Field(ge=0, allow_inf_nan=False)
    rationale: str = Field(min_length=1, max_length=400)


def bridge_footprint(geometry: dict, requirement: BridgeRequirement) -> dict:
    r = requirement
    if r.contact_overlap_um + r.placement_margin_um > geometry["contact_depth_um"]:
        raise ValueError("requested overlap/margin exceeds the contact depth")
    if r.channel_width_um + 2 * r.placement_margin_um > geometry["contact_span_um"]:
        raise ValueError("channel width plus placement allowance exceeds the contact span")
    width = geometry["contact_gap_um"] + 2 * (r.contact_overlap_um + r.placement_margin_um)
    height = r.channel_width_um + 2 * r.placement_margin_um
    return {"width_um": width, "height_um": height, "min_area_um2": width * height,
            "target_chip_id": geometry["target_chip_id"], "source": geometry["source"],
            "reference": geometry["reference"], **r.model_dump(),
            "fit_rule": "A clean continuous inscribed rectangle must fit this footprint; total area is only a coarse filter."}


def geometry_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def target_contact_snapshot(store: RunStore, chip_id: str, region_id: str) -> dict:
    """Named public design facts, without an inferred specimen or receiving interface."""
    geometry = target_geometry(store, chip_id)
    if geometry["status"] != "resolved":
        raise ValueError("named contact geometry is unresolved")
    regions = [r for r in geometry.get("contact_regions") or [] if r["region_id"] == region_id]
    if len(regions) != 1:
        raise ValueError("named direct contact requires an explicit contact region; dimensions alone are insufficient")
    size = store.get_sample(chip_id)["props"].get("size_mm")
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in size)):
        raise ValueError("named contact geometry requires the declared chip size for BL/centre conversion")
    result = {"target_chip_id": chip_id, "source": geometry["source"], "reference": geometry["reference"],
              "size_mm": list(size), "region": regions[0], **{k: geometry[k] for k in
                  ("contact_gap_um", "contact_span_um", "contact_depth_um")}}
    return {**result, "geometry_sha256": geometry_digest(result)}


def actual_target_contact_snapshot(chip_id: str, size_mm, geometry: dict | None, named_region: dict) -> dict | None:
    """Normalize supplied actual polygons for D2, never regenerate intended shapes.

    The private evaluator can supply its persisted snapshot; control supplies
    public design data. Unknown legacy geometry/provenance remains unknown.
    """
    try:
        if not geometry or geometry.get("units") != "mm" or geometry.get("frame") != "chip_center_axes":
            return None
        pair = named_region["contact_pair"]
        polygons = [p for p in geometry["polygons"] if p.get("id") in pair]
        region = ElectrodeContactRegion.model_validate({**named_region, "polygons": [
            {"id": p["id"], "kind": p["kind"], "points_mm": p["points"]} for p in polygons]}).model_dump(mode="json")
        by_id = {p["id"]: p for p in region["polygons"]}
        left, right = (by_id[cid]["points_mm"] for cid in pair)
        lx, rx = max(x for x, _ in left), min(x for x, _ in right)
        tip = [y for x, y in right if x == rx]
        result = {"target_chip_id": chip_id, "source": geometry["source"], "reference": geometry["reference"],
            "size_mm": list(size_mm), "region": region, "contact_gap_um": (rx-lx)*1000,
            "contact_span_um": (max(tip)-min(tip))*1000,
            "contact_depth_um": (max(x for x, _ in right)-rx)*1000}
        return {**result, "geometry_sha256": geometry_digest(result)}
    except (ValueError, TypeError, KeyError):
        return None


def direct_contact_intent(store: RunStore, chip_id: str, region_id: str,
                          requirement: BridgeRequirement | dict) -> dict:
    """Target-only frozen guidance. Source choice remains the find/select task."""
    requirement = BridgeRequirement.model_validate(requirement)
    if requirement.placement_margin_um <= 0:
        raise ValueError("direct contact requires a declared positive placement margin")
    snapshot = target_contact_snapshot(store, chip_id, region_id)
    bridge_footprint({**snapshot, "target_chip_id": chip_id}, requirement)
    chip = store.get_sample(chip_id)
    if chip["props"].get("layers_top_first") != []:
        raise ValueError("direct contact requires an explicit fresh/unlayered receiving declaration; missing is unknown")
    if any(f["location"] == chip_id for f in store.list_samples(kind="flake")):
        raise ValueError("the declared fresh target already carries a recorded flake")
    intent = {"version": "direct-gr-contact-v1", "target_contact": snapshot,
              "direct_contact_material": "Gr", "stamp_film": "PC", "sole_transferred_layer": True,
              "target_declared_rotation_deg": 0.0,
              "receiving_layers_above_contacts": [], "bridge_requirement": requirement.model_dump()}
    return {**intent, "intent_sha256": geometry_digest(intent)}


class ContactLayer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    flake_id: str = Field(min_length=1)
    material: str = Field(min_length=1)


class DirectContactContract(BaseModel):
    """Resolved after public selection and before dispatch, never a campaign prerequisite."""
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    version: Literal["direct-gr-contact-v1"] = "direct-gr-contact-v1"
    target_intent: dict
    target_intent_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_flake_id: str
    source_geometry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_outline_mm: tuple[tuple[float, float], ...]
    source_outline_frame: Literal["flake_center_chip_axes"] = "flake_center_chip_axes"
    source_observation: dict
    outline_rotation_deg: float = Field(0, allow_inf_nan=False)
    incoming_layers_top_first: tuple[ContactLayer, ...]
    frame: Literal["chip_center_axes"] = "chip_center_axes"
    units: Literal["mm"] = "mm"
    stamp_film: Literal["PC"] = "PC"

    @model_validator(mode="after")
    def _binding(self):
        from qumus_simlab.core.polygons import convex_polygon

        convex_polygon(self.source_outline_mm)
        intent = dict(self.target_intent)
        digest = intent.pop("intent_sha256", None)
        if digest != self.target_intent_sha256 or geometry_digest(intent) != digest:
            raise ValueError("target intent digest changed")
        target = intent["target_contact"]
        g = dict(target)
        if geometry_digest({k: v for k, v in g.items() if k != "geometry_sha256"}) != g["geometry_sha256"]:
            raise ValueError("target geometry digest changed")
        ElectrodeContactRegion.model_validate(target["region"])
        r = BridgeRequirement.model_validate(intent["bridge_requirement"])
        if (intent.get("version") != self.version or intent.get("stamp_film") != "PC"
                or intent.get("direct_contact_material") != "Gr" or intent.get("sole_transferred_layer") is not True
                or intent.get("receiving_layers_above_contacts") != [] or r.placement_margin_um <= 0):
            raise ValueError("the direct Gr contract requires sole Gr, PC, positive margin and an explicit fresh receiver")
        if self.incoming_layers_top_first != (ContactLayer(flake_id=self.source_flake_id, material="Gr"),):
            raise ValueError("the incoming target-facing layer must be the sole selected Gr")
        if geometry_digest(self.source_observation) != self.source_geometry_sha256:
            raise ValueError("source observation digest changed")
        if self.source_observation.get("outline_mm") != [list(p) for p in self.source_outline_mm]:
            raise ValueError("source outline differs from the selected complete observation")
        return self


def resolve_direct_contact_contract(store, flake_ids: list[str], intent: dict,
                                    *, outline_rotation_deg: float | None = None) -> DirectContactContract:
    from qumus_simlab.core.materials import normalize_material_code

    target = intent["target_contact"]
    current = direct_contact_intent(store, target["target_chip_id"], target["region"]["region_id"], intent["bridge_requirement"])
    if current != intent:
        raise ValueError("direct contact target intent changed; revalidate guidance")
    if len(flake_ids) != 1:
        raise ValueError("direct Gr contact requires exactly one transferred Gr flake")
    f = store.get_sample(flake_ids[0])
    if not f or f["kind"] != "flake" or normalize_material_code(f["props"].get("material")) != "Gr":
        raise ValueError("direct Gr contact requires the selected Gr identity")
    _pickup_outline(store, f)  # Current complete calibrated observation, without taking a hull.
    if outline_rotation_deg is None:
        source_chip = store.get_sample(f["location"])
        rotation = source_chip["props"].get("rotation_deg")
        if isinstance(rotation, bool) or not isinstance(rotation, (float, int)) or not math.isfinite(rotation):
            raise ValueError("direct contact needs the current public source rotation for fixed-orientation screening")
        outline_rotation_deg = rotation-intent["target_declared_rotation_deg"]
    observation = json.loads(json.dumps(f["props"]["optical_geometry"], allow_nan=False))
    if not observation.get("image"):
        raise ValueError("selected Gr lacks its source image")
    from pathlib import Path

    observation["image_sha256"] = hashlib.sha256(Path(observation["image"]).read_bytes()).hexdigest()
    return DirectContactContract(target_intent=intent, target_intent_sha256=intent["intent_sha256"],
        source_flake_id=f["id"], source_geometry_sha256=geometry_digest(observation),
        source_outline_mm=observation["outline_mm"], source_observation=observation,
        outline_rotation_deg=outline_rotation_deg,
        incoming_layers_top_first=(ContactLayer(flake_id=f["id"], material="Gr"),))


def contact_bridge_geometry(outline_mm, target: dict, requirement: BridgeRequirement | dict) -> dict:
    """Shared straight-corridor math; layout is prospective, endpoint uses observed outline."""
    from qumus_simlab.core.polygons import (
        area_um2,
        containment_clearance,
        convex_polygon,
        intersection,
        rectangle,
    )

    try:
        polygon = convex_polygon(outline_mm)
        r = BridgeRequirement.model_validate(requirement)
        region = ElectrodeContactRegion.model_validate(target["region"]).model_dump(mode="json")
        footprint = bridge_footprint(target, r)
        corridor = rectangle(footprint["width_um"]/1000, footprint["height_um"]/1000, tuple(region["anchor_mm"]))
        clearance = containment_clearance(polygon, corridor)
        contacts = {p["id"]: convex_polygon(p["points_mm"]) for p in region["polygons"]}
        # The declared straight corridor must itself reach both actual named
        # contacts by overlap+margin, across the full channel width+margin.
        overlap, bands_fit = {}, True
        band_width = (r.contact_overlap_um+r.placement_margin_um)/1000
        x, y = region["anchor_mm"]
        for index, cid in enumerate(region["contact_pair"]):
            side = -1 if index == 0 else 1
            band = rectangle(band_width, footprint["height_um"]/1000,
                (x+side*(target["contact_gap_um"]/2000+band_width/2), y))
            bands_fit &= containment_clearance(contacts[cid], band) >= -1e-12
            overlap[cid] = area_um2(intersection(polygon, contacts[cid]))
        metrics = dict(required_footprint_um={"width": footprint["width_um"], "height": footprint["height_um"]},
            corridor_clearance_um=clearance*1000, contact_overlap_um2=overlap,
            contact_bands_supported=bool(bands_fit), full_outline_area_um2=area_um2(polygon))
        passed = clearance >= -1e-12 and bands_fit and all(a > 0 for a in overlap.values())
        return {"status": "pass" if passed else "fail", "metrics": metrics}
    except (ValueError, TypeError, KeyError) as exc:
        return {"status": "unresolved", "metrics": {}, "reason": str(exc)[:400]}


def direct_contact_result(actual_gr: dict, actual_target: dict,
                          contract: DirectContactContract | dict) -> dict:
    """One pure whole-footprint/interface predicate for public consumers and private Eval.

    Actual outlines are already in chip-centre axes. This function never places
    the saved source at the intended target or reads simulator/ledger state.
    Only supported convex connected outlines qualify; no hull or growth repair.
    """
    from qumus_simlab.core.polygons import convex_polygon, transform

    result = {"status": "unresolved", "metrics": {}, "binding": {}, "reasons": [],
              "scope": "geometric/recorded-layer interface only; no electrical or chemical claim"}
    try:
        c = DirectContactContract.model_validate(contract)
        target = c.target_intent["target_contact"]
        region = target["region"]
        result["binding"] = {"contract_sha256": geometry_digest(c.model_dump(mode="json")),
            "target_intent_sha256": c.target_intent_sha256, "target_chip_id": target["target_chip_id"],
            "region_id": region["region_id"], "contact_pair": region["contact_pair"],
            "target_geometry_sha256": target["geometry_sha256"], "source_flake_id": c.source_flake_id,
            "source_geometry_sha256": c.source_geometry_sha256, "stamp_film": "PC", "frame": c.frame, "units": c.units}
        required = ("flake_id", "material", "outline_mm", "outline_complete", "frame", "units", "stamp_film",
                    "incoming_layers_top_first", "source_geometry_sha256", "observation_binding")
        if any(actual_gr.get(k) is None for k in required) or any(actual_target.get(k) is None for k in
                ("chip_id", "geometry", "receiving_layers_above_contacts", "before_observation_binding")):
            raise ValueError("actual complete outline, interface, pose and image/snapshot evidence are required")
        if not actual_gr["observation_binding"] or not actual_target["before_observation_binding"]:
            raise ValueError("actual endpoint and receiving baseline need their evidence bindings")
        wrong = (actual_gr["flake_id"] != c.source_flake_id or actual_gr["material"] != "Gr"
            or actual_gr["stamp_film"] != "PC" or actual_gr["frame"] != c.frame or actual_gr["units"] != c.units
            or actual_gr["incoming_layers_top_first"] != [layer.model_dump() for layer in c.incoming_layers_top_first]
            or actual_gr["source_geometry_sha256"] != c.source_geometry_sha256
            or actual_target["chip_id"] != target["target_chip_id"]
            or actual_target["geometry"] != target or actual_target["receiving_layers_above_contacts"] != [])
        if wrong:
            return {**result, "status": "fail", "reasons": ["actual identity, PC film, named contacts or recorded target-facing interface differs"]}
        if actual_gr["outline_complete"] is not True:
            raise ValueError("terminal Gr outline is incomplete")
        polygon = convex_polygon(actual_gr["outline_mm"])
        source = transform(convex_polygon(c.source_outline_mm), rotation_deg=c.outline_rotation_deg)
        # Translation preserves every vertex/edge, full shape and fixed twist. Numeric
        # tolerance is 1e-9 mm, not a placement allowance or an optical waiver.
        def shape_error(a, b):
            if len(a) != len(b):
                return math.inf
            best = math.inf
            for shift in range(len(b)):
                ordered = b[shift:]+b[:shift]
                best = min(best, max(math.dist((x-a[0][0], y-a[0][1]),
                    (u-ordered[0][0], v-ordered[0][1]))
                    for (x,y),(u,v) in zip(a, ordered, strict=True)))
            return best
        error = shape_error(polygon, source)
        result["metrics"]["full_shape_error_mm"] = error if math.isfinite(error) else None
        if error > 1e-9:
            return {**result, "status": "fail", "reasons": ["terminal Gr does not preserve the complete selected source shape and fixed orientation"]}
        geometry = contact_bridge_geometry(polygon, target, c.target_intent["bridge_requirement"])
        result["metrics"].update(geometry["metrics"])
        passed = geometry["status"] == "pass"
        return {**result, "status": "pass" if passed else "fail", "reasons": [] if passed else
                ["the full Gr footprint must continuously bridge the gap and cover both declared positive contact overlap/margin bands"]}
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {**result, "reasons": [str(exc)[:400]]}


def source_evidence_current(store, flake: dict, evidence: dict) -> bool:
    """Existing event history invalidates observations after specimen replacement."""
    import json

    job_id = evidence.get("job_id", "")
    row = store._x("SELECT MAX(id) AS id FROM event WHERE run_id=? AND kind='job.status' "
                   "AND json_extract(payload_json,'$.status')='succeeded'", (job_id,)).fetchone()
    if not row or row["id"] is None or evidence.get("source_chip_id") != flake["location"]:
        return False
    after = row["id"]
    for event in store._x("SELECT kind,payload_json FROM event WHERE id>? AND kind IN "
                         "('job.submitted','sample.updated','sample.removed')", (after,)):
        payload = json.loads(event["payload_json"])
        if event["kind"] == "job.submitted" and payload.get("flow") == "exfoliate_chips":
            if flake["location"] in payload.get("params", {}).get("chip_ids", []):
                return False
        if payload.get("id") == flake["location"]:
            if event["kind"] == "sample.removed" or payload.get("status") in ("bare", "exfoliated"):
                return False
            material = payload.get("props", {}).get("material")
            if material and material != flake["props"].get("material"):
                return False
        if payload.get("id") == flake["id"] and event["kind"] == "sample.removed":
            return False
    return not flake["props"].get("location_uncertain")


def selection_source_snapshot(store, flake_ids: list[str]) -> dict:
    """Bind an accepted choice to observed identity/geometry, not just reusable IDs."""
    snapshots = {}
    for fid in flake_ids:
        sample = store.get_sample(fid)
        snapshots[fid] = {"location": sample["location"], **{key: sample["props"].get(key)
            for key in ("material", "images", "optical_geometry")}}
    return snapshots


def _pickup_outline(store, flake: dict, placement: dict | None = None):
    """One current observation, shared by admission and scripted ranking."""
    from qumus_simlab.core.polygons import convex_polygon, transform

    g = flake["props"].get("optical_geometry") or {}
    if "region_mm" in g:
        raise ValueError("General observed regions require the shared region admission path")
    run = store.get_run(g.get("job_id", ""))
    if (not g.get("image") or g.get("source_chip_id") != flake["location"]
            or g.get("units") != "mm" or g.get("frame") != "flake_center_chip_axes"
            or not run or run["status"] != "succeeded" or not source_evidence_current(store, flake, g)
            or g.get("source") not in ("simulated_optical_outline", "calibrated_optical_outline", "rgb_optical_outline")
            or (store.bench != "sim" and g.get("source") == "simulated_optical_outline")):
        raise ValueError("missing current calibrated outline/image provenance")
    if g.get("source") == "rgb_optical_outline" or "recognition_chain" in g:
        from qumus_simlab.records.recognition import (
            validate_rgb_capture_binding,
            validate_rgb_outline,
        )

        if g.get("source") != "rgb_optical_outline":
            raise ValueError("RGB recognition evidence cannot use a legacy outline source")
        validate_rgb_capture_binding(store, flake, g)
        validate_rgb_outline(g, material=flake["props"].get("material"))
    p = placement or {}
    return transform(convex_polygon(g["outline_mm"]), xy=tuple(p.get("xy_mm", (0, 0))),
                     rotation_deg=p.get("rotation_deg", 0))


def _pickup_region(store, flake: dict, placement: dict | None = None):
    """Current complete source region; original pixels and all topology retained."""
    from qumus_simlab.core.regions import as_region, transform_region
    from qumus_simlab.records.recognition import validate_rgb_capture_binding, validate_rgb_outline

    geometry = flake["props"].get("optical_geometry") or {}
    if "region_mm" not in geometry:
        return as_region(_pickup_outline(store, flake, placement))
    if geometry.get("source") == "rgb_native100_measured_region":
        raise ValueError("Historical derived100 geometry is not admitted by the current original20-mask policy; "
                         "obtain a fresh normal20X source observation")
    run = store.get_run(geometry.get("job_id", ""))
    if (not geometry.get("image") or geometry.get("source_chip_id") != flake["location"]
            or geometry.get("units") != "mm" or geometry.get("frame") != "flake_center_chip_axes"
            or not run or run["status"] != "succeeded" or not source_evidence_current(store, flake, geometry)
            or geometry.get("source") != "rgb_optical_outline" or store.bench != "sim"):
        raise ValueError("Missing current RGB region provenance or unsupported non-SIM uncertainty profile")
    validate_rgb_capture_binding(store, flake, geometry)
    validate_rgb_outline(geometry, material=flake["props"].get("material"))
    region = as_region(geometry["region_mm"])
    if not region.components:
        raise ValueError("A pickup source cannot be an empty material region")
    placement = placement or {}
    return transform_region(region, xy=tuple(placement.get("xy_mm", (0, 0))),
                            rotation_deg=placement.get("rotation_deg", 0))


def pickup_coverage(outer, inner, *, frame: str, binding: dict, placement_mm=(0., 0.), required_bound=None) -> dict:
    """Complete fixed-twist outlines, with an optional applicable positioning bound.

    Placement translates the OUTER carried body in the stated common frame.
    An unknown future localization budget is nominal screening, never zero error.
    The caller owns evidence currentness; exact bound/binding equality prevents
    applying a different specimen, exposure or assessment-time budget.
    """
    from qumus_simlab.core.polygons import containment_clearance, convex_polygon, transform
    from qumus_simlab.core.regions import PolygonRegion, region_clearance, transform_region

    result = {"state": "unresolved", "frame": frame if isinstance(frame,str) else None,
              "units": "mm", "binding": None,
              "nominal_clearance_mm": None, "required_clearance_mm": None, "headroom_mm": None}
    try:
        if not isinstance(frame, str) or not frame or not isinstance(binding, dict) or not binding:
            raise ValueError("coverage needs an explicit common frame and evidence binding")
        result["binding"] = json.loads(json.dumps(binding, allow_nan=False))
        if (len(placement_mm) != 2 or any(isinstance(v, bool) or not isinstance(v, (int,float))
                or not math.isfinite(v) for v in placement_mm)):
            raise ValueError("coverage placement needs two finite coordinates")
        general = isinstance(outer, (PolygonRegion, dict)) or isinstance(inner, (PolygonRegion, dict))
        if general:
            clearance = region_clearance(transform_region(outer, xy=placement_mm), inner)
            result["nominal_contains"] = clearance is not None
        else:
            outer, inner = convex_polygon(outer), convex_polygon(inner)
            clearance = containment_clearance(transform(outer, xy=placement_mm), inner)
        result.update(nominal_clearance_mm=clearance, placement_mm=list(placement_mm))
        if required_bound is None:
            return {**result, "state": "nominal_only"}
        json.dumps(required_bound, allow_nan=False)
        required = required_bound.get("required_clearance_mm")
        if (required_bound.get("binding") != binding or required_bound.get("frame") != frame
                or not isinstance(required_bound.get("provenance"), str) or not required_bound["provenance"]
                or isinstance(required, bool) or not isinstance(required, (int, float))
                or not math.isfinite(required) or required < 0):
            raise ValueError("coverage bound is missing, nonfinite or belongs to different evidence")
        result.update(required_clearance_mm=required, headroom_mm=None if clearance is None else clearance-required,
                      bound_provenance=required_bound["provenance"],
                      state="bound_satisfied" if clearance is not None and clearance >= required else "bound_failed")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result.update(state="unresolved", reason=str(exc))
    return result


def pickup_coverage_errors(store, flake_ids: list[str], placements: dict | None = None) -> list[str]:
    """Common whole-stack check. Fixed receiving substrates are not pickup layers.

    Footprints are observation-derived, centre-relative, in source-chip axes.
    Placement transforms are millimetres/degrees in a shared assembly frame.
    Missing/legacy display geometry cannot establish containment. Legacy convex
    outlines and current original-pixel regions share whole-material coverage;
    no exterior, hole or component may be replaced by a hull.
    """
    from qumus_simlab.core.materials import normalize_material_code

    samples = [store.get_sample(fid) for fid in flake_ids]
    if any(not f or f["kind"] != "flake" for f in samples):
        return ["Coverage needs known pickup flakes."]
    if len(samples) < 2:
        return []
    if normalize_material_code(samples[0]["props"].get("material")) != "hBN":
        return ["The first pickup layer must be hBN and cover every underlying lifted layer."]
    outlines, errors = [], []
    for f in samples:
        try:
            geometry = f["props"].get("optical_geometry") or {}
            reader = _pickup_region if "region_mm" in geometry else _pickup_outline
            outlines.append(reader(store, f, (placements or {}).get(f["id"])))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"{f['id']}: {exc}; request targeted optical evidence.")
    if errors:
        return errors
    top = outlines[0]
    for f, polygon in zip(samples[1:], outlines[1:], strict=True):
        coverage = pickup_coverage(top, polygon, frame="assembly", binding={"flake_ids": flake_ids})
        if (coverage["state"] != "nominal_only" or coverage["nominal_clearance_mm"] is None
                or coverage["nominal_clearance_mm"] < 0):
            errors.append(f"Top hBN does not fully cover {f['id']} at the proposed placement (outline containment, not area).")
    return errors


def ranked_pickup_combinations(store, choices: list[list[str]]) -> list[dict]:
    """Nominal compatible combinations, ranked only from current optical evidence.

    Each list is one ordered pickup slot. Callers supply role-eligible IDs;
    physical identity, hBN containment and provenance are common rules here.
    This scripted preference does not certify later noisy pickup or optical quality.
    """
    from itertools import product

    from qumus_simlab.core.materials import normalize_material_code
    from qumus_simlab.records.materials import pickup_admission_error

    if not choices or any(not slot for slot in choices):
        return []
    samples = {fid: store.get_sample(fid) for slot in choices for fid in slot}
    outlines = {}
    for fid, sample in samples.items():
        if not sample or sample["kind"] != "flake" or pickup_admission_error(store, sample):
            continue
        try:
            geometry = sample["props"].get("optical_geometry") or {}
            outlines[fid] = (_pickup_region if "region_mm" in geometry else _pickup_outline)(store, sample)
        except (ValueError, TypeError, KeyError):
            continue
    ranked = []
    for ids in product(*choices):
        if len(set(ids)) != len(ids) or any(fid not in outlines for fid in ids):
            continue
        if len(ids) > 1 and normalize_material_code(samples[ids[0]]["props"].get("material")) != "hBN":
            continue
        top = outlines[ids[0]]
        coverage = [pickup_coverage(top, outlines[fid], frame="assembly", binding={"flake_ids": list(ids)})
                    for fid in ids[1:]]
        if all(c["state"] == "nominal_only" and c["nominal_clearance_mm"] is not None
               and c["nominal_clearance_mm"] >= 0 for c in coverage):
            ranked.append({"flake_ids": list(ids), "nominal_clearance_mm":
                min((c["nominal_clearance_mm"] for c in coverage), default=0.0),
                "coverage_state": "nominal_only", "coverage": coverage,
                "preparation_scope": "conditional; future carried localization budget is unknown"})
    return sorted(ranked, key=lambda item: item["nominal_clearance_mm"], reverse=True)


def current_pickup_evidence(store, flake: dict) -> dict | None:
    if flake["props"].get("location_uncertain") or (store.get_sample(flake["location"]) or {}).get("props", {}).get("location_uncertain"):
        return None
    evidence = flake["props"].get("pickup_assessment") or flake["props"].get("reactor_pickup_assessment") or {}
    run = store.get_run(evidence.get("job_id", ""))
    if (evidence.get("flake_id") != flake["id"]
            or evidence.get("source_chip_id") != flake["location"] or not evidence.get("images")
            or not run or run["status"] != "succeeded" or not source_evidence_current(store, flake, evidence)
            or run["flow"] not in ("search_chip", "revisit_chip", "assess_reactor_inputs")):
        return None
    return evidence
