"""Read original RGB and paired mask evidence without changing its geometry."""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
from PIL import Image

from qumus_simlab.core.run_paths import resolve_run_path


def validate_native_anchor_event(store, event_id):
    """Join native anchor evidence to original journal events; no physical admission."""
    from qumus_simlab.perception.observations import (
        native100_source_anchor,
        register_rgb_measurements,
    )

    def event(identifier, kind, *, run_id=None, before=None):
        if type(identifier) is not int or identifier <= 0 or before is not None and identifier >= before:
            raise ValueError("Native anchor event order is invalid")
        row = store._x("SELECT id,run_id,kind,payload_json FROM event WHERE id=?", (identifier,)).fetchone()
        if not row or row["kind"] != kind or run_id is not None and row["run_id"] != run_id:
            raise ValueError("Native anchor event is missing or belongs to another job")
        return {"id": row["id"], "run_id": row["run_id"], "kind": row["kind"],
                "payload": json.loads(row["payload_json"])}

    try:
        parent = event(event_id, "microscope.native_source_anchor")
        payload, run_id = parent["payload"], parent["run_id"]
        measurements = {}
        for objective in (20, 100):
            observed = event(payload[f"measurement_event_{objective}x"], "microscope.rgb_measurement",
                             run_id=run_id, before=event_id)
            body = observed["payload"]
            if (body["objective"] != objective or body["source_chip_id"] != payload["source_chip_id"]
                    or body["flake_id"] != payload["flake_id"]):
                raise ValueError("Native measurement belongs to another material source")
            rotation = body["chip_rotation_evidence"]
            for key, kind in (("rotation_event", "step.verified"), ("capture_event", "device.command")):
                if event(rotation[key]["id"], kind, run_id=run_id, before=observed["id"]) != rotation[key]:
                    raise ValueError("Native anchor claims different original rotation/capture evidence")
            if body["capture_event_id"] != rotation["capture_event"]["id"]:
                raise ValueError("Native measurement capture link differs")
            measurements[objective] = body
        first, current = measurements[20], measurements[100]
        if first["capture_event_id"] >= current["capture_event_id"]:
            raise ValueError("Native100 capture does not follow its original20 identity")
        registration = register_rgb_measurements(first["measurement"], [current["measurement"]])
        expected = native100_source_anchor(current["measurement"],
            first["measurement"]["original_recognition_chain"],
            chip_rotation_evidence=current["chip_rotation_evidence"], registration_evidence=registration)
        if payload["anchor"] != expected or payload.get("admission_status") != "blocked":
            raise ValueError("Native anchor does not reproduce its original journal evidence")
        return expected
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("Native anchor journal binding is incomplete") from exc


def native_source_observation(store, event_id):
    """Publish measured geometry under the owner's inclusive engineering budget."""
    from qumus_simlab.perception.measurement import owner_error_budget_policy

    anchor = validate_native_anchor_event(store, event_id)
    row = store._x("SELECT run_id,payload_json FROM event WHERE id=?", (event_id,)).fetchone()
    body = json.loads(row["payload_json"])
    measured = store._x("SELECT payload_json FROM event WHERE id=?",
                        (body["measurement_event_100x"],)).fetchone()
    receipt = json.loads(measured["payload_json"])["measurement"]
    policy = owner_error_budget_policy()
    return {"source": "rgb_native100_measured_region", "units": "mm", "frame": anchor["frame"],
            "region_mm": anchor["region_mm"], "source_chip_id": body["source_chip_id"],
            "image": anchor["native100_image"]["path"], "job_id": row["run_id"],
            "chip_rotation_deg": anchor["chip_rotation_deg"],
            "recognition_chain": receipt["original_recognition_chain"],
            "original20_identity": anchor["original20_identity"],
            "native_source_anchor_event_id": event_id,
            "native_source_anchor_sha256": anchor["anchor_receipt_sha256"],
            "owner_error_budget": policy, "boundary_uncertainty_mm": .001,
            "boundary_uncertainty_provenance": policy}


def validate_native_source_observation(store, flake, observation):
    """Require current original flow evidence; engineering budget is not calibration."""
    from qumus_simlab.perception.measurement import validate_owner_error_budget
    from qumus_simlab.records.geometry import source_evidence_current

    validate_owner_error_budget(observation)
    identifier = observation.get("native_source_anchor_event_id")
    expected = native_source_observation(store, identifier)
    row = store._x("SELECT run_id,payload_json FROM event WHERE id=?", (identifier,)).fetchone()
    payload = json.loads(row["payload_json"])
    run = store.get_run(row["run_id"])
    completion = store._x("SELECT MIN(id) AS id FROM event WHERE run_id=? AND kind='job.status' "
        "AND json_extract(payload_json,'$.status')='succeeded'", (row["run_id"],)).fetchone()
    if (observation != expected or store.bench != "sim" or not run
            or run["flow"] not in ("search_chip", "revisit_chip") or run["status"] != "succeeded"
            or not completion or completion["id"] is None or identifier >= completion["id"]
            or payload["flake_id"] != flake["id"] or payload["source_chip_id"] != flake["location"]
            or expected["recognition_chain"]["material"] != flake["props"].get("material")
            or not source_evidence_current(store, flake, expected)):
        raise ValueError("Native source differs from its current original completed observation")
    return expected


def validate_boundary_uncertainty(observation: dict) -> float:
    """An independently qualified, image-bound SIM measurement budget."""
    from qumus_simlab.perception.measurement import validate_qualified_boundary
    from qumus_simlab.records.geometry import geometry_digest

    try:
        bound = observation["boundary_uncertainty_mm"]
        provenance = observation["boundary_uncertainty_provenance"]
        chain = observation["recognition_chain"]
        if type(bound) not in (int, float) or not math.isfinite(bound) or bound <= 0:
            raise ValueError("New RGB regions need an explicit positive boundary uncertainty")
        if (not isinstance(provenance, dict)
                or provenance.get("version") != "sim-boundary-uncertainty-v1"
                or provenance.get("source") != "simulation_observation_profile"
                or not isinstance(provenance.get("profile_id"), str) or not provenance["profile_id"]
                or not isinstance(provenance.get("profile_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", provenance["profile_sha256"]) is None
                or provenance.get("image_sha256") != chain["image"]["sha256"]
                or provenance.get("calibration_sha256") != geometry_digest(chain["calibration"])):
            raise ValueError("RGB boundary uncertainty lacks its original SIM profile/image/calibration binding")
        return validate_qualified_boundary(observation, bound)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Missing source-bound RGB boundary uncertainty") from exc


def _needs_boundary_qualification(observation):
    """New regions/native RGB cannot use the historical convex-outline route."""
    chain = observation.get("recognition_chain", {})
    return ("region_mm" in observation
        or chain.get("calibration", {}).get("material_capture_profile") is not None
        or chain.get("material_policy", {}).get("capture_profile") is not None)


def _validate_rgb_geometry_budget(observation):
    """The owner accepts original20 pixels, not a calibrated boundary claim."""
    if "owner_error_budget" not in observation:
        if _needs_boundary_qualification(observation):
            validate_boundary_uncertainty(observation)
        return
    from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE
    from qumus_simlab.perception.measurement import validate_owner_error_budget
    from qumus_simlab.perception.rgb_material import algorithm_sha256

    validate_owner_error_budget(observation)
    chain = observation["recognition_chain"]
    if (observation.get("geometry_policy") != "original20-selected-pixel-union-v1"
            or observation.get("source") != "rgb_optical_outline" or "region_mm" not in observation
            or observation.get("frame") != "flake_center_chip_axes"
            or observation.get("boundary_uncertainty_provenance") != observation["owner_error_budget"]
            or chain["image"]["objective"] != 20
            or chain["calibration"].get("material_capture_profile") != V1_MATERIAL_CAPTURE_PROFILE
            or chain["algorithm_sha256"] != algorithm_sha256()):
        raise ValueError("Owner original20 geometry policy or current recognition domain differs")


def validate_recognition_chain(chain: dict, *, image_path: str, objective: int = 20) -> dict:
    """Verify original bytes and exact component selection, never a repaired mask.

    This establishes evidence identity, not material accuracy or support by the
    polygon engine. The existing geometry/topology admission remains separate.
    """
    try:
        if (chain["version"] != "v1-rgb-mask-chain-v1" or chain["backend"] != "v1_lab_rgb"
                or chain["image"]["objective"] != objective
                or type(chain["image"]["objective"]) is not int):
            raise ValueError("Recognition backend/objective binding changed")
        for key in ("algorithm_sha256", "material_policy_sha256"):
            if not isinstance(chain[key], str) or re.fullmatch(r"[0-9a-f]{64}", chain[key]) is None:
                raise ValueError("Recognition algorithm/material policy identity is absent")
        if resolve_run_path(chain["image"]["path"]) != resolve_run_path(image_path):
            raise ValueError("Recognition mask belongs to a different original image")
        arrays, paths = {}, []
        for key in ("image", "original_mask", "geometry_mask"):
            receipt = chain[key]
            path = resolve_run_path(receipt["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError("Original recognition image/mask bytes changed")
            with Image.open(path) as image:
                if list(image.size) != chain["image"]["size_px"]:
                    raise ValueError("Recognition image and masks have different raster dimensions")
                arrays[key] = np.asarray(image).copy()
            paths.append(path)
        if len(set(paths)) != 3:
            raise ValueError("Original image and both mask identities must remain distinct")
        labels, geometry = arrays["original_mask"], arrays["geometry_mask"]
        if (labels.ndim != 2 or labels.dtype.kind not in "ui" or labels.min() < 0
                or labels.max() > 65535 or geometry.ndim != 2
                or not np.isin(geometry, (0, 1, 255)).all()):
            raise ValueError("Invalid original label mask or binary geometry mask")
        selected = chain["selected_label_ids"]
        if (not isinstance(selected, list) or len(selected) != 1
                or type(selected[0]) is not int or not 1 <= selected[0] <= 65535):
            raise ValueError("Geometry requires one explicitly selected original label")
        original_component = labels == selected[0]
        if not original_component.any() or not np.array_equal(geometry != 0, original_component):
            raise ValueError("Geometry mask expanded, truncated or filled the original component")
        derivation = chain["derivation"]
        if (derivation["policy"] != "center-priority-original-component-v1"
                or derivation["operation"] != "select_original_label_only"
                or derivation["morphology"] != "none"
                or derivation["selected_label_id"] != selected[0]
                or not isinstance(chain["calibration"], dict) or not chain["calibration"]):
            raise ValueError("Original mask derivation or calibration binding changed")
        return chain
    except (KeyError, TypeError, AttributeError, OSError, IndexError) as exc:
        raise ValueError("Missing or malformed original RGB/mask evidence") from exc


def validate_rgb_outline(observation: dict, *, material: str) -> None:
    """Reconstruct the admitted outline from original selected pixel cells.

    No contour metadata, convex hull, morphological repair or claimed area can
    substitute for the recorded pixels. Legacy observations use their existing
    admission contract and do not call this validator.
    """
    from qumus_simlab.core.polygons import convex_polygon, transform
    from qumus_simlab.core.regions import as_region, transform_region
    from qumus_simlab.perception.rgb_material import component_geometry

    try:
        chain = validate_recognition_chain(observation["recognition_chain"],
                                           image_path=observation["image"])
        calibration = chain["calibration"]
        if calibration.get("image_size_px") != chain["image"]["size_px"]:
            raise ValueError("RGB calibration and original raster dimensions differ")
        rotation = observation["chip_rotation_deg"]
        if type(rotation) not in (int, float) or not math.isfinite(rotation):
            raise ValueError("RGB outline needs the observed finite chip rotation")
        if material not in ("Gr", "hBN"):
            raise ValueError("Unsupported RGB source material")
        policy = chain["material_policy"]
        if (chain["material"] != material or policy["material"] != material
                or policy["objective"] != 20 or policy["backend"] != chain["backend"]
                or hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
                != chain["material_policy_sha256"]):
            raise ValueError("RGB material policy differs from its saved source binding")
        with Image.open(resolve_run_path(chain["geometry_mask"]["path"])) as image:
            geometry = component_geometry(np.asarray(image) != 0, calibration)
        if geometry["status"] != "supported":
            raise ValueError("Original RGB geometry is unsupported: " + str(geometry["unsupported_reason"]))
        derivation = chain["derivation"]
        if (derivation.get("status") != "supported"
                or derivation.get("ambiguity_status") != "unique"
                or derivation.get("selection_context") != "v1_center_priority_revisit"
                or derivation.get("rings_px") != geometry["rings_px"]
                or derivation.get("centroid_px") != geometry["centroid_px"]):
            raise ValueError("Declared RGB derivation differs from original pixel cells")
        _validate_rgb_geometry_budget(observation)
        if "owner_error_budget" in observation:
            anchor = observation["source_anchor20"]
            stage = anchor["capture_stage_xy"]
            if (anchor["version"] != "original20-mask-centroid-stage-v1" or anchor["objective"] != 20
                    or anchor["image_sha256"] != chain["image"]["sha256"]
                    or anchor["centroid_px"] != geometry["centroid_px"]
                    or anchor["detection_offset_mm"] != [geometry["dx_mm"], geometry["dy_mm"]]
                    or len(stage) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in stage)
                    or anchor["stage_xy"] != [stage[0]-geometry["dx_mm"], stage[1]-geometry["dy_mm"]]):
                raise ValueError("Original20 source anchor differs from its exact selected mask centroid")
        if "region_mm" in observation:
            expected_region = transform_region(geometry["region_mm"], rotation_deg=-rotation)
            actual_region = as_region(observation["region_mm"])
            if (len(expected_region.components) != len(actual_region.components)
                    or any(len(a.holes) != len(b.holes)
                           for a, b in zip(actual_region.components, expected_region.components, strict=True))):
                raise ValueError("Admitted RGB region topology differs from original calibrated mask")
            for actual_part, expected_part in zip(actual_region.components, expected_region.components, strict=True):
                for actual_ring, expected_ring in zip((actual_part.exterior, *actual_part.holes),
                        (expected_part.exterior, *expected_part.holes), strict=True):
                    if len(actual_ring) != len(expected_ring) or any(
                            math.dist(a, b) > 1e-12 for a, b in zip(actual_ring, expected_ring, strict=True)):
                        raise ValueError("Admitted RGB region differs from original calibrated pixel cells")
            return
        if "outline_mm" not in geometry:
            region = as_region(geometry["region_mm"])
            reason = ("holes_or_multiple_boundaries" if len(region.components) != 1
                      or region.components[0].holes else "nonconvex_geometry_requires_region")
            raise ValueError("Original RGB geometry is unsupported by legacy outline schema: "+reason)
        expected = transform(convex_polygon(geometry["outline_mm"]), rotation_deg=-rotation)
        actual = convex_polygon(observation["outline_mm"])
        # Cyclic starting vertex is immaterial; every exact boundary vertex and
        # its calibrated position must still match. No fitting or rescaling.
        if len(actual) != len(expected) or not any(
            all(math.dist(a, b) <= 1e-12 for a, b in zip(actual, expected[k:] + expected[:k], strict=True))
            for k in range(len(expected))
        ):
            raise ValueError("Admitted RGB outline differs from original calibrated mask")
    except (KeyError, TypeError, AttributeError, OSError, IndexError) as exc:
        raise ValueError("Missing or malformed RGB outline binding") from exc


def validate_rgb_capture_binding(store, flake: dict, observation: dict) -> None:
    """Join mutable inventory to its original flow-produced capture receipt."""
    try:
        event_id = observation["recognition_event_id"]
        if type(event_id) is not int or event_id <= 0:
            raise ValueError("RGB evidence requires its original capture event")
        run = store.get_run(observation["job_id"])
        if not run or run["flow"] not in ("search_chip", "revisit_chip") or run["status"] != "succeeded":
            raise ValueError("RGB source evidence requires a completed search or revisit")
        event = store._x("SELECT * FROM event WHERE id=?", (event_id,)).fetchone()
        completion = store._x("SELECT MIN(id) AS id FROM event WHERE run_id=? AND kind='job.status' "
                              "AND json_extract(payload_json,'$.status')='succeeded'",
                              (observation["job_id"],)).fetchone()
        if (not event or event["kind"] != "microscope.rgb_recognition"
                or event["run_id"] != observation["job_id"] or not completion
                or completion["id"] is None or event_id >= completion["id"]):
            raise ValueError("RGB capture receipt is foreign, missing or later than job completion")
        receipt = json.loads(event["payload_json"])
        if (type(receipt.get("version")) is not int or receipt.get("version") != 1 or receipt.get("flake_id") != flake["id"]
                or receipt.get("source_chip_id") != flake["location"]
                or receipt.get("material") != flake["props"].get("material")
                or receipt.get("recognition_chain") != observation["recognition_chain"]
                or receipt.get("chip_rotation_deg") != observation["chip_rotation_deg"]):
            raise ValueError("RGB source/calibration/rotation differs from original capture receipt")
        _validate_rgb_geometry_budget(observation)
        if "region_mm" in observation:
            keys = ["region_mm", "boundary_uncertainty_mm", "boundary_uncertainty_provenance"]
            if "owner_error_budget" in observation:
                keys += ["owner_error_budget", "source_anchor20", "geometry_policy"]
            if any(receipt.get(key) != observation[key] for key in keys):
                raise ValueError("RGB region/boundary budget differs from its original capture receipt")
        linked = {}
        for key, kind in (("capture_event_id", "device.command"), ("chip_rotation_event_id", "step.verified")):
            linked_id = receipt[key]
            if type(linked_id) is not int or not 0 < linked_id < event_id:
                raise ValueError("Original RGB capture/rotation link is absent or out of order")
            row = store._x("SELECT * FROM event WHERE id=?", (linked_id,)).fetchone()
            if not row or row["kind"] != kind or row["run_id"] != observation["job_id"]:
                raise ValueError("Original RGB capture/rotation event belongs to another operation")
            linked[key] = json.loads(row["payload_json"])
        capture, rotation = linked["capture_event_id"], linked["chip_rotation_event_id"]
        if (receipt["chip_rotation_event_id"] >= receipt["capture_event_id"]
                or capture.get("device") != "microscope" or capture.get("command") != "capture"
                or rotation.get("name") != "measure_chip_rotation" or rotation.get("verdict") != "ok"
                or rotation.get("measurements", {}).get("rotation_deg") != receipt["chip_rotation_deg"]):
            raise ValueError("RGB capture does not follow its original observed rotation")
        if "owner_error_budget" in observation:
            from qumus_simlab.records.geometry import source_evidence_current

            anchor = observation["source_anchor20"]
            identifier = anchor["stage_readback_event_id"]
            readback = store._x("SELECT kind,run_id,payload_json FROM event WHERE id=?", (identifier,)).fetchone()
            if (type(identifier) is not int or not receipt["chip_rotation_event_id"] < identifier < receipt["capture_event_id"]
                    or not readback or readback["kind"] != "microscope.source_stage_readback"
                    or readback["run_id"] != observation["job_id"]):
                raise ValueError("Original20 source anchor lacks its same-job public stage readback")
            public = json.loads(readback["payload_json"])
            if (public["source_chip_id"] != flake["location"] or public["flake_id"] != flake["id"]
                    or public["objective"] != 20 or public["stage_xy"] != anchor["capture_stage_xy"]
                    or public["stage_xy"] != [a+b for a, b in zip(public["coarse_xy"], public["fine_xy"], strict=True)]
                    or anchor["capture_event_id"] != receipt["capture_event_id"]
                    or flake["props"].get("stage_xy") != anchor["stage_xy"]
                    or not source_evidence_current(store, flake, observation)):
                raise ValueError("Original20 source anchor differs from current public readback or inventory")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Missing or malformed original RGB capture receipt") from exc
