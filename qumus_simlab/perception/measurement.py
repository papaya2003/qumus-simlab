"""One pure authority for conditional SIM RGB boundary qualification.

Saved image/mask/region identity is independent of measurement accuracy. The
current native capture profile has no qualified boundary-error method. Input
provenance, a positive number and empirical fixture maxima cannot grant it one.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _measurement_profile():
    """Code-owned authority; never loaded from a sample, event or caller input."""
    return {"version": "rgb-boundary-measurement-profile-v1",
        "profile_id": "v1-reference-material-boundary-v1", "status": "unqualified",
        "capture_profile": V1_MATERIAL_CAPTURE_PROFILE,
        "reason": "No qualified continuous-boundary error domain; fixed fixture errors are evaluator measurements only"}


def boundary_measurement_status():
    """A serializable disposition for raw recognition evidence, without a bound."""
    profile = _measurement_profile()
    return {**profile, "profile_sha256": _digest(profile)}


def owner_error_budget_policy():
    """Owner-selected inclusive engineering acceptance, not metrology.

    The one-micrometre endpoint tolerance already includes measurement,
    registration and positioning errors. It is applied once, alongside the
    one-degree rotation acceptance; it is not an extra physical boundary bound.
    The independent scientific measurement profile remains unqualified.
    """
    policy = {"version": "owner-inclusive-error-budget-v1", "source": "owner_instruction",
        "status": "engineering_acceptance_assumption", "translation_tolerance_um": 1.,
        "rotation_tolerance_deg": 1., "inclusive_error_budget_mm": .001,
        "includes": ["measurement", "registration", "positioning"],
        "application": "Apply the inclusive pose acceptance once; no additional sampling or registration deduction",
        "scientific_qualification": "not_calibrated_or_certified"}
    return {**policy, "policy_sha256": _digest(policy)}


def validate_owner_error_budget(observation):
    """Validate the explicit owner policy, independently of evidence identity."""
    try:
        bound = observation["boundary_uncertainty_mm"]
        if (observation["owner_error_budget"] != owner_error_budget_policy()
                or type(bound) not in (int, float) or not math.isfinite(bound) or bound != .001):
            raise ValueError("Observation differs from the owner's inclusive engineering error budget")
        return .001
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Observation lacks the owner's explicit inclusive engineering error budget") from exc


def _capture_domain(chain):
    """Public capture/recognizer inputs only; no material truth or geometry."""
    return {"objective": chain["image"]["objective"],
        "image_size_px": chain["image"].get("size_px"), "material": chain["material"],
        "calibration_sha256": _digest(chain["calibration"]),
        "material_policy_sha256": chain.get("material_policy_sha256"),
        "original_mask_producer": chain.get("material_policy", {}).get("original_mask_producer")}


def validate_qualified_boundary(observation, bound):
    """Bind a budget to the code-owned qualified method and exact public domain.

    There is currently no qualified production profile. The explicit qualified
    branch allows isolated analytic test doubles to check provenance and budget
    binding; it does not create a runtime registration or caller approval API.
    """
    try:
        profile = _measurement_profile()
        provenance = observation["boundary_uncertainty_provenance"]
        if (provenance.get("profile_id") != profile["profile_id"]
                or provenance.get("profile_sha256") != _digest(profile)):
            raise ValueError("RGB boundary measurement profile is unknown or changed")
        if profile.get("status") != "qualified_conditional_sim":
            raise ValueError("RGB boundary measurement profile is unqualified")
        chain = observation["recognition_chain"]
        expected = profile["boundary_uncertainty_mm"]
        domain = _capture_domain(chain)
        if (type(expected) not in (int, float) or not math.isfinite(expected) or expected <= 0
                or bound != expected or not isinstance(profile.get("method_id"), str) or not profile["method_id"]
                or not isinstance(profile.get("qualification_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", profile["qualification_sha256"]) is None
                or not isinstance(profile.get("algorithm_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", profile["algorithm_sha256"]) is None
                or provenance.get("method_id") != profile["method_id"]
                or provenance.get("qualification_sha256") != profile["qualification_sha256"]
                or chain.get("algorithm_sha256") != profile["algorithm_sha256"]
                or chain["calibration"].get("material_capture_profile") != profile["capture_profile"]
                or provenance.get("capture_profile") != profile["capture_profile"]
                or domain != profile["domain"] or provenance.get("domain_sha256") != _digest(domain)):
            raise ValueError("RGB boundary budget differs from its qualified method, domain, algorithm or capture")
        return float(expected)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("RGB boundary measurement qualification is missing or malformed") from exc


def measure_recognition_boundary(chain, *, output_dir=None):
    """Measure saved RGB separately from its immutable original v1 mask chain.

    This explicit evidence operation does not replace recognition geometry or
    grant source/design admission. The result uses corrected exposure coordinates;
    historical integer-index geometry and all three original artifacts stay intact.
    """
    import io
    from copy import deepcopy

    import numpy as np
    from PIL import Image

    from qumus_simlab.core.regions import region_dict, transform_region
    from qumus_simlab.core.run_paths import resolve_run_path
    from qumus_simlab.perception.boundary import refine_rgb_boundary
    from qumus_simlab.perception.rgb_material import component_geometry
    from qumus_simlab.records.recognition import validate_recognition_chain

    source = deepcopy(chain)
    validate_recognition_chain(source, image_path=source["image"]["path"],
                               objective=source["image"]["objective"])
    image_path = resolve_run_path(source["image"]["path"])
    image_bytes = image_path.read_bytes()
    seed_bytes = resolve_run_path(source["geometry_mask"]["path"]).read_bytes()
    if (hashlib.sha256(image_bytes).hexdigest() != source["image"]["sha256"]
            or hashlib.sha256(seed_bytes).hexdigest() != source["geometry_mask"]["sha256"]):
        raise ValueError("Original RGB/mask bytes changed before measurement decoding")
    with Image.open(io.BytesIO(image_bytes)) as image:
        pixels = np.array(image.convert("RGB"))
    with Image.open(io.BytesIO(seed_bytes)) as image:
        seed = np.asarray(image) != 0
    pixels.setflags(write=False)
    seed.setflags(write=False)
    result = refine_rgb_boundary(pixels, seed, deepcopy(source["calibration"]),
        material=source["material"], objective=source["image"]["objective"],
        recognition_policy=deepcopy(source["material_policy"]))
    if (result.get("status") not in ("measured", "unknown")
            or result.get("physical_qualification") != "unqualified"):
        raise ValueError("Raster measurement cannot self-qualify a physical boundary")
    # Recheck bytes after computation as well as before; never publish a receipt
    # linking a measurement to an original that changed while it was running.
    validate_recognition_chain(source, image_path=source["image"]["path"],
                               objective=source["image"]["objective"])
    maps = {}
    summary = {key: value for key, value in result.items()
               if key not in ("derived_mask", "uncertain_mask", "edge_mask")}
    for key in ("derived_mask", "uncertain_mask", "edge_mask"):
        value = result.get(key)
        if value is None and key == "derived_mask" and result["status"] == "unknown":
            continue
        if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != seed.shape:
            raise ValueError("Boundary measurement returned a malformed pixel map: "+key)
        maps[key] = value
    if result["status"] == "measured":
        geometry = component_geometry(maps["derived_mask"], source["calibration"])
        if geometry["status"] != "supported":
            summary.update(status="unknown", reason="Measured raster has unsupported exact geometry: "
                           +str(geometry["unsupported_reason"]))
        else:
            sx, sy = source["calibration"]["mm_per_pixel"]
            centre = [geometry["dx_mm"]+sx/2, geometry["dy_mm"]-sy/2]
            summary.update(region_exposure_mm=region_dict(transform_region(geometry["region_mm"], xy=centre)),
                centroid_exposure_mm=centre, area_um2=geometry["area_um2"])
    destination = Path(output_dir) if output_dir is not None else image_path.parent
    destination.mkdir(parents=True, exist_ok=True)

    def save_immutable(payload, suffix):
        digest = hashlib.sha256(payload).hexdigest()
        path = destination/f"{image_path.stem}-boundary-{digest[:16]}.{suffix}"
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Immutable RGB boundary artifact changed")
        if not path.exists():
            path.write_bytes(payload)
        return {"path": str(path.resolve()), "sha256": digest}

    artifacts = {}
    for key, array in maps.items():
        encoded = io.BytesIO()
        Image.fromarray(array.astype(np.uint8)*255).save(encoded, format="PNG")
        artifacts[key] = save_immutable(encoded.getvalue(), "png")
    import PIL
    import shapely

    wrapper_path = Path(__file__).resolve()
    wrapper_dependencies = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
        "measurement.py": wrapper_path, "rgb_material.py": wrapper_path.with_name("rgb_material.py"),
        "regions.py": wrapper_path.parent.parent/"core/regions.py"}.items()}
    wrapper_dependencies.update(pillow_version=PIL.__version__, shapely_version=shapely.__version__)
    receipt = {"version": "rgb-boundary-measurement-receipt-v1",
        "original_recognition_chain": source, "recognition_chain_sha256": _digest(source),
        "wrapper_dependencies": wrapper_dependencies,
        "qualification": boundary_measurement_status(), "measurement": summary, "artifacts": artifacts}
    encoded = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode()+b"\n"
    return {**receipt, "receipt_artifact": save_immutable(encoded, "json")}


def validate_measurement_receipt(receipt):
    """Reproduce a saved raster measurement without trusting claimed geometry.

    A hash of a caller-created receipt is not an observation authority. This
    operation checks the actual original image/seed and independently recomputes
    the unchanged RGB estimator, all three output maps and the absolute region.
    It establishes reproducible image evidence, never physical qualification.
    """
    import io
    from copy import deepcopy

    import numpy as np
    from PIL import Image

    from qumus_simlab.core.regions import region_dict, transform_region
    from qumus_simlab.core.run_paths import resolve_run_path
    from qumus_simlab.perception.boundary import refine_rgb_boundary
    from qumus_simlab.perception.rgb_material import component_geometry
    from qumus_simlab.records.recognition import validate_recognition_chain

    try:
        value = deepcopy(receipt)
        artifact = value.pop("receipt_artifact")
        payload = resolve_run_path(artifact["path"]).read_bytes()
        if (hashlib.sha256(payload).hexdigest() != artifact["sha256"]
                or json.loads(payload) != value
                or value["version"] != "rgb-boundary-measurement-receipt-v1"):
            raise ValueError("Measurement receipt bytes or identity changed")
        chain = value["original_recognition_chain"]
        validate_recognition_chain(chain, image_path=chain["image"]["path"],
                                   objective=chain["image"]["objective"])
        if value["recognition_chain_sha256"] != _digest(chain):
            raise ValueError("Measurement reference recognition chain changed")
        with Image.open(resolve_run_path(chain["image"]["path"])) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(resolve_run_path(chain["geometry_mask"]["path"])) as image:
            seed = np.asarray(image) != 0
        rgb.setflags(write=False)
        seed.setflags(write=False)
        reproduced = refine_rgb_boundary(rgb, seed, deepcopy(chain["calibration"]),
            material=chain["material"], objective=chain["image"]["objective"],
            recognition_policy=deepcopy(chain["material_policy"]))
        maps = {}
        for key in ("derived_mask", "uncertain_mask", "edge_mask"):
            expected = reproduced[key]
            if expected is None and key == "derived_mask" and reproduced["status"] == "unknown":
                if key in value["artifacts"]:
                    raise ValueError("Unknown measurement acquired a fabricated derived mask")
                continue
            stored = value["artifacts"][key]
            raw = resolve_run_path(stored["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != stored["sha256"]:
                raise ValueError("Measurement pixel artifact bytes changed: "+key)
            with Image.open(io.BytesIO(raw)) as image:
                array = np.asarray(image).copy()
            if (array.ndim != 2 or array.shape != seed.shape
                    or not np.isin(array, (0, 255)).all()
                    or not np.array_equal(array != 0, expected)):
                raise ValueError("Measurement pixels do not reproduce original RGB: "+key)
            maps[key] = array != 0
        expected_summary = {key: item for key, item in reproduced.items()
                            if key not in ("derived_mask", "uncertain_mask", "edge_mask")}
        if reproduced["status"] == "measured":
            geometry = component_geometry(maps["derived_mask"], chain["calibration"])
            if geometry["status"] != "supported":
                expected_summary.update(status="unknown", reason="Measured raster has unsupported exact geometry: "
                                        +str(geometry["unsupported_reason"]))
            else:
                sx, sy = chain["calibration"]["mm_per_pixel"]
                centre = [geometry["dx_mm"]+sx/2, geometry["dy_mm"]-sy/2]
                expected_summary.update(
                    region_exposure_mm=region_dict(transform_region(geometry["region_mm"], xy=centre)),
                    centroid_exposure_mm=centre, area_um2=geometry["area_um2"])
        if value["measurement"] != expected_summary:
            raise ValueError("Measurement method, status, region or scientific metadata does not reproduce")
        # Recheck the original chain after computation; concurrent changes must
        # not produce an apparently valid association to different input bytes.
        validate_recognition_chain(chain, image_path=chain["image"]["path"],
                                   objective=chain["image"]["objective"])
        return {**value, "receipt_artifact": deepcopy(artifact)}
    except (KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("Missing or malformed immutable RGB measurement receipt") from exc
