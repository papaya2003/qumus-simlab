"""Compose image material observations with the existing optical owner.

Material methods never fall back to simulator annotations. Public capture
calibration, objective and focal-plane readbacks are legitimate inputs; object
classes, layers, outlines and attachment state in Frame.meta are not.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from qumus_simlab.core.optical_profiles import CAMERA_RESPONSE_PROFILE, SYNTHETIC_SUBSTRATE_RGB
from qumus_simlab.core.regions import region_dict, transform_region
from qumus_simlab.perception.rgb_material import BACKEND, RGBMaterialAnalyzer

RGB_DISCOVERY_PROFILE = "rgb-substrate-compatible-visible-cells-v1"
RGB_CARRIED_PROFILE = "rgb-carried-material-observation-v1"
ASSOCIATION_PROFILE = "rgb-whole-region-association-v1"
VISIBLE_CELL_SCOPE = "Conditional visible pixel support only; invisible/subresolution material is unknown"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _pixels(frame):
    raw = frame.data
    if raw is None:
        path = frame.saved_path()
        if not path:
            raise ValueError("RGB observation lacks captured image bytes")
        raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    calibration = frame.meta.get("calibration", {})
    if calibration.get("image_size_px") != [pixels.shape[1], pixels.shape[0]]:
        raise ValueError("RGB observation/calibration dimensions differ")
    return pixels, hashlib.sha256(raw).hexdigest()


def _runs(mask, roi):
    """Lossless [row, start, stop) runs of every selected original pixel cell."""
    x0, y0, x1, y1 = roi
    runs = []
    for y in range(y0, y1):
        selected = np.flatnonzero(mask[y, x0:x1])+x0
        if not len(selected):
            continue
        cuts = np.flatnonzero(np.diff(selected) != 1)+1
        for part in np.split(selected, cuts):
            runs.append([y, int(part[0]), int(part[-1])+1])
    return runs


def rgb_discovery_policy():
    """Fixed public support declaration, independently checked by its consumer."""
    return {"version": RGB_DISCOVERY_PROFILE, "camera_response": CAMERA_RESPONSE_PROFILE,
        "substrate_rgb": list(SYNTHETIC_SUBSTRATE_RGB), "support": "exact-achromatic-noise-compatible-cells",
        "supported_roi_px": [8, 30, 472, 288], "minimum_chip_plane_quality": 9,
        "unknown_rule": "all_noncompatible_or_saturated_cells", "run_stop": "exclusive",
        "scope": VISIBLE_CELL_SCOPE}


def rgb_discovery_obstacles(frame, *, plane_quality=None):
    """Conservative native5 visible-cell support, conditional on a fixed SIM law.

    A background-compatible pixel satisfies the *same achromatic noise* in all
    three channels, rather than three unrelated colour tolerances. Everything
    else is unknown, including off-chip surface, residue, material and metal.
    No class count, contour or complete-flake claim is made. Subresolution and
    invisible material are outside this explicitly conditional support.
    """
    pixels, image_sha = _pixels(frame)
    height, width = pixels.shape[:2]
    calibration = deepcopy(frame.meta.get("calibration", {}))
    result = {"profile": RGB_DISCOVERY_PROFILE, "source": "synthetic",
        "status": "unsupported", "conditional_visible_completeness": False,
        "image_sha256": image_sha, "captured_at": frame.timestamp,
        "calibration": calibration, "obstacles": [],
        "supported_roi_px": [0, 0, 0, 0], "unknown_runs_px": [],
        "unknown_outside_roi": True, "scope": VISIBLE_CELL_SCOPE}
    photometry = calibration.get("photometry_profile", {})
    noise = calibration.get("observation_condition", {}).get("observation_noise_scale")
    supported = (calibration.get("source") == "synthetic"
        and frame.meta.get("objective") == 5 and frame.meta.get("focus_plane") == "chip"
        and frame.meta.get("fov_mm") == [2.6, 1.72]
        and [width, height] == [480, 320]
        and calibration.get("mm_per_pixel") == [2.6/480, 1.72/320]
        and calibration.get("u_axis") == "+world_x" and calibration.get("v_axis") == "-world_y"
        and photometry.get("source") == "synthetic"
        and photometry.get("camera_response") == CAMERA_RESPONSE_PROFILE
        and photometry.get("chip_surface") == "synthetic-chip-footprint-v2"
        and type(noise) in (int, float) and math.isfinite(noise) and 0 <= noise <= 1
        and type(plane_quality) in (int, float) and math.isfinite(plane_quality) and plane_quality >= 9
        and not frame.meta.get("replayed_from"))
    if not supported:
        result["reason"] = "Unsupported native5 camera, focus, calibration or observation condition"
        return result
    yy, xx = np.mgrid[:height, :width]
    gain = ((1-.14*((xx-width/2)/(width/2))**2)
            *(1-.14*((yy-height/2)/(height/2))**2))
    baseline = np.trunc(gain[:, :, None]*np.array(SYNTHETIC_SUBSTRATE_RGB)).astype(np.int16)
    delta = pixels.astype(np.int16)-baseline
    permitted_noise = np.array([round(i*noise) for i in range(-3, 4)], dtype=np.int16)
    bare = ((delta[:, :, 0] == delta[:, :, 1]) & (delta[:, :, 1] == delta[:, :, 2])
            & np.isin(delta[:, :, 0], permitted_noise))
    saturated = (pixels.min(axis=2) == 0) | (pixels.max(axis=2) == 255)
    unknown = ~bare | saturated
    roi = [8, 30, width-8, height-32]
    policy = rgb_discovery_policy()
    result.update(status="supported", conditional_visible_completeness=True,
        supported_roi_px=roi, unknown_runs_px=_runs(unknown, roi),
        unknown_mask_sha256=hashlib.sha256(np.packbits(unknown).tobytes()).hexdigest(),
        boundary_cell_radius_mm=math.hypot(*calibration["mm_per_pixel"])/2,
        observation_policy=policy, observation_policy_sha256=_digest(policy))
    return result


def load_pixels(frame):
    from qumus_simlab.perception.tracking import load_pixels as load

    return load(frame)


def pixel_offset_mm(frame, xy_px, shape):
    """Image point (continuous px, origin at the top-left corner) to mm from the view centre."""
    calibration = frame.meta.get("calibration") or {}
    height, width = shape[:2]
    scale = calibration.get("mm_per_pixel")
    if not scale:
        fw, fh = frame.meta["fov_mm"]
        scale = [fw/width, fh/height]
    if (calibration.get("u_axis", "+world_x"), calibration.get("v_axis", "-world_y")) != ("+world_x", "-world_y"):
        raise ValueError("Pixel offsets need the +world_x/-world_y camera axes")
    x, y = xy_px
    return (x+.5-width/2)*scale[0], -(y+.5-height/2)*scale[1]


def detection_region(detection):
    """The complete measured body in this exposure's microscope XY frame."""
    if not detection.meta.get("outline_complete") or "region_mm" not in detection.meta:
        raise ValueError("Whole-body observation needs complete original pixel regions")
    return transform_region(detection.meta["region_mm"], xy=(detection.dx_mm, detection.dy_mm))


def _boundary_distance_interval(first, second, spacing_mm):
    """Numerical full-ring discrepancy, not an observation-error budget."""
    import shapely

    from qumus_simlab.core.regions import _geos, as_region

    regions = [as_region(first), as_region(second)]
    geometries = [_geos(region) for region in regions]
    if any(geometry.is_empty for geometry in geometries):
        raise ValueError("Image registration needs two nonempty measured bodies")
    maximum, total = 0., 0
    for source, destination in ((0, 1), (1, 0)):
        samples = []
        for component in regions[source].components:
            for ring in (component.exterior, *component.holes):
                for start, stop in zip(ring, ring[1:]+ring[:1], strict=True):
                    count = max(1, math.ceil(math.dist(start, stop)/spacing_mm))
                    total += count
                    if total > 1_000_000:
                        raise ValueError("Measured boundary exceeds the bounded registration work limit")
                    fractions = np.arange(count)/count
                    samples.append(np.asarray(start)+(np.asarray(stop)-start)*fractions[:, None])
        distances = shapely.distance(shapely.points(np.concatenate(samples)), geometries[destination].boundary)
        maximum = max(maximum, float(np.max(distances)))
    guard = 32*math.ulp(max(1., maximum))
    return [max(0., maximum-guard), maximum+spacing_mm/2+guard]


def register_rgb_measurements(reference20_receipt, candidate100_receipts, *, rotation_deg=0):
    """Record actual-image cross-scale registration diagnostics, fail closed.

    This estimator aligns observed support extrema, not an intended device or
    truth outline. The reported residual includes every measured boundary ring.
    It cannot establish a physical transform bound from unqualified rasters or
    an unbounded commanded-stage prior. Its status therefore stays unknown;
    root-owned admission must separately establish the factory/optical domain,
    candidate visibility and a bounded registration authority.
    """
    from qumus_simlab.core.regions import _geos, as_region
    from qumus_simlab.perception.measurement import validate_measurement_receipt

    if type(rotation_deg) not in (int, float) or not math.isfinite(rotation_deg):
        raise ValueError("Cross-scale image registration needs a finite explicit rotation")
    reference = validate_measurement_receipt(reference20_receipt)
    source_chain = reference["original_recognition_chain"]
    if (source_chain["image"]["objective"] != 20
            or reference["measurement"]["status"] != "measured"):
        raise ValueError("Cross-scale reference must be an actual measured original20 image")
    source_region = reference["measurement"]["region_exposure_mm"]
    rotated = transform_region(source_region, rotation_deg=rotation_deg)
    source_bounds = _geos(rotated).bounds
    result = {"version": "rgb-cross-scale-registration-diagnostic-v1", "status": "unknown",
        "physical_qualification": "unqualified", "registration_uncertainty_mm": None,
        "reference20_chain_sha256": reference["recognition_chain_sha256"],
        "reference20_receipt_sha256": reference["receipt_artifact"]["sha256"],
        "reference20_image_sha256": source_chain["image"]["sha256"],
        "material": source_chain["material"], "rotation_deg": rotation_deg,
        "rotation_authority": "explicit input; not independently qualified by this diagnostic",
        "translation_estimator": "corresponding observed bounding-support extrema midpoint",
        "candidates": [], "unsupported_candidates": [],
        "reason": "Physical domain, candidate visibility and registration bound are not qualified"}
    for index, supplied in enumerate(candidate100_receipts):
        try:
            current = validate_measurement_receipt(supplied)
            chain = current["original_recognition_chain"]
            if chain["material"] != source_chain["material"]:
                continue
            if chain["image"]["objective"] != 100 or current["measurement"]["status"] != "measured":
                raise ValueError("Current same-material image is not a measured native100 body")
            current_region = current["measurement"]["region_exposure_mm"]
            current_bounds = _geos(as_region(current_region)).bounds
            extrema = [[current_bounds[axis]-source_bounds[axis],
                        current_bounds[axis+2]-source_bounds[axis+2]] for axis in (0, 1)]
            translation = [sum(values)/2 for values in extrema]
            aligned = transform_region(rotated, xy=translation)
            spacing = min(*source_chain["calibration"]["mm_per_pixel"],
                          *chain["calibration"]["mm_per_pixel"])/2
            discrepancy = _boundary_distance_interval(aligned, current_region, spacing)
            result["candidates"].append({"candidate_index": index,
                "native100_receipt_sha256": current["receipt_artifact"]["sha256"],
                "native100_chain_sha256": current["recognition_chain_sha256"],
                "native100_image_sha256": chain["image"]["sha256"],
                "translation_mm": translation, "observed_extrema_translation_mm": extrema,
                "boundary_discrepancy_interval_mm": discrepancy,
                "numerical_boundary_spacing_mm": spacing,
                "registration_uncertainty_mm": None,
                "native100_region_exposure_mm": current_region})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result["unsupported_candidates"].append({"candidate_index": index, "reason": str(exc)})
    ordered = sorted(result["candidates"], key=lambda item: item["boundary_discrepancy_interval_mm"][0])
    if not ordered:
        result["image_ranking"] = "no_supported_same_material_candidate"
    elif len(ordered) > 1 and (ordered[1]["boundary_discrepancy_interval_mm"][0]
                              <= ordered[0]["boundary_discrepancy_interval_mm"][1]):
        result["image_ranking"] = "ambiguous_overlapping_numerical_residual_intervals"
    else:
        result["image_ranking"] = "one_best_observed_residual_not_source_identity_proof"
    result["diagnostic_sha256"] = _digest(result)
    return result


def native100_source_anchor(native100_measurement_receipt, original20_recognition_chain, *,
                            chip_rotation_evidence, registration_evidence=None):
    """Bind a measured100 centroid frame without granting physical admission.

    Original event dictionaries establish structural/image binding here. The
    records authority must independently join those IDs and payloads to the
    original RunStore events before trusting their provenance. Registration
    evidence concerns the original20 identity only; it never moves100 geometry.
    """
    from qumus_simlab.perception.measurement import validate_measurement_receipt
    from qumus_simlab.records.recognition import validate_recognition_chain

    measured = validate_measurement_receipt(native100_measurement_receipt)
    current = measured["original_recognition_chain"]
    summary = measured["measurement"]
    if current["image"]["objective"] != 100 or summary["status"] != "measured":
        raise ValueError("Source anchor requires an actual measured native100 image")
    original = deepcopy(original20_recognition_chain)
    validate_recognition_chain(original, image_path=original["image"]["path"], objective=20)
    if original["material"] != current["material"] or current["material"] not in ("Gr", "hBN"):
        raise ValueError("Original20 identity and native100 requested material differ")
    try:
        rotation = deepcopy(chip_rotation_evidence)
        observed, captured = rotation["rotation_event"], rotation["capture_event"]
        step, command = observed["payload"], captured["payload"]
        angle = step["measurements"]["rotation_deg"]
        if (rotation["version"] != "observed-chip-rotation-v1"
                or not isinstance(rotation["source_chip_id"], str) or not rotation["source_chip_id"]
                or rotation["native100_image_sha256"] != current["image"]["sha256"]
                or type(observed["id"]) is not int or type(captured["id"]) is not int
                or not 0 < observed["id"] < captured["id"]
                or not isinstance(observed["run_id"], str) or not observed["run_id"]
                or observed["run_id"] != captured["run_id"]
                or observed["kind"] != "step.verified" or captured["kind"] != "device.command"
                or step["name"] != "measure_chip_rotation" or step["verdict"] != "ok"
                or command["device"] != "microscope" or command["command"] != "capture"
                or type(angle) not in (int, float) or not math.isfinite(angle)):
            raise ValueError("Original observed chip rotation/capture binding is invalid")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Source anchor requires original observed chip rotation and capture events") from exc
    reference_digest = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":"),
                                                 allow_nan=False).encode()).hexdigest()
    registration = {"status": "unknown", "physical_qualification": "unqualified",
                    "registration_uncertainty_mm": None, "reason": "No cross-scale registration evidence"}
    if registration_evidence is not None:
        supplied = deepcopy(registration_evidence)
        digest = supplied.pop("diagnostic_sha256", None)
        matches = [candidate for candidate in supplied.get("candidates", [])
                   if candidate.get("native100_receipt_sha256") == measured["receipt_artifact"]["sha256"]
                   and candidate.get("native100_chain_sha256") == measured["recognition_chain_sha256"]
                   and candidate.get("native100_image_sha256") == current["image"]["sha256"]]
        if (supplied.get("version") != "rgb-cross-scale-registration-diagnostic-v1"
                or supplied.get("status") != "unknown" or supplied.get("physical_qualification") != "unqualified"
                or supplied.get("registration_uncertainty_mm") is not None
                or digest != _digest(supplied) or supplied.get("reference20_chain_sha256") != reference_digest
                or supplied.get("reference20_image_sha256") != original["image"]["sha256"]
                or supplied.get("material") != current["material"] or not matches):
            raise ValueError("Cross-scale registration evidence belongs to different images or identity")
        registration.update(diagnostic_sha256=digest, image_ranking=supplied.get("image_ranking"),
                            matching_candidate_indices=[item["candidate_index"] for item in matches],
                            reason=supplied.get("reason"))
    centre = summary["centroid_exposure_mm"]
    if (not isinstance(centre, (list, tuple)) or len(centre) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in centre)):
        raise ValueError("Native100 measured centroid is invalid")
    local = transform_region(summary["region_exposure_mm"], xy=(-centre[0], -centre[1]))
    local = transform_region(local, rotation_deg=-angle)
    result = {"version": "rgb-native100-source-anchor-v1", "status": "anchored_unqualified",
        "admission": "blocked", "physical_qualification": "unqualified", "units": "mm",
        "frame": "native100_derived_centroid_chip_axes_v1", "region_mm": region_dict(local),
        "source_chip_id": rotation["source_chip_id"], "material": current["material"],
        "anchor_exposure_mm": list(centre), "anchor_definition": "RGB-derived pixel-cell area centroid in corrected Canvas coordinates",
        "chip_rotation_deg": angle, "chip_rotation_evidence": rotation,
        "event_validation_scope": "Structural/image binding only; original RunStore event authority is required separately",
        "native100_image": deepcopy(current["image"]), "calibration": deepcopy(current["calibration"]),
        "calibration_sha256": hashlib.sha256(json.dumps(current["calibration"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "native100_measurement_receipt": deepcopy(measured["receipt_artifact"]),
        "native100_chain_sha256": measured["recognition_chain_sha256"],
        "original20_identity": {"recognition_chain_sha256": reference_digest,
            "image": deepcopy(original["image"]), "material": original["material"],
            "selected_label_ids": deepcopy(original["selected_label_ids"]),
            "same_source_status": "unqualified", "scope": "Immutable original20 recognition identity; not geometry or registration authority"},
        "registration": registration, "boundary_uncertainty_mm": None,
        "cross_scale_registration_uncertainty_mm": None,
        "source_hashes": {"anchor_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "measurement_wrapper_dependencies": deepcopy(measured["wrapper_dependencies"]),
            "boundary_algorithm_sha256": summary["algorithm_sha256"],
            "boundary_dependencies": deepcopy(summary["provenance"].get("dependencies", {}))}}
    result["anchor_receipt_sha256"] = _digest(result)
    return result


def associate_source_observations(source_observation, detections, *, material,
        source_to_observation_xy_mm, source_to_observation_rotation_deg,
        registration_uncertainty_mm, observation_complete=False):
    """Associate an original exact body with later independently measured RGB.

    The transform must come from a separate captured-image registration; this
    function does not fit a favourable pose to candidate contours. Boundary and
    registration uncertainty are explicit, additive and never pose tolerances.
    ``absent`` reports observable source absence only; it does not mean pickup.
    """
    from qumus_simlab.core.regions import _geos, region_bidirectional_within
    from qumus_simlab.records.recognition import validate_boundary_uncertainty

    if material not in ("Gr", "hBN"):
        raise ValueError("Source association needs a supported public material class")
    if (type(registration_uncertainty_mm) not in (int, float)
            or not math.isfinite(registration_uncertainty_mm) or registration_uncertainty_mm < 0):
        raise ValueError("Source association needs a finite nonnegative registration uncertainty")
    if type(observation_complete) is not bool:
        raise ValueError("Source association needs an explicit observation-completeness disposition")
    result = {"profile": ASSOCIATION_PROFILE, "status": "unknown", "material": material,
        "registration_uncertainty_mm": registration_uncertainty_mm,
        "source_to_observation_xy_mm": list(source_to_observation_xy_mm),
        "source_to_observation_rotation_deg": source_to_observation_rotation_deg,
        "matches": [], "scope": "Whole-body pixel association under explicit SIM measurement bounds; no attachment inference"}
    try:
        source_bound = validate_boundary_uncertainty(source_observation)
        source_chain = source_observation["recognition_chain"]
        if source_chain.get("material") != material or source_chain["image"].get("objective") != 20:
            raise ValueError("Original source material/native20 identity differs")
        expected = transform_region(source_observation["region_mm"],
            xy=source_to_observation_xy_mm, rotation_deg=source_to_observation_rotation_deg)
        if _geos(expected).is_empty:
            raise ValueError("Original source body is empty")
        result.update(source_image_sha256=source_observation["recognition_chain"]["image"]["sha256"],
            source_boundary_uncertainty_mm=source_bound)
    except (KeyError, ValueError, TypeError) as exc:
        return {**result, "reason": "Original source observation is unsupported: "+str(exc)}
    unsupported, confounders = [], []
    for index, detection in enumerate(detections):
        if detection.material != material:
            continue
        try:
            if detection.meta.get("carried_observation_supported") is False:
                raise ValueError("Cross-material carried pixels are ambiguous")
            current_bound = validate_boundary_uncertainty(detection.meta)
            if detection.meta["recognition_chain"].get("material") != material:
                raise ValueError("Current RGB material policy differs")
            current = detection_region(detection)
            if _geos(current).is_empty:
                raise ValueError("Current RGB material body is empty")
            total = source_bound+current_bound+registration_uncertainty_mm
            if region_bidirectional_within(expected, current, total):
                result["matches"].append({"detection_index": index,
                    "image_sha256": detection.meta["recognition_chain"]["image"]["sha256"],
                    "boundary_uncertainty_mm": current_bound, "combined_bound_mm": total,
                    "region_mm": region_dict(current)})
            elif _geos(expected).distance(_geos(current)) <= total:
                confounders.append(index)
        except (KeyError, ValueError, TypeError) as exc:
            unsupported.append({"detection_index": index, "reason": str(exc)})
    if unsupported:
        return {**result, "reason": "One or more same-material observations are unsupported", "unsupported": unsupported}
    count = len(result["matches"])
    if confounders:
        return {**result, "reason": "Observed material remains at the source support but does not match its complete body",
                "confounding_detection_indices": confounders}
    if count == 1:
        return {**result, "status": "unique"}
    if count > 1:
        return {**result, "status": "ambiguous", "reason": "Multiple complete bodies match the registered original source"}
    if observation_complete:
        return {**result, "status": "absent", "reason": "No complete body matches within the explicit measurement bounds"}
    return {**result, "reason": "No source match; complete visibility is not established"}


class RGBMicroscopeAnalyzer:
    """Material-specific composition, with no implicit legacy method fallback."""

    recognition_backend = BACKEND

    def __init__(self, optical_provider, material_analyzer=None, *, materials=("Gr", "hBN")):
        self.optical_provider = optical_provider
        self.material_analyzer = material_analyzer or RGBMaterialAnalyzer()
        if not materials or any(m not in ("Gr", "hBN") for m in materials) or len(set(materials)) != len(materials):
            raise ValueError("RGB composition requires distinct supported public material classes")
        self.materials = tuple(materials)

    #: The scan keeps its best frame below min_blur, as v1 wait_focus_and_snap did.
    keeps_best_focus_frame = True

    def plane_sharpness(self, frame, plane):
        # Which focal plane (chip or stamp film) an exposure shows: no v1 algorithm.
        return self.optical_provider.plane_sharpness(frame, plane)

    def sharpness(self, frame):
        """v1 blur_score of the captured pixels (log Tenengrad, shrink 2)."""
        from qumus_simlab.perception.v1_vision import blur_score

        return blur_score(np.asarray(load_pixels(frame)))

    def find_corner(self, frame, corner="bl"):
        """v1 edge_follow + corner_coordinate_adjustment on the captured pixels."""
        from qumus_simlab.perception.base import CornerDetection
        from qumus_simlab.perception.v1_vision import corner_from_detector, detect_edge_and_motion

        rgb = np.asarray(load_pixels(frame))
        found = corner_from_detector(detect_edge_and_motion(rgb), rgb.shape)
        if found is None or found.chip_corner != corner:
            return None
        dx, dy = pixel_offset_mm(frame, found.xy_px, rgb.shape)
        return CornerDetection(corner=corner, dx_mm=dx, dy_mm=dy, confidence=0.9)

    def chip_in_view(self, frame):
        return self.optical_provider.chip_in_view(frame)

    def contact(self, frame):
        return self.optical_provider.contact(frame)

    def fringes(self, frame):
        return self.optical_provider.fringes(frame)

    def find_flakes(self, frame, material, **kwargs):
        return self.material_analyzer.find_flakes(frame, material, **kwargs)

    @property
    def prefetch_depth(self):
        return getattr(self.material_analyzer, "prefetch_depth", 0)

    def prefetch_flakes(self, frame, material):
        prefetch = getattr(self.material_analyzer, "prefetch_flakes", None)
        if prefetch is not None:
            prefetch(frame, material)

    def revisit_target(self, frame, material, **kwargs):
        return self.material_analyzer.revisit_target(frame, material, **kwargs)

    def persist_detection_masks(self, frame, detection):
        return self.material_analyzer.persist_detection_masks(frame, detection)

    def discovery_obstacles(self, frame):
        return rgb_discovery_obstacles(frame, plane_quality=self.plane_sharpness(frame, "chip"))

    def find_stamp_flakes(self, frame, *, materials=None):
        if frame.meta.get("focus_plane") != "stamp":
            raise ValueError("Carried-material RGB observation needs a captured stamp focal plane")
        requested = self.materials if materials is None else tuple(materials)
        if not requested or len(set(requested)) != len(requested) or any(m not in self.materials for m in requested):
            raise ValueError("Carried RGB recognition needs distinct supported public catalog classes")
        detections = []
        errors = {}
        for material in requested:
            try:
                found = self.material_analyzer.find_flakes(frame, material)
            except ValueError as exc:
                errors[material] = str(exc)
                continue
            for detection in found:
                detection.meta.update(recognition_profile=RGB_CARRIED_PROFILE,
                    observation_plane="stamp", material_class_source="public_requested_catalog_class",
                    material_identity_status="conditional_catalog_class")
                detections.append(detection)
        # The caller must not mistake an unclassifiable second public class for
        # an observed empty stamp. No incomplete list can certify the footprint.
        if errors:
            raise ValueError("Carried-material RGB classification is unresolved: "+json.dumps(errors, sort_keys=True))
        from qumus_simlab.core.regions import _geos

        for index, first in enumerate(detections):
            for second in detections[index+1:]:
                if first.material == second.material:
                    continue
                try:
                    overlap = _geos(detection_region(first)).intersection(_geos(detection_region(second))).area
                except ValueError:
                    overlap = math.inf
                if overlap > 0:
                    for detection in (first, second):
                        detection.meta.update(material_identity_status="ambiguous_cross_material_pixels",
                            carried_observation_supported=False)
        return detections
