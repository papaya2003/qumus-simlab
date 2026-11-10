"""Captured RGB -> original v1 labels -> separate exact selected-component mask.

No simulator object annotations, World, devices, SDKs or learned model are read.
Historical LAB tables are implementation provenance, not physical calibration.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from qumus_simlab.core.polygons import convex_polygon
from qumus_simlab.perception.base import FlakeDetection

BACKEND = "v1_lab_rgb"
CHAIN_VERSION = "v1-rgb-mask-chain-v1"
DERIVATION_POLICY = "center-priority-original-component-v1"
MAX_RGB_PIXELS = 3560*2360
MAX_RGB_WORK_SECONDS = 90.


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def algorithm_sha256():
    return hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in
        ("_v1_rgb_core.py", "_v1_rgb_materials.py", "_v1_rgb_revisit.py", "rgb_material.py"))).hexdigest()


def material_policy(material, objective):
    from qumus_simlab.perception._v1_rgb_materials import graphene, hBN

    if material not in ("Gr", "hBN") or objective not in (5, 10, 20, 50, 100):
        raise ValueError("Unsupported v1 RGB material or magnification")
    selected = graphene if material == "Gr" else hBN
    return selected, {"backend": BACKEND, "material": material, "objective": objective,
        "background_lab": list(selected.target_bg_lab_for_mag(objective)),
        "labels": [{"lab": list(lab), "label": label} for lab, label in selected.layer_labels_for_mag(objective).items()],
        "edge_args": deepcopy(selected.args[:2] if objective >= 50 else selected.args[1:]),
        "minimum_area_px": 2000 if objective >= 50 else 200,
        "maximum_area_px": 10000000 if objective >= 50 else 100000,
        "surround_background": True, "border_px": 40,
        "sampling": "private-image-sha256-randomstate-v1", "consensus_runs": 3 if objective >= 50 else 1}


def _pixel_cell_ring(coordinates, *, exterior):
    """Canonical exact lattice boundary; coalesce only collinear grid edges."""
    ring = [tuple(point) for point in coordinates][:-1]
    if len(ring) < 3 or len(set(ring)) != len(ring):
        raise ValueError("Nonsimple exact pixel-union boundary")
    signed = sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(ring, ring[1:]+ring[:1], strict=True))
    if (signed > 0) != exterior:
        ring.reverse()
    # GEOS may retain vertices where adjacent exact row rectangles met. Removing
    # a straight-line subdivision changes no edge, corner, selected cell or area.
    reduced = [point for i, point in enumerate(ring)
        if (point[0]-ring[i-1][0])*(ring[(i+1)%len(ring)][1]-point[1])
        != (point[1]-ring[i-1][1])*(ring[(i+1)%len(ring)][0]-point[0])]
    start = min(range(len(reduced)), key=reduced.__getitem__)
    return [[x-.5, y-.5] for x, y in reduced[start:]+reduced[:start]]


def _mask_cell_components(mask):
    """Union exact horizontal runs of closed pixel cells, retaining all topology.

    Integer-grid Boolean union handles diagonal contacts and point-pinched holes
    without a custom ambiguous boundary walk. No fill, buffer, precision snap,
    hull, morphology or approximate contour simplification is applied.
    """
    from shapely import box, union_all

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("Empty or malformed geometry mask")
    if binary.size > MAX_RGB_PIXELS:
        raise ValueError("Pixel geometry exceeds the bounded native raster profile")
    transitions = np.diff(np.pad(binary, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(transitions == 1)
    end_rows, ends = np.nonzero(transitions == -1)
    if not np.array_equal(rows, end_rows):
        raise ValueError("Unpaired exact pixel-cell runs")
    # Bound Boolean work before allocating native geometries. This refuses an
    # excessive input, never drops runs or changes a selected component.
    if len(rows) > 500_000:
        raise ValueError("Pixel geometry exceeds the bounded exact-run profile")
    geometry = union_all(box(starts, rows, ends, rows+1))
    if (not geometry.is_valid or geometry.geom_type not in ("Polygon", "MultiPolygon")
            or geometry.area != int(np.count_nonzero(binary))):
        raise ValueError("Invalid or area-changing exact pixel-cell union")
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    components = [{"exterior": _pixel_cell_ring(p.exterior.coords, exterior=True),
                   "holes": sorted((_pixel_cell_ring(h.coords, exterior=False) for h in p.interiors),
                                   key=lambda ring: tuple(map(tuple, ring)))} for p in polygons]
    return sorted(components, key=lambda part: tuple(map(tuple, part["exterior"])))


def _component_rings(components):
    rings = [ring for part in components for ring in (part["exterior"], *part["holes"])]
    return sorted(rings, key=lambda ring: -abs(sum(a[0]*b[1]-a[1]*b[0]
        for a, b in zip(ring, ring[1:]+ring[:1], strict=True))))


def mask_cell_rings(mask):
    """Exact pixel-cell exteriors/holes, including valid point-touch topology."""
    return _component_rings(_mask_cell_components(mask))


def component_geometry(mask, calibration):
    from qumus_simlab.core.regions import region_dict

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Malformed selected original component")
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("Empty selected original component")
    sx, sy = calibration["mm_per_pixel"]
    if (calibration.get("u_axis") != "+world_x" or calibration.get("v_axis") != "-world_y"
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (sx, sy))):
        raise ValueError("Unsupported signed RGB pixel calibration")
    sy = -sy
    cx, cy = float(xs.mean()), float(ys.mean())
    height, width = binary.shape
    result = {"centroid_px": [cx, cy], "area_um2": int(binary.sum())*abs(sx*sy)*1e6,
        "dx_mm": (cx-width/2)*sx, "dy_mm": (cy-height/2)*sy,
        "status": "unsupported", "unsupported_reason": None, "rings_px": []}
    try:
        components = _mask_cell_components(binary)
        rings = _component_rings(components)
        result["rings_px"] = rings
        def calibrated(ring):
            return [((u-cx)*sx, (v-cy)*sy) for u, v in ring]
        region = region_dict({"version": "polygon-region-v1", "components": [
            {"exterior": calibrated(c["exterior"]), "holes": [calibrated(h) for h in c["holes"]]}
            for c in components]})
        if binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any():
            # Only the in-frame part was observed. It is kept for neighbour
            # clearance, never as this flake's own outline.
            result["visible_region_mm"] = region
            raise ValueError("clipped_component")
        result.update(status="supported", region_mm=region)
        # Historical convex observations remain readable. New detections use
        # the region codec; never invent a ring alias for holes or concavity.
        if len(components) == 1 and not components[0]["holes"]:
            try:
                outline = convex_polygon(calibrated(components[0]["exterior"]))
                result["outline_mm"] = [list(p) for p in outline]
            except ValueError:
                pass
    except ValueError as exc:
        result["unsupported_reason"] = str(exc)
    return result


@dataclass(frozen=True)
class RGBRecognition:
    raw_labels: np.ndarray
    candidates: tuple[dict, ...]
    policy: dict
    image_sha256: str
    objective: int
    diagnostics: tuple[dict, ...] = ()


class RGBRecognitionError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def segment_rgb(rgb, material, *, objective, image_sha256=None, revisit=False, accepted_labels=None, capture_profile=None,
                observation_condition=None, camera_response=None, centered=False):
    """Maintained v1 numerical processor with private image-bound consensus RNG."""
    from qumus_simlab.perception._v1_rgb_core import Segmenter

    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 30:
        raise ValueError("V1 RGB processing needs a native uint8 RGB raster of at least30px per axis")
    if pixels.shape[0]*pixels.shape[1] > MAX_RGB_PIXELS:
        raise ValueError("RGB raster exceeds the bounded processing profile")
    started = time.monotonic()
    source_sha = image_sha256 or hashlib.sha256(pixels.tobytes()).hexdigest()
    selected, policy = material_policy(material, objective)
    if capture_profile is not None:
        from qumus_simlab.core.optical_profiles import (
            V1_MATERIAL_CAPTURE_PROFILE,
            V1_MATERIAL_SENSOR_PX,
            V1_SEARCH_CAMERA_PROFILE,
        )

        if capture_profile != V1_MATERIAL_CAPTURE_PROFILE:
            raise ValueError("Unsupported RGB material capture profile")
        noise = observation_condition.get("observation_noise_scale") if isinstance(observation_condition, dict) else None
        if (camera_response != V1_SEARCH_CAMERA_PROFILE or type(noise) not in (int, float)
                or not math.isfinite(noise) or not 0 <= noise <= 1):
            raise RGBRecognitionError("Native RGB camera/observation condition is unsupported", ())
        if (pixels.shape[1], pixels.shape[0]) != V1_MATERIAL_SENSOR_PX:
            raise RGBRecognitionError("Native material recognition requires its actual native analysis raster", ())
        policy.update(capture_profile=capture_profile, background_support="native_border_and_joint_foreground_v1",
            background_sampling="private_uniform_observed_support_v1", camera_response=camera_response,
            observation_condition=deepcopy(observation_condition))
    if revisit:
        from qumus_simlab.perception._v1_rgb_revisit import RECOVERY_VERSION

        policy.update(centered_revisit=True, recovery_policy=RECOVERY_VERSION,
            recovery_labels=sorted(accepted_labels or ("monolayer", "bilayer")))
        if objective < 50:
            policy["maximum_area_px"] = max(policy["maximum_area_px"], round(.10*pixels.shape[0]*pixels.shape[1]))
    elif centered:
        # v1 LiveSegment._effective_max_area: a capture centred on a chosen
        # flake may hold it at up to 10% of the frame below 50X. Every other
        # component is still listed (no centred recovery), so neighbours stay.
        policy["centered_capture"] = True
        if objective < 50:
            policy["maximum_area_px"] = max(policy["maximum_area_px"], round(.10*pixels.shape[0]*pixels.shape[1]))
    runs, diagnostics = [], []
    for i in range(policy["consensus_runs"]):
        if time.monotonic()-started > MAX_RGB_WORK_SECONDS:
            raise ValueError("V1 RGB recognition exceeded its bounded work deadline")
        seed = (int(source_sha[:8], 16)+i) % 2**32
        processor = Segmenter(pixels, selected, magnification=objective, surround_bg=True, seed=seed,
            min_area=policy["minimum_area_px"], max_area=policy["maximum_area_px"],
            background_support=policy.get("background_support", "historical_v1"),
            observation_noise_scale=(observation_condition or {}).get("observation_noise_scale"))
        diagnostic = {"run": i}
        diagnostics.append(diagnostic)
        try:
            processor.make_masks()
            diagnostic.update(background_pixels_before_crop=int(processor.bg_mask.sum()),
                original_region_count=len(processor.mask_ids),
                background_support_observations=deepcopy(processor.background_support_observations))
            processor.lab_equalize()
        except ValueError as exc:
            diagnostic["failure"] = str(exc)
            if revisit and material == "Gr":
                runs.append((np.zeros(pixels.shape[:2], dtype=np.uint16), []))
                break
            raise RGBRecognitionError(str(exc), diagnostics) from exc
        if time.monotonic()-started > MAX_RGB_WORK_SECONDS:
            raise ValueError("V1 RGB recognition exceeded its bounded work deadline")
        if not processor.bg_mask.any() or not np.isfinite(processor.lab).all():
            diagnostic["failure"] = "background_support_unresolved"
            if revisit and material == "Gr":
                runs.append((np.zeros(pixels.shape[:2], dtype=np.uint16), []))
                break
            raise RGBRecognitionError("Original v1 RGB background support is unresolved after padding/cropping", diagnostics)
        processor.get_all_avg_lab()
        processor.adjust_layer_labels()
        processor.label_masks()
        diagnostic.update(background_pixels_after_crop=int(processor.bg_mask.sum()),
            average_background_lab=processor.avg_bg_lab.tolist(),
            regions=[{"label_id": int(label_id), "label": processor.mask_labels[j],
                      "area_px": int(processor.mask_areas[j]), "average_lab": processor.avg_labs[label_id].tolist()}
                     for j, label_id in enumerate(processor.mask_ids)
                     if processor.mask_areas[j] > policy["minimum_area_px"]])
        raw = processor.masks
        if raw.shape != pixels.shape[:2] or raw.max() > 65535:
            raise ValueError("Original RGB label raster has unsupported shape or label count")
        candidates = []
        # Bounding box of every label once; the pixels of one label are then
        # looked up inside its box only (the same coordinates in the same
        # row-major order as over the whole raster).
        from scipy.ndimage import find_objects

        boxes = find_objects(raw.astype(np.int64, copy=False)) if raw.max() > 0 else []
        for j, label_id in enumerate(processor.mask_ids):
            area, label = processor.mask_areas[j], processor.mask_labels[j]
            if label_id and policy["minimum_area_px"] < area < policy["maximum_area_px"] and label not in ("dirt", "bg", "black", None, ""):
                box = boxes[int(label_id)-1] if int(label_id) <= len(boxes) else None
                if box is None:
                    ys = xs = np.empty(0, dtype=np.intp)
                else:
                    ys, xs = np.nonzero(raw[box] == label_id)
                    ys, xs = ys+box[0].start, xs+box[1].start
                if len(xs):
                    candidate = {"label_id": int(label_id), "thickness_label": label,
                        "area_px": int(area), "centroid_px": [float(xs.mean()), float(ys.mean())]}
                    if capture_profile:
                        guard = max(item["edge_support_band_px"] for item in processor.background_support_observations)
                        near_edge = (xs.min() < guard or ys.min() < guard
                            or xs.max() >= raw.shape[1]-guard or ys.max() >= raw.shape[0]-guard)
                        candidate["support"] = {"producer": "segmenter_original_labels",
                            "version": "native-segmenter-frame-support-v1",
                            "source_size_px": [raw.shape[1], raw.shape[0]],
                            "support_roi_px": [0, 0, raw.shape[1], raw.shape[0]],
                            "operator_guard_native_px": guard,
                            "whole_body_status": "unsupported" if near_edge else "eligible_within_declared_roi",
                            "whole_body_reason": "segmenter_frame_edge_unobserved" if near_edge else None}
                    candidates.append(candidate)
        candidates.sort(key=lambda row: (row["thickness_label"], -row["area_px"], row["centroid_px"][1], row["centroid_px"][0]))
        runs.append((np.array(raw, dtype=np.uint16, copy=True), candidates))
    # Frozen LiveSegment's same-label centroid/relative-area consensus bounds.
    base, candidates = runs[0]
    consensus_ok = True
    for _, other in runs[1:]:
        unused = list(other)
        if len(unused) != len(candidates):
            consensus_ok = False
            break
        for candidate in candidates:
            matched = next((row for row in unused if row["thickness_label"] == candidate["thickness_label"]
                and math.dist(row["centroid_px"], candidate["centroid_px"]) <= 35.
                and abs(row["area_px"]-candidate["area_px"])/max(row["area_px"], candidate["area_px"], 1.) <= .35), None)
            if matched is None:
                consensus_ok = False
                break
            unused.remove(matched)
    if not consensus_ok and not revisit:
        raise RGBRecognitionError("V1 RGB repeated classifications are unresolved", diagnostics)
    if revisit:
        from qumus_simlab.perception._v1_rgb_revisit import CenteredRevisitRecovery

        recovery = CenteredRevisitRecovery(selected, material, objective,
            policy["minimum_area_px"], policy["maximum_area_px"])
        try:
            target = center_priority_candidate(candidates, [pixels.shape[1], pixels.shape[0]])
        except ValueError:
            target = None
        fallback = recovery._center_color_fallback(pixels, accepted_labels=accepted_labels)
        prefer_fallback = bool(fallback is not None and (
            recovery._is_centered_monolayer_recovery(fallback[1], pixels.shape[:2])
            or accepted_labels == {"manylayer"} and objective >= 50 and fallback[1]["label"] == "manylayer"))
        diagnostics.append({"consensus_ok": consensus_ok, "centered_recovery_available": fallback is not None,
            "centered_recovery_preferred": prefer_fallback})
        if target is None or prefer_fallback:
            if fallback is None:
                raise RGBRecognitionError("No centered RGB material revisit candidate", diagnostics)
            mask, target = fallback
            # As in LiveSegment's output image, recovery is the recognizer's
            # original result. Geometry subsequently selects these exact pixels.
            base = mask.astype(np.uint16)
            centroid_row, centroid_col = target["centroid"]
            target = {"label_id": 1, "thickness_label": target["label"], "area_px": int(mask.sum()),
                "centroid_px": [float(centroid_col), float(centroid_row)], "support": deepcopy(target["support"])}
            policy["original_mask_producer"] = "live_segment_center_color_recovery"
        else:
            policy["original_mask_producer"] = "segmenter_original_labels"
        candidates = [target]
    base.setflags(write=False)
    return RGBRecognition(base, tuple(candidates), policy, source_sha, objective, tuple(diagnostics))


def center_priority_candidate(candidates, size_px):
    """Frozen v1 central-zone/near-tie selection, with exact ties explicit."""
    if not candidates:
        raise ValueError("No RGB material candidate")
    center = [size_px[0]/2, size_px[1]/2]
    ranked = [(math.dist(c["centroid_px"], center), c) for c in candidates]
    nearest = min(distance for distance, _ in ranked)
    if nearest > .35*min(size_px):
        raise ValueError("No centered RGB revisit target")
    central = [(d, c) for d, c in ranked if d <= .12*min(size_px)]
    pool = central or [(d, c) for d, c in ranked if d <= nearest+50.]
    maximum = max(c["area_px"] for _, c in pool)
    chosen = [(d, c) for d, c in pool if c["area_px"] == maximum]
    if len(chosen) != 1:
        raise ValueError("Ambiguous v1 center-priority target")
    return chosen[0][1]


_POOL = None


def _pool_workers():
    """Worker processes for scan-tile recognition: QUMUS_RGB_WORKERS, else
    cores - 1 (0 or 1 core: serial)."""
    import os

    value = os.environ.get("QUMUS_RGB_WORKERS")
    if value is not None:
        return max(0, int(value))
    return max(0, (os.cpu_count() or 1)-1)


def _pool():
    """One lazily started process pool (spawned, so no forked lab state)."""
    global _POOL
    if _POOL is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        _POOL = ProcessPoolExecutor(max_workers=_pool_workers(),
                                    mp_context=multiprocessing.get_context("spawn"))
        import atexit

        # Queued tiles are dropped at exit; workers hold no store or lab state.
        atexit.register(_POOL.shutdown, wait=False, cancel_futures=True)
    return _POOL


class RGBMaterialAnalyzer:
    """Material-only analyzer adapter; other optical methods stay with their owner."""
    recognition_backend = BACKEND

    def __init__(self):
        self._cache = {}
        self._pending = {}

    @property
    def prefetch_depth(self):
        """How many scan tiles may be recognised ahead of their use (0: serial)."""
        return _pool_workers()

    def prefetch_flakes(self, frame, material):
        """Start recognising `frame` in a worker process; find_flakes collects it.

        The worker runs the same deterministic segment_rgb on the same pixels,
        so results are identical to serial recognition. Without a pool this
        does nothing and find_flakes recognises in-process as before.
        """
        if not _pool_workers():
            return
        rgb, key, kwargs, source_sha, objective = self._request(frame, material, False)[1:6]
        if key in self._cache or key in self._pending:
            return
        while len(self._pending) > 2*_pool_workers()+2:  # abandoned (e.g. a cancelled scan)
            self._pending.pop(next(iter(self._pending))).cancel()
        try:
            self._pending[key] = _pool().submit(segment_rgb, rgb, material, objective=objective,
                                                image_sha256=source_sha, **kwargs)
        except Exception:  # noqa: BLE001 - no pool: recognise serially later
            self._pending.pop(key, None)

    def _request(self, frame, material, revisit, centered=False, accepted_labels=None):
        raw = frame.data if frame.data is not None else Path(frame.saved_path()).read_bytes()
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        objective = frame.meta.get("objective")
        calibration = deepcopy(frame.meta.get("calibration", {}))
        if calibration.get("image_size_px") != [rgb.shape[1], rgb.shape[0]]:
            raise ValueError("RGB capture/calibration dimensions differ")
        source_sha = hashlib.sha256(raw).hexdigest()
        capture_profile = calibration.get("material_capture_profile")
        if capture_profile is not None:
            _validate_native_capture_geometry(frame, rgb, calibration, capture_profile)
        condition = calibration.get("observation_condition") if capture_profile else None
        response = (calibration.get("photometry_profile") or {}).get("material_camera_response") if capture_profile else None
        centered = bool(centered and not revisit)
        labels = tuple(sorted(accepted_labels or ("monolayer", "bilayer"))) if revisit else None
        key = (source_sha, material, objective, revisit, capture_profile, _digest(condition or {}), response, centered, labels)
        kwargs = {"revisit": True} if revisit else {"centered": True} if centered else {}
        if revisit and accepted_labels:
            kwargs["accepted_labels"] = set(accepted_labels)
        if capture_profile is not None:
            kwargs["capture_profile"] = capture_profile
            kwargs.update(observation_condition=condition, camera_response=response)
        return calibration, rgb, key, kwargs, source_sha, objective

    def find_flakes(self, frame, material, *, revisit=False, centered=False, accepted_labels=None):
        """`centered`: the capture is centred on a chosen flake (v1 revisit
        area ceiling) but every component is returned, as in a scan.
        `accepted_labels`: revisit colour-recovery labels (v1 graphite revisit: manylayer)."""
        calibration, rgb, key, kwargs, source_sha, objective = self._request(
            frame, material, revisit, centered, accepted_labels)
        if key not in self._cache:
            recognition = None
            future = self._pending.pop(key, None)
            if future is not None:
                try:
                    recognition = future.result()
                    recognition.raw_labels.setflags(write=False)
                except Exception:  # noqa: BLE001 - recompute here: same result or same error
                    recognition = None
            if recognition is None:
                recognition = segment_rgb(rgb, material, objective=objective, image_sha256=source_sha, **kwargs)
            self._cache[key] = recognition
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
        recognition = self._cache[key]
        detections = []
        for candidate in recognition.candidates:
            mask = recognition.raw_labels == candidate["label_id"]
            geometry = component_geometry(mask, calibration)
            support = deepcopy(candidate.get("support") or {})
            if support.get("whole_body_status") == "unsupported":
                geometry.update(status="unsupported", unsupported_reason=support["whole_body_reason"])
            chain = {"version": CHAIN_VERSION, "backend": BACKEND, "algorithm_sha256": algorithm_sha256(),
                "material": material, "material_policy": deepcopy(recognition.policy), "material_policy_sha256": _digest(recognition.policy),
                "image": {"path": frame.saved_path(), "sha256": source_sha, "objective": objective,
                          "size_px": [rgb.shape[1], rgb.shape[0]]},
                "selected_label_ids": [candidate["label_id"]], "calibration": calibration,
                "derivation": {"policy": DERIVATION_POLICY, "operation": "select_original_label_only", "morphology": "none",
                    "selected_label_id": candidate["label_id"],
                    "selection_context": "all_candidates", "ambiguity_status": "not_evaluated_all_candidates",
                    "selection_distance_px": math.dist(geometry["centroid_px"], [rgb.shape[1]/2, rgb.shape[0]/2]),
                    "candidate_count": len(recognition.candidates), "centroid_px": geometry["centroid_px"],
                    "rings_px": geometry["rings_px"], "status": geometry["status"], "unsupported_reason": geometry["unsupported_reason"],
                    "recognizer_support": support}}
            detections.append(FlakeDetection(dx_mm=geometry["dx_mm"], dy_mm=geometry["dy_mm"], area_um2=geometry["area_um2"],
                material=material, thickness_label=candidate["thickness_label"], confidence=0., meta={
                    "recognition_chain": chain, "recognition_diagnostics": deepcopy(recognition.diagnostics),
                    "recognizer_support": support,
                    "geometry_source": BACKEND, "outline_complete": geometry["status"] == "supported",
                    "confidence_source": "historical_v1_classifier_not_probability_calibrated",
                    "unsupported_reason": geometry["unsupported_reason"], **{key: geometry[key] for key in ("region_mm", "visible_region_mm") if key in geometry}}))
        return detections

    def revisit_target(self, frame, material, accepted_labels=None):
        detections = self.find_flakes(frame, material, revisit=True, accepted_labels=accepted_labels)
        candidates = [{"label_id": d.meta["recognition_chain"]["selected_label_ids"][0],
            "area_px": d.area_um2, "centroid_px": d.meta["recognition_chain"]["derivation"]["centroid_px"], "detection": d} for d in detections]
        detection = center_priority_candidate(candidates, frame.meta["calibration"]["image_size_px"])["detection"]
        detection.meta["recognition_chain"]["derivation"].update(selection_context="v1_center_priority_revisit", ambiguity_status="unique")
        return detection

    def persist_detection_masks(self, frame, detection):
        chain = deepcopy(detection.meta["recognition_chain"])
        if not frame.saved_path():
            raise ValueError("Paired RGB masks require their existing saved source image")
        image = Path(frame.saved_path()).resolve()
        raw = image.read_bytes()
        if hashlib.sha256(raw).hexdigest() != chain["image"]["sha256"]:
            raise ValueError("RGB source image changed before paired-mask persistence")
        recognition = self._cache[(chain["image"]["sha256"], detection.material, chain["image"]["objective"],
            bool(chain["material_policy"].get("centered_revisit")), chain["material_policy"].get("capture_profile"),
            _digest(chain["material_policy"].get("observation_condition") or {}), chain["material_policy"].get("camera_response"),
            bool(chain["material_policy"].get("centered_capture")),
            tuple(chain["material_policy"]["recovery_labels"]) if chain["material_policy"].get("centered_revisit") else None)]
        selected = chain["selected_label_ids"][0]
        for name, pixels, suffix in (("original_mask", recognition.raw_labels, "labels"),
            ("geometry_mask", np.where(recognition.raw_labels == selected, 255, 0).astype(np.uint8), f"geometry-{selected}")):
            output = io.BytesIO()
            Image.fromarray(pixels).save(output, format="PNG")
            encoded = output.getvalue()
            digest = hashlib.sha256(encoded).hexdigest()
            path = image.parent/f"{image.stem}-{suffix}-{digest[:16]}.png"
            if path.exists() and path.read_bytes() != encoded:
                raise ValueError("Immutable RGB mask artifact changed")
            if not path.exists():
                path.write_bytes(encoded)
            chain[name] = {"path": str(path), "sha256": digest}
        detection.meta["recognition_chain"] = chain
        return chain


def _validate_native_capture_geometry(frame, pixels, calibration, capture_profile):
    """Refuse a metadata-only reinterpretation of the explicit native sensor."""
    from qumus_simlab.core.optical_profiles import (
        V1_MATERIAL_CAPTURE_PROFILE,
        V1_MATERIAL_PITCH_UM,
        V1_MATERIAL_SENSOR_PX,
    )

    objective = frame.meta.get("objective")
    if (capture_profile != V1_MATERIAL_CAPTURE_PROFILE or type(objective) is not int
            or objective not in V1_MATERIAL_PITCH_UM
            or (pixels.shape[1], pixels.shape[0]) != V1_MATERIAL_SENSOR_PX
            or calibration.get("image_size_px") != list(V1_MATERIAL_SENSOR_PX)
            or calibration.get("u_axis") != "+world_x" or calibration.get("v_axis") != "-world_y"):
        raise ValueError("Native RGB capture objective/raster/signed axes are unsupported")
    pitch = V1_MATERIAL_PITCH_UM[objective]/1000
    def same(values, expected):
        return (isinstance(values, (tuple, list)) and len(values) == len(expected)
            and all(type(actual) in (int, float) and math.isfinite(actual)
                    and math.isclose(actual, target, rel_tol=1e-12, abs_tol=8*math.ulp(target))
                    for actual, target in zip(values, expected, strict=True)))
    fov = [pixels*pitch for pixels in V1_MATERIAL_SENSOR_PX]
    if (not same(calibration.get("mm_per_pixel"), [pitch, pitch])
            or not same(frame.meta.get("fov_mm"), fov)
            or "fov_mm" in calibration and not same(calibration["fov_mm"], fov)):
        raise ValueError("Native RGB capture pixel pitch/FOV differs from its declared sensor profile")
