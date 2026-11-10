"""RGB-derived raster measurement seeded by, but separate from, a v1 mask.

This conditional SIM method learns foreground/background chromatic differences
from the image. It does not invert a material palette, use truth geometry, fit a
shape, dilate an outline or qualify a continuous physical boundary. The public
camera law supplies only gain and the achromatic-noise/truncation assumptions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage

from qumus_simlab.core.optical_profiles import (
    CAMERA_RESPONSE_PROFILE,
    V1_MATERIAL_CAPTURE_PROFILE,
    V1_MATERIAL_PITCH_UM,
    V1_MATERIAL_SENSOR_PX,
    V1_SEARCH_CAMERA_PROFILE,
    camera_vignette_axes,
)

METHOD_ID = "seeded-rgb-chroma-interval-raster-v2"
PIXEL_CONVENTION = {
    "version": "canvas-corner-sampled-cell-centre-v1",
    "sample_corner_px": "(column+0.5,row+0.5)",
    "world_mm": "((column+0.5-width/2)*sx,(height/2-row-0.5)*sy)",
    "boundary_cells_corner_px": "[column,column+1] x [row,row+1]",
    "scope": "Raster coordinates only; no universal continuous-boundary error bound",
}
_MAX_SAMPLES = 32_768
_MIN_SAMPLES = 64
_ROUND_GUARD = 1e-9  # Chromatic RGB units, not a geometric tolerance.
_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def algorithm_identity():
    """Method and optical/codec identities, excluding mutable wrapper receipts."""
    here = Path(__file__).resolve()
    paths = {"boundary.py": here, "optical_profiles.py": here.parent.parent/"core/optical_profiles.py",
             "rgb_material.py": here.with_name("rgb_material.py")}
    dependencies = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    dependencies.update(numpy_version=np.__version__, scipy_version=scipy.__version__)
    policy = {"method_id": METHOD_ID, "max_statistical_samples": _MAX_SAMPLES,
        "minimum_statistical_samples": _MIN_SAMPLES, "class_model": "dominant_constant_chromatic_difference",
        "quantization_error_per_difference": "strictly less than1 RGB unit before gain normalization",
        "censoring": "Use only paired differences with majority observable FG/BG samples; required pixel channels must be uncensored",
        "seed_core": "one-pixel4-neighbour interior, statistics only",
        "physical_qualification": "unqualified", "pixel_convention": PIXEL_CONVENTION}
    return {"algorithm_sha256": _digest({"dependencies": dependencies, "policy": policy}),
            "dependencies": dependencies, "policy": policy}


def _sample_indices(mask):
    indices = np.flatnonzero(mask)
    if len(indices) > _MAX_SAMPLES:
        indices = indices[np.linspace(0, len(indices)-1, _MAX_SAMPLES, dtype=np.int64)]
    return indices


def _fit_chroma(rgb, indices, fx, fy):
    """Robust image-learned class interval; no preselected material colours."""
    if len(indices) < _MIN_SAMPLES:
        return None, "insufficient_colour_samples"
    width = rgb.shape[1]
    ys, xs = np.divmod(indices, width)
    values = rgb.reshape(-1, 3)[indices].astype(np.float64)
    valid_channels = (values > 0) & (values < 255)
    valid_pairs = np.column_stack((valid_channels[:, 0] & valid_channels[:, 1],
                                   valid_channels[:, 2] & valid_channels[:, 1]))
    gain = fx[xs]*fy[ys]
    differences = np.column_stack((values[:, 0]-values[:, 1], values[:, 2]-values[:, 1]))
    normalized = differences/gain[:, None]
    error = 1./gain
    lower, upper = np.zeros(2), np.zeros(2)
    counts = valid_pairs.sum(axis=0)
    observable = (counts > len(indices)/2) & (counts >= _MIN_SAMPLES)
    if not observable.any():
        return None, "saturated_colour_samples"
    inliers_by_dimension = np.zeros((len(indices), 2), dtype=bool)
    for dimension in np.flatnonzero(observable):
        valid = valid_pairs[:, dimension]
        centre = np.median(normalized[valid, dimension])
        inliers_by_dimension[:, dimension] = (valid
            & (abs(normalized[:, dimension]-centre) <= error+error[valid].max()+_ROUND_GUARD))
    # Keep multivariate class support: a contaminant close in one difference
    # cannot poison that interval if the other observed difference excludes it.
    inliers = np.all(inliers_by_dimension[:, observable], axis=1)
    joint_valid = np.all(valid_pairs[:, observable], axis=1)
    if int(inliers.sum()) <= int(joint_valid.sum())/2 or int(inliers.sum()) < _MIN_SAMPLES:
        return None, "no_dominant_colour_class"
    for dimension in np.flatnonzero(observable):
        lo = float(np.max(normalized[inliers, dimension]-error[inliers]))
        hi = float(np.min(normalized[inliers, dimension]+error[inliers]))
        if lo > hi+_ROUND_GUARD:
            # A well-observed but nonuniform dimension cannot be discarded to
            # force a fit to a second thickness or an unmodeled optical region.
            return None, "nonuniform_colour_class"
        lower[dimension], upper[dimension] = lo, hi
        inliers_by_dimension[:, dimension] = inliers
    return {"lower": lower, "upper": upper, "observable_dimensions": observable,
        "sample_indices": indices, "inliers_by_dimension": inliers_by_dimension,
        "inlier_indices": indices[inliers], "observed_sample_counts": valid_pairs.sum(axis=0).tolist(),
        "sample_count": int(len(indices)), "inlier_count": int(inliers.sum()),
        "dominant_fraction": float(inliers.mean())}, None


def _nearby(mask):
    """Read four-neighbour adjacency; never use it as the measured footprint."""
    result = np.zeros_like(mask)
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _capture_reason(rgb, calibration, material, objective):
    if material not in ("Gr", "hBN") or type(objective) is not int or objective not in (20, 100):
        return "unsupported_material_or_objective"
    if (calibration.get("source") != "synthetic"
            or calibration.get("material_capture_profile") != V1_MATERIAL_CAPTURE_PROFILE
            or (calibration.get("photometry_profile") or {}).get("camera_response") != CAMERA_RESPONSE_PROFILE
            or (calibration.get("photometry_profile") or {}).get("material_camera_response") != V1_SEARCH_CAMERA_PROFILE):
        return "unsupported_camera_profile"
    noise = (calibration.get("observation_condition") or {}).get("observation_noise_scale")
    if type(noise) not in (int, float) or not math.isfinite(noise) or not 0 <= noise <= 1:
        return "unsupported_noise_condition"
    height, width = rgb.shape[:2]
    if ((width, height) != V1_MATERIAL_SENSOR_PX or calibration.get("image_size_px") != [width, height]
            or calibration.get("u_axis") != "+world_x" or calibration.get("v_axis") != "-world_y"):
        return "unsupported_raster_or_axes"
    pitch = V1_MATERIAL_PITCH_UM[objective]/1000
    scales = calibration.get("mm_per_pixel")
    if (not isinstance(scales, (list, tuple)) or len(scales) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or not math.isclose(v, pitch, rel_tol=1e-12, abs_tol=8*math.ulp(pitch)) for v in scales)):
        return "unsupported_pixel_calibration"
    return None


def refine_rgb_boundary(rgb_uint8, original_v1_mask_bool, calibration_dict, *, material, objective,
                        recognition_policy=None):
    """Return an independent measured raster or explicit unresolved evidence.

    ``status`` is measured/unknown. Physical qualification is always unqualified.
    Returned NumPy masks use the original absolute pixel grid; no input is edited.
    Real holes remain background. A hole in the old classifier mask is restored
    only if its original RGB independently supports foreground connectivity.
    """
    rgb, original = np.asarray(rgb_uint8), np.asarray(original_v1_mask_bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Boundary measurement requires original RGB8 pixels")
    if original.dtype != np.bool_ or original.shape != rgb.shape[:2]:
        raise ValueError("Boundary measurement requires the same-size original binary mask")
    identity = algorithm_identity()
    height, width = original.shape
    provenance = {"method_id": METHOD_ID, **identity,
        "rgb_raster_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "original_mask_raster_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
        "calibration_sha256": _digest(calibration_dict), "recognition_policy_sha256": _digest(recognition_policy or {}),
        "inputs": "Original RGB, immutable v1 seed, public calibration/requested class only",
        "physical_assumptions": ["Known SIM achromatic additive camera response; required channel differences are uncensored",
            "Dominant locally uniform foreground/background chromatic classes",
            "Image-connected selected material; optical/raster observation, not specimen identity proof"],
        "continuous_boundary_qualification": "unqualified"}
    result = {"status": "unknown", "physical_qualification": "unqualified", "derived_mask": None,
        "uncertain_mask": np.ones(original.shape, dtype=bool), "edge_mask": np.zeros(original.shape, dtype=bool),
        "reason": None, "method_id": METHOD_ID, "algorithm_sha256": identity["algorithm_sha256"],
        "pixel_convention": dict(PIXEL_CONVENTION), "provenance": provenance, "diagnostics": {}}
    reason = _capture_reason(rgb, calibration_dict, material, objective)
    if reason:
        return {**result, "reason": reason}
    if not original.any():
        return {**result, "reason": "empty_original_identity_seed"}
    fx, fy = (np.asarray(v) for v in camera_vignette_axes(width, height, V1_SEARCH_CAMERA_PROFILE))
    core = ndimage.binary_erosion(original, structure=_FOUR, iterations=1, border_value=0)
    fg, reason = _fit_chroma(rgb, _sample_indices(core), fx, fy)
    if reason:
        return {**result, "reason": "foreground_"+reason}
    rim = np.zeros(original.shape, dtype=bool)
    rim[0] = rim[-1] = True
    rim[:, 0] = rim[:, -1] = True
    bg, reason = _fit_chroma(rgb, _sample_indices(rim), fx, fy)
    if reason:
        return {**result, "reason": "background_"+reason}
    def model_record(model):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in model.items() if key not in ("inlier_indices", "sample_indices", "inliers_by_dimension")}
    dimensions = fg["observable_dimensions"] & bg["observable_dimensions"]
    if not dimensions.any():
        return {**result, "reason": "no_common_observable_colour_difference"}
    fg["inlier_indices"] = fg["sample_indices"][np.all(fg["inliers_by_dimension"][:, dimensions], axis=1)]
    result["diagnostics"].update(foreground=model_record(fg), background=model_record(bg),
        seed_statistics="one-pixelinterior; dominant learned class; original membership is not boundary truth")
    result["diagnostics"]["common_observable_differences"] = np.asarray(["R-G", "B-G"])[dimensions].tolist()
    gap = np.maximum(np.maximum(fg["lower"][dimensions]-bg["upper"][dimensions],
                               bg["lower"][dimensions]-fg["upper"][dimensions]), 0.)
    result["diagnostics"]["class_interval_gap_rgb"] = gap.tolist()
    if not np.any(gap > _ROUND_GUARD):
        return {**result, "reason": "foreground_background_colour_ambiguous"}
    definite_fg, uncertain = np.zeros_like(original), np.zeros_like(original)
    for top in range(0, height, 64):
        bottom = min(top+64, height)
        values = rgb[top:bottom].astype(np.float64)
        gain = fy[top:bottom, None]*fx[None, :]
        difference = np.stack((values[:, :, 0]-values[:, :, 1], values[:, :, 2]-values[:, :, 1]), axis=2)
        lower, upper = (difference-1.)/gain[:, :, None], (difference+1.)/gain[:, :, None]
        valid_channels = (values > 0) & (values < 255)
        valid_pairs = np.stack((valid_channels[:, :, 0] & valid_channels[:, :, 1],
                                valid_channels[:, :, 2] & valid_channels[:, :, 1]), axis=2)
        valid = np.all(valid_pairs[:, :, dimensions], axis=2)
        possible_fg = valid & np.all((upper[:, :, dimensions] >= fg["lower"][dimensions]-_ROUND_GUARD)
            & (lower[:, :, dimensions] <= fg["upper"][dimensions]+_ROUND_GUARD), axis=2)
        possible_bg = valid & np.all((upper[:, :, dimensions] >= bg["lower"][dimensions]-_ROUND_GUARD)
            & (lower[:, :, dimensions] <= bg["upper"][dimensions]+_ROUND_GUARD), axis=2)
        definite_fg[top:bottom] = possible_fg & ~possible_bg
        uncertain[top:bottom] = ~(possible_fg ^ possible_bg)
    labels, count = ndimage.label(definite_fg, structure=_FOUR)
    seed_labels = np.unique(labels.ravel()[fg["inlier_indices"]])
    seed_labels = seed_labels[seed_labels != 0]
    result["uncertain_mask"] = uncertain
    if len(seed_labels) != 1:
        return {**result, "reason": "disconnected_or_unobserved_material_seed"}
    selected = labels == seed_labels[0]
    neighbors = _nearby(selected)
    outside = neighbors & ~selected
    inside_edge = selected & _nearby(~selected)
    result.update(derived_mask=selected, edge_mask=outside | inside_edge)
    result["diagnostics"].update(foreground_components=int(count), selected_pixels=int(selected.sum()),
        added_to_v1_pixels=int((selected & ~original).sum()), removed_from_v1_pixels=int((original & ~selected).sum()),
        uncertain_boundary_pixels=int((uncertain & outside).sum()),
        physical_boundary_budget_mm=None, local_edge_evidence="not_established")
    if selected[0].any() or selected[-1].any() or selected[:, 0].any() or selected[:, -1].any():
        return {**result, "reason": "selected_material_clipped_by_image"}
    if np.any(uncertain & outside):
        return {**result, "reason": "unresolved_rgb_boundary_pixels"}
    result["diagnostics"]["local_edge_evidence"] = "Selected foreground has a closed uniquely classified background neighbor set"
    return {**result, "status": "measured", "reason": None}
