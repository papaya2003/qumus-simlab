"""Image-learned raster boundary controls; no World, v1 rerun or truth input."""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
import pytest

from qumus_simlab.core.optical_profiles import (
    CAMERA_RESPONSE_PROFILE,
    V1_MATERIAL_CAPTURE_PROFILE,
    V1_MATERIAL_SENSOR_PX,
    V1_SEARCH_CAMERA_PROFILE,
)
from qumus_simlab.perception.boundary import refine_rgb_boundary
from qumus_simlab.testbench.render import Canvas, _camera


def fixture_case(*, split=False, clipped=False, uniform=False, foreground=(166, 97, 136)):
    width, height = V1_MATERIAL_SENSOR_PX
    background = (79, 101, 143)
    canvas = Canvas(width, height, background)
    truth = np.zeros((height, width), dtype=bool)
    left = 0 if clipped else 1300
    if not uniform:
        canvas.rect(left, 900, 2250, 1500, foreground)
        truth[900:1500, left:2250] = True
        if split:
            canvas.rect(1775, 900, 2250, 1500, (120, 92, 172))
        # A real image-visible hole; the original classifier seed contains it.
        canvas.rect(1600, 1000, 1700, 1100, background)
        truth[1000:1100, 1600:1700] = False
    _camera(canvas, 733, 1., V1_SEARCH_CAMERA_PROFILE)
    rgb = np.frombuffer(canvas.px, dtype=np.uint8).reshape(height, width, 3).copy()
    seed = np.zeros_like(truth)
    seed[950:1450, 1350:2200] = True
    # A false classifier hole where original RGB still supports foreground.
    seed[1200:1230, 1800:1850] = False
    pitch = .000182
    calibration = {"source": "synthetic", "material_capture_profile": V1_MATERIAL_CAPTURE_PROFILE,
        "image_size_px": [width, height], "mm_per_pixel": [pitch, pitch],
        "fov_mm": [width*pitch, height*pitch], "u_axis": "+world_x", "v_axis": "-world_y",
        "photometry_profile": {"camera_response": CAMERA_RESPONSE_PROFILE,
                               "material_camera_response": V1_SEARCH_CAMERA_PROFILE},
        "observation_condition": {"observation_noise_scale": 1.}}
    return rgb, seed, calibration, truth


@pytest.fixture(scope="module")
def nominal():
    return fixture_case()


def test_image_support_restores_false_seed_holes_and_preserves_real_holes(nominal):
    rgb, seed, calibration, truth = nominal
    before = hashlib.sha256(seed.tobytes()).hexdigest()
    result = refine_rgb_boundary(rgb, seed, calibration, material="Gr", objective=20)
    assert result["status"] == "measured" and result["reason"] is None
    assert np.array_equal(result["derived_mask"], truth)
    assert result["derived_mask"][1210, 1820]  # Independently RGB-supported restoration.
    assert not result["derived_mask"][1050, 1650]  # A real hole is not filled.
    assert hashlib.sha256(seed.tobytes()).hexdigest() == before
    assert result["physical_qualification"] == "unqualified"
    assert result["diagnostics"]["physical_boundary_budget_mm"] is None
    assert result["pixel_convention"]["sample_corner_px"] == "(column+0.5,row+0.5)"
    assert set(result["provenance"]["dependencies"]) == {
        "boundary.py", "optical_profiles.py", "rgb_material.py", "numpy_version", "scipy_version"}


def test_unretuned_custom_image_colours_do_not_depend_on_requested_material_palette(nominal):
    rgb, seed, calibration, truth = nominal
    result = refine_rgb_boundary(rgb, seed, calibration, material="hBN", objective=20)
    assert result["status"] == "measured" and np.array_equal(result["derived_mask"], truth)


@pytest.mark.parametrize("case,reasons", [("uniform", {"foreground_background_colour_ambiguous"}),
    ("split", {"unresolved_rgb_boundary_pixels", "foreground_no_dominant_colour_class", "foreground_nonuniform_colour_class"}),
    ("clipped", {"selected_material_clipped_by_image"})])
def test_unobservable_or_nonuniform_body_remains_unknown(case, reasons):
    rgb, seed, calibration, _ = fixture_case(**{case: True})
    result = refine_rgb_boundary(rgb, seed, calibration, material="Gr", objective=20)
    assert result["status"] == "unknown" and result["reason"] in reasons
    assert result["physical_qualification"] == "unqualified"
    assert result["diagnostics"].get("local_edge_evidence", "not_established") == "not_established"


def test_saturated_foreground_is_unknown_without_force_fitting(nominal):
    rgb, seed, calibration, _ = nominal
    saturated = rgb.copy()
    saturated[seed] = 0
    result = refine_rgb_boundary(saturated, seed, calibration, material="Gr", objective=20)
    assert result["status"] == "unknown" and result["reason"] == "foreground_saturated_colour_samples"


def test_censored_red_does_not_discard_separable_observed_blue_green_difference():
    rgb, seed, calibration, truth = fixture_case(foreground=(0, 83, 170))
    result = refine_rgb_boundary(rgb, seed, calibration, material="Gr", objective=20)
    assert result["status"] == "measured"
    assert result["diagnostics"]["common_observable_differences"] == ["B-G"]
    assert np.array_equal(result["derived_mask"], truth)
    assert result["physical_qualification"] == "unqualified"


def test_censored_red_with_ambiguous_blue_green_stays_unknown():
    # The observed B-G contrast equals background. A saturated R channel must
    # not be invented as an exact colour to manufacture class separation.
    rgb, seed, calibration, _ = fixture_case(foreground=(0, 100, 142))
    result = refine_rgb_boundary(rgb, seed, calibration, material="Gr", objective=20)
    assert result["status"] == "unknown"
    assert result["reason"] == "foreground_background_colour_ambiguous"


@pytest.mark.parametrize("field,value,reason", [("noise", 1.8, "unsupported_noise_condition"),
                                             ("axis", "+world_y", "unsupported_raster_or_axes"),
                                             ("pitch", .001, "unsupported_pixel_calibration")])
def test_unknown_capture_conditions_never_gain_physical_qualification(nominal, field, value, reason):
    rgb, seed, calibration, _ = nominal
    calibration = deepcopy(calibration)
    if field == "noise":
        calibration["observation_condition"]["observation_noise_scale"] = value
    elif field == "axis":
        calibration["v_axis"] = value
    else:
        calibration["mm_per_pixel"] = [value, value]
    result = refine_rgb_boundary(rgb, seed, calibration, material="Gr", objective=20)
    assert result["status"] == "unknown" and result["reason"] == reason
    assert result["physical_qualification"] == "unqualified"


def test_invalid_original_mask_type_is_not_silently_reinterpreted(nominal):
    rgb, seed, calibration, _ = nominal
    with pytest.raises(ValueError, match="original binary mask"):
        refine_rgb_boundary(rgb, seed.astype(np.uint8), calibration, material="Gr", objective=20)
