"""Image-only composition, visible-cell support and whole-body association."""
from __future__ import annotations

import hashlib
import io
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.optical_profiles import CAMERA_RESPONSE_PROFILE, SYNTHETIC_SUBSTRATE_RGB
from qumus_simlab.core.regions import region_dict
from qumus_simlab.devices.base import Frame
from qumus_simlab.perception import measurement
from qumus_simlab.perception.base import FlakeDetection
from qumus_simlab.perception.observations import (
    RGBMicroscopeAnalyzer,
    associate_source_observations,
    rgb_discovery_obstacles,
)
from qumus_simlab.records.discovery import choose_clear_domain, discovery_policy
from qumus_simlab.records.geometry import geometry_digest


def region(exterior=None, holes=()):
    return region_dict({"version": "polygon-region-v1", "components": [{
        "exterior": exterior or [[-.01, -.01], [.01, -.01], [.01, .01], [-.01, .01]],
        "holes": holes}]})


def measured(body=None, *, image="a", bound=.0001, material="Gr"):
    calibration = {"source": "declared_unit", "mm_per_pixel": [.0001, .0001]}
    image_sha = hashlib.sha256(image.encode()).hexdigest()
    return {"region_mm": body or region(), "outline_complete": True,
        "recognition_chain": {"image": {"sha256": image_sha, "objective": 20},
            "material": material, "calibration": calibration, "algorithm_sha256": "a"*64,
            "material_policy_sha256": "b"*64, "material_policy": {"original_mask_producer": "analytic_unit_labels"}},
        "boundary_uncertainty_mm": bound,
        "boundary_uncertainty_provenance": {"version": "sim-boundary-uncertainty-v1",
            "source": "simulation_observation_profile", "profile_id": "declared_unit",
            "profile_sha256": "0"*64, "image_sha256": image_sha,
            "calibration_sha256": geometry_digest(calibration)}}


@pytest.fixture(autouse=True)
def analytic_measurement_authority_test_double(monkeypatch):
    """These unit shapes test association, not production SIM qualification."""
    sample = measured()
    profile = {"version": "rgb-boundary-measurement-profile-v1", "profile_id": "declared_unit",
        "status": "qualified_conditional_sim", "capture_profile": None,
        "method_id": "analytic-association-test-double-only", "qualification_sha256": "7"*64,
        "algorithm_sha256": "a"*64, "domain": measurement._capture_domain(sample["recognition_chain"]),
        "boundary_uncertainty_mm": sample["boundary_uncertainty_mm"]}
    monkeypatch.setattr(measurement, "_measurement_profile", lambda: deepcopy(profile))
    original = globals()["measured"]
    def test_measurement(*args, **kwargs):
        observation = original(*args, **kwargs)
        observation["boundary_uncertainty_provenance"].update(profile_sha256=geometry_digest(profile),
            method_id=profile["method_id"], qualification_sha256=profile["qualification_sha256"],
            capture_profile=None, domain_sha256=geometry_digest(profile["domain"]))
        return observation
    monkeypatch.setitem(globals(), "measured", test_measurement)


def detection(body=None, *, xy=(0., 0.), material="Gr", image="b"):
    return FlakeDetection(dx_mm=xy[0], dy_mm=xy[1], area_um2=400,
        material=material, thickness_label="monolayer", confidence=0,
        meta=measured(body, image=image, material=material))


def associate(source, candidates, *, xy=(0., 0.), complete=True):
    return associate_source_observations(source, candidates, material="Gr",
        source_to_observation_xy_mm=xy, source_to_observation_rotation_deg=0.,
        registration_uncertainty_mm=.0001, observation_complete=complete)


def bare_frame(*, changes=None):
    width, height = 480, 320
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    rgb = np.trunc(gain[:, :, None]*np.array(SYNTHETIC_SUBSTRATE_RGB)).astype(np.uint8)
    if changes:
        changes(rgb)
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return Frame("microscope", 123., data=output.getvalue(), meta={
        "objective": 5, "focus_plane": "chip", "fov_mm": [2.6, 1.72],
        "calibration": {"source": "synthetic", "image_size_px": [width, height],
            "mm_per_pixel": [2.6/width, 1.72/height], "u_axis": "+world_x", "v_axis": "-world_y",
            "photometry_profile": {"source": "synthetic", "camera_response": CAMERA_RESPONSE_PROFILE,
                "chip_surface": "synthetic-chip-footprint-v2"},
            "observation_condition": {"observation_noise_scale": 1.}}})


def test_composition_forwards_only_declared_optical_methods_and_rgb_material():
    calls = []
    optical = SimpleNamespace(**{name: lambda *args, _name=name: calls.append(_name) or _name
        for name in ("plane_sharpness", "sharpness", "find_corner", "chip_in_view", "contact", "fringes")})
    optical.find_flakes = lambda *args: pytest.fail("legacy material fallback")
    optical.find_stamp_flakes = lambda *args: pytest.fail("legacy carried fallback")
    optical.reactor_preparation = lambda *args: pytest.fail("legacy preparation fallback")
    material = SimpleNamespace(find_flakes=lambda *args: ["rgb"],
        revisit_target=lambda *args, **kwargs: kwargs, persist_detection_masks=lambda *args: "masks")
    analyzer = RGBMicroscopeAnalyzer(optical, material)
    frame = Frame("microscope", 1.)
    assert analyzer.find_flakes(frame, "Gr") == ["rgb"]
    assert analyzer.revisit_target(frame, "Gr", accepted_labels=["monolayer"]) == {"accepted_labels": ["monolayer"]}
    assert analyzer.persist_detection_masks(frame, None) == "masks"
    assert analyzer.plane_sharpness(frame, "chip") == "plane_sharpness"
    # Focus quality and the chip corner come from the pixels (v1 algorithms), never the provider.
    for method in (analyzer.sharpness, analyzer.find_corner):
        with pytest.raises(ValueError, match="raster pixels"):
            method(frame)
    assert calls == ["plane_sharpness"]
    assert not hasattr(analyzer, "reactor_preparation")


def test_visible_cell_support_preserves_foreground_and_clear_hole_without_annotations():
    def changes(rgb):
        rgb[100:105, 100:106] = [65, 44, 85]
        # One original cell remains bare inside the surrounding unknown region.
        rgb[102, 102] = [120, 94, 166]  # exact original centre-near baseline below
    frame = bare_frame(changes=changes)
    # Restore one exact substrate cell from a separate bare exposure.
    bare = bare_frame()
    with Image.open(io.BytesIO(bare.data)) as image:
        original = np.asarray(image)
    with Image.open(io.BytesIO(frame.data)) as image:
        pixels = np.array(image)
    pixels[102, 102] = original[102, 102]
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    frame.data = output.getvalue()
    class Forbidden:
        def __iter__(self):
            pytest.fail("material annotations read")
    frame.meta.update(flakes=Forbidden(), stamp_flakes=Forbidden(), on_chip=Forbidden())
    observed = rgb_discovery_obstacles(frame, plane_quality=10.)
    assert observed["status"] == "supported" and observed["obstacles"] == []
    assert [102, 100, 102] in observed["unknown_runs_px"]
    assert [102, 103, 106] in observed["unknown_runs_px"]
    assert not any(row == 102 and start <= 102 < stop for row, start, stop in observed["unknown_runs_px"])
    assert observed["boundary_cell_radius_mm"] > 0
    assert "centroid_sigma_mm" not in observed and "chip_id" not in observed


def test_exact_achromatic_noise_support_keeps_saturation_and_wrong_colour_unknown():
    def changes(rgb):
        rgb[100, 100] += 2
        rgb[100, 101] += np.array([1, 2, 1], dtype=np.uint8)
        rgb[100, 102] = [255, 255, 255]
    observed = rgb_discovery_obstacles(bare_frame(changes=changes), plane_quality=10.)
    assert observed["unknown_runs_px"] == [[100, 101, 103]]


@pytest.mark.parametrize("field", ["camera", "focus", "quality", "noise", "replay"])
def test_unsupported_capture_never_certifies_background(field):
    frame, quality = bare_frame(), 10.
    if field == "camera":
        frame.meta["calibration"]["photometry_profile"].pop("camera_response")
    elif field == "focus":
        frame.meta["focus_plane"] = "stamp"
    elif field == "quality":
        quality = 8.
    elif field == "noise":
        frame.meta["calibration"]["observation_condition"]["observation_noise_scale"] = 2.
    else:
        frame.meta["replayed_from"] = "old.png"
    observed = rgb_discovery_obstacles(frame, plane_quality=quality)
    assert observed["status"] == "unsupported"
    assert observed["conditional_visible_completeness"] is False
    assert observed["supported_roi_px"] == [0, 0, 0, 0]


def test_stamp_pixels_cannot_count_one_region_as_two_materials():
    material = SimpleNamespace(find_flakes=lambda frame, name: [detection(material=name)])
    analyzer = RGBMicroscopeAnalyzer(SimpleNamespace(), material)
    frame = Frame("microscope", 1., meta={"focus_plane": "stamp", "stamp_flakes": [{"truth": "ignored"}]})
    found = analyzer.find_stamp_flakes(frame)
    assert len(found) == 2
    assert all(d.meta["material_identity_status"] == "ambiguous_cross_material_pixels" for d in found)
    assert all(d.meta["carried_observation_supported"] is False for d in found)
    assert associate(measured(), found)["status"] == "unknown"


def test_stamp_classification_error_cannot_become_an_empty_stamp():
    def broken(frame, material):
        if material == "hBN":
            raise ValueError("unresolved background")
        return []
    analyzer = RGBMicroscopeAnalyzer(SimpleNamespace(), SimpleNamespace(find_flakes=broken))
    with pytest.raises(ValueError, match="unresolved background"):
        analyzer.find_stamp_flakes(Frame("microscope", 1., meta={"focus_plane": "stamp"}))
    # An original source's public catalog class can explicitly constrain a
    # selected-source revisit; no hidden stamp annotation chooses that class.
    assert analyzer.find_stamp_flakes(Frame("microscope", 1., meta={"focus_plane": "stamp"}), materials=("Gr",)) == []


def test_registered_resampling_accepts_whole_body_without_equal_vertices():
    # A collinear extra boundary vertex is a different raster tessellation, not
    # a changed body. Pose comes from the supplied registration only.
    current = region([[-.01, -.01], [0., -.01], [.01, -.01], [.01, .01], [-.01, .01]])
    source = measured()
    original = deepcopy(source)
    result = associate(source, [detection(current, xy=(.004, -.003))], xy=(.004, -.003))
    assert result["status"] == "unique"
    assert source == original
    assert result["matches"][0]["combined_bound_mm"] == pytest.approx(.0003)


def test_same_material_duplicate_and_partial_body_remain_unresolved():
    assert associate(measured(), [detection(), detection(image="c")])["status"] == "ambiguous"
    partial = region([[-.01, -.01], [0., -.01], [0., .01], [-.01, .01]])
    result = associate(measured(), [detection(partial)])
    assert result["status"] == "unknown" and result["confounding_detection_indices"] == [0]


def test_filled_source_hole_is_not_a_whole_body_match():
    source = measured(region(holes=[[[-.004, -.004], [-.004, .004], [.004, .004], [.004, -.004]]]))
    result = associate(source, [detection()])
    assert result["status"] == "unknown" and not result["matches"]


def test_absence_is_separate_from_pickup_and_requires_observation_completeness():
    assert associate(measured(), [], complete=False)["status"] == "unknown"
    absent = associate(measured(), [])
    assert absent["status"] == "absent"
    assert "transfer_confirmed" not in absent and "picked_up" not in absent
    unbound = detection()
    unbound.meta["boundary_uncertainty_provenance"]["image_sha256"] = "f"*64
    assert associate(measured(), [unbound])["status"] == "unknown"


@pytest.mark.parametrize("change", ["source_empty", "current_empty", "source_material", "current_material", "source100"])
def test_incompatible_source_identity_or_empty_body_cannot_be_called_absent(change):
    source, current = measured(), detection()
    if change == "source_empty":
        source["region_mm"] = {"version": "polygon-region-v1", "components": []}
    elif change == "current_empty":
        current.meta["region_mm"] = {"version": "polygon-region-v1", "components": []}
    elif change == "source_material":
        source["recognition_chain"]["material"] = "hBN"
    elif change == "current_material":
        current.meta["recognition_chain"]["material"] = "hBN"
    else:
        source["recognition_chain"]["image"]["objective"] = 100
    assert associate(source, [current])["status"] == "unknown"


def discovery_maps(*, unknown=False):
    policy = discovery_policy()
    maps = []
    for index, center in enumerate(policy["map_offsets_mm"]):
        def changes(rgb, index=index):
            rgb[:] = (rgb.astype(np.int16)+index-1).astype(np.uint8)
            if unknown and index == 0:
                rgb[150:160, 230:240] = [65, 44, 85]
        frame = bare_frame(changes=changes)
        frame.meta["calibration"]["stamp_discovery"] = policy
        maps.append({"observation": rgb_discovery_obstacles(frame, plane_quality=10.),
            "camera_center_mm": center})
    return policy, maps


def clear_domain(policy, maps):
    return choose_clear_domain(policy, maps, nominal_center=[0., 0.],
        chip_polygon=[[-5., -5.], [5., -5.], [5., 5.], [-5., 5.]],
        chip_id="public_chip_record", exact_center=True)


def test_four_original_pixel_maps_admit_only_observed_conditional_clear_domain():
    policy, maps = discovery_maps()
    result = clear_domain(policy, maps)
    assert result["clear_radius_mm"] == policy["clear_radius_mm"]
    assert result["observed_exclusion_radius_mm"] > result["clear_radius_mm"]
    assert result["obstacle_position_model"] == "original_unknown_pixel_cells_no_centroid_noise"
    assert "invisible/subresolution" in result["observation_scope"]
    assert all("centroid_sigma_mm" not in item["observation"] for item in maps)


def test_one_unknown_material_cell_run_refuses_clear_domain_instead_of_using_outline():
    policy, maps = discovery_maps(unknown=True)
    with pytest.raises(ValueError, match="No fixed observed RGB"):
        clear_domain(policy, maps)


@pytest.mark.parametrize("change", ["support", "runs", "duplicate", "sigma", "policy"])
def test_pixel_clear_domain_rejects_changed_or_incomplete_observation(change):
    policy, maps = discovery_maps()
    observed = maps[0]["observation"]
    if change == "support":
        observed["conditional_visible_completeness"] = False
    elif change == "runs":
        observed["unknown_runs_px"] = [[100, 10, 15], [100, 14, 20]]
    elif change == "duplicate":
        observed["image_sha256"] = maps[1]["observation"]["image_sha256"]
    elif change == "sigma":
        observed["centroid_sigma_mm"] = .001
    else:
        observed["observation_policy"]["unknown_rule"] = "ignore_small_material"
    with pytest.raises(ValueError, match="Discovery RGB"):
        clear_domain(policy, maps)
