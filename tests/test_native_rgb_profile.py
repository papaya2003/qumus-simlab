"""Native material image formation; no World, jobs, devices or model calls."""
from __future__ import annotations

import io
import math
from copy import deepcopy

import cv2
import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.optical_profiles import (
    V1_MATERIAL_CAPTURE_PROFILE,
    V1_MATERIAL_PITCH_UM,
    V1_MATERIAL_SENSOR_PX,
    V1_NATIVE_SUBSTRATE_RGB,
    V1_SEARCH_CAMERA_PROFILE,
    native_substrate_support,
    v1_material_reference,
)
from qumus_simlab.devices.base import Frame
from qumus_simlab.perception._v1_rgb_core import Segmenter
from qumus_simlab.perception._v1_rgb_revisit import CenteredRevisitRecovery, _private_kmeans
from qumus_simlab.perception.rgb_material import (
    RGBMaterialAnalyzer,
    RGBRecognition,
    material_policy,
    segment_rgb,
)
from qumus_simlab.testbench.render import Canvas, MicroscopeRenderer, _camera, _v1_material_tint


@pytest.mark.parametrize("objective,pitch_um", [(10, .365), (20, .182), (100, .036)])
def test_native_and_focus_rasters_have_one_physical_field_with_honest_pixel_pitch(objective, pitch_um):
    renderer = MicroscopeRenderer(None, material_capture_profile=V1_MATERIAL_CAPTURE_PROFILE)
    native = renderer.exposure_geometry(objective, [1.3, .86])
    focus = renderer.exposure_geometry(objective, [1.3, .86], "focus_capture")
    preview = renderer.exposure_geometry(objective, [1.3, .86], "live_preview")
    assert native["image_size_px"] == list(V1_MATERIAL_SENSOR_PX)
    assert native["fov_mm"] == pytest.approx([axis*pitch_um/1000 for axis in V1_MATERIAL_SENSOR_PX])
    assert native["mm_per_pixel"] == pytest.approx([pitch_um/1000]*2)
    assert focus == preview
    assert focus["image_size_px"] == [480, 320]
    assert focus["fov_mm"] == native["fov_mm"]
    assert focus["mm_per_pixel"] == pytest.approx([native["fov_mm"][0]/480, native["fov_mm"][1]/320])


def test_native_profile_is_explicit_and_does_not_relabel_historical_rasters():
    renderer = MicroscopeRenderer(None)
    old = renderer.exposure_geometry(20, [.65, .43])
    assert old["image_size_px"] == [480, 320]
    assert old["fov_mm"] == [.65, .43]
    assert old["material_capture_profile"] is None
    assert 5 not in V1_MATERIAL_PITCH_UM  # No invented measured native5 calibration.
    with pytest.raises(ValueError, match="Unknown material"):
        MicroscopeRenderer(None, material_capture_profile="unqualified")


@pytest.mark.parametrize("material,layers", [("Gr", 1), ("Gr", 2), ("Gr", 3), ("Gr", 6),
                                             ("Gr", 20), ("hBN", 15), ("hBN", 45),
                                             ("hBN", 75), ("hBN", 105), ("hBN", 150)])
@pytest.mark.parametrize("objective", [10, 20, 100])
def test_forward_palette_uses_unchanged_empirical_donor_references(material, layers, objective):
    reference, background = v1_material_reference(material, layers, objective)
    selected, _ = material_policy(material, objective)
    profile = 100 if material == "Gr" and objective == 20 and layers <= 2 else objective
    assert tuple(reference) in selected.layer_labels_for_mag(profile)
    assert tuple(background) == tuple(selected.target_bg_lab_for_mag(objective))


def test_private_recovery_clustering_is_repeatable_and_preserves_numpy_and_opencv_rng():
    samples = np.asarray([[1., 2., 3.]]*20+[[30., 40., 50.]]*20, dtype=np.float32)
    np_before = np.random.get_state()
    cv2.setRNGSeed(733)
    expected = np.empty((8, 3), dtype=np.float32)
    cv2.randn(expected, 0, 1)
    cv2.setRNGSeed(733)
    first = _private_kmeans(samples, 2)
    actual = np.empty_like(expected)
    cv2.randn(actual, 0, 1)
    second = _private_kmeans(samples, 2)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(map(tuple, first[1])) == {(1., 2., 3.), (30., 40., 50.)}
    assert np.array_equal(actual, expected)
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1]) and np_before[2:] == np_after[2:]


def rgb_from_lab(lab):
    return np.rint(cv2.cvtColor(np.asarray([[lab]], dtype=np.float32), cv2.COLOR_LAB2RGB)[0, 0]*255).astype(np.uint8)


def native_camera_fixture(width=3840, height=2160, *, rectangle=None, noise_scale=1., layers=15):
    canvas = Canvas(width, height, V1_NATIVE_SUBSTRATE_RGB)
    if rectangle is not None:
        canvas.rect(*rectangle, _v1_material_tint("hBN", layers, 100))
    _camera(canvas, 733, noise_scale, V1_SEARCH_CAMERA_PROFILE)
    return np.frombuffer(canvas.px, dtype=np.uint8).reshape(height, width, 3).copy()


def native_arguments(noise_scale=1.):
    return {"capture_profile": V1_MATERIAL_CAPTURE_PROFILE, "camera_response": V1_SEARCH_CAMERA_PROFILE,
        "observation_condition": {"observation_noise_scale": noise_scale}}


def test_isolated_thin_interior_is_not_background_despite_donor_colour_radius():
    material, _ = material_policy("hBN", 100)
    image = native_camera_fixture(rectangle=(1860, 1030, 1980, 1130))
    historical = Segmenter(image, material, magnification=100, surround_bg=True)
    native = Segmenter(image, material, magnification=100, surround_bg=True,
        background_support="native_border_and_joint_foreground_v1", observation_noise_scale=1.)
    historical.make_masks()
    native.make_masks()
    center = (1080+40, 1920+40)
    assert historical.bg_mask[center]  # The old <=20 LAB rule swallows this anchor.
    assert not native.bg_mask[center]
    assert native.bg_mask[70, 70]  # Actual substrate beyond the finite operator halo.
    assert all(o["edge_support_band_px"] == o["operator_radius_px"]+1 for o in native.background_support_observations)
    native.lab_equalize()
    assert np.linalg.norm(native.lab[1080, 1920]-material.target_bg_lab_for_mag(100)) > 8


def test_joint_operator_foreground_cannot_be_used_as_normalization_background(monkeypatch):
    # Isolate normalization topology with the adversarial assumption that every
    # original pixel is camera-compatible; the independent camera guard is
    # exercised by the actual-raster tests below.
    monkeypatch.setattr("qumus_simlab.core.optical_profiles.native_substrate_support",
        lambda pixels, noise: np.ones(pixels.shape[:2], dtype=bool))
    material, _ = material_policy("Gr", 20)
    image = np.tile(rgb_from_lab(material.target_bg_lab_for_mag(20)), (160, 240, 1))
    image[50:110, 80:160] = rgb_from_lab((40.4, 23.6, -.6))
    def no_edge(segmenter):
        return np.zeros(segmenter.lab.shape[:2], dtype=bool)
    def source_edge(segmenter):
        edge = no_edge(segmenter)
        edge[90, 120:200] = edge[149, 120:200] = True
        edge[90:150, 120] = edge[90:150, 199] = True
        return edge
    no_edge.args = source_edge.args = [1, 0, 1, 1]
    for policy, expected_background in (("historical_v1", True), ("native_border_and_joint_foreground_v1", False)):
        segmenter = Segmenter(image, material, magnification=20, surround_bg=True, background_support=policy,
            observation_noise_scale=1.)
        segmenter.edge_methods = [no_edge, source_edge]
        segmenter.make_masks()
        assert bool(segmenter.full_template[120, 160])
        assert bool(segmenter.bg_mask[120, 160]) is expected_background


@pytest.mark.parametrize("support_pixels", [0, 20])
def test_native_insufficient_background_fails_before_sampling(support_pixels):
    material, _ = material_policy("hBN", 100)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    segmenter = Segmenter(image, material, magnification=100, surround_bg=True,
        background_support="native_border_and_joint_foreground_v1")
    segmenter.bg_mask = np.zeros(segmenter.lab.shape[:2], dtype=bool)
    segmenter.bg_mask[40:42, 40:40+support_pixels] = True
    with pytest.raises(ValueError, match="insufficient two-dimensional"):
        segmenter.lab_equalize()
    assert segmenter.rng.draws == 0


def test_uniform_native_background_has_no_centered_material_candidate():
    image = native_camera_fixture()
    with pytest.raises(ValueError, match="No centered RGB"):
        segment_rgb(image, "hBN", objective=100, revisit=True, **native_arguments())


@pytest.mark.parametrize("top", [0, 8])
def test_full_or_edge_spanning_thin_foreground_is_unknown_not_substrate(top):
    rectangle = (0, top, 3840, 2160-top)
    image = native_camera_fixture(rectangle=rectangle)
    assert not native_substrate_support(image, 1.)[1080, 1920]
    with pytest.raises(ValueError, match="unambiguous frame/camera support"):
        segment_rgb(image, "hBN", objective=100, revisit=True, **native_arguments())


@pytest.mark.parametrize("rectangle", [(0, 0, 2600, 2160), (0, 200, 2400, 2000)])
def test_centred_flake_covering_a_whole_frame_side_is_classified(rectangle):
    # Seed 8: at 100X the largest hBN covers the left side of the field. v1
    # reads the substrate at the frame edge as background; so does native.
    image = native_camera_fixture(rectangle=rectangle, layers=60)
    [candidate] = segment_rgb(image, "hBN", objective=100, revisit=True, **native_arguments()).candidates
    assert candidate["area_px"] > .9*(rectangle[2]-rectangle[0])*(rectangle[3]-rectangle[1])


def test_near_edge_interior_thin_material_never_enters_background_support():
    material, _ = material_policy("hBN", 100)
    image = native_camera_fixture(rectangle=(3, 900, 200, 1260))
    segmenter = Segmenter(image, material, magnification=100, surround_bg=True,
        background_support="native_border_and_joint_foreground_v1", observation_noise_scale=1.)
    segmenter.make_masks()
    assert not segmenter.bg_mask[1080+40, 60+40]
    assert not native_substrate_support(image, 1.)[1080, 60]


@pytest.mark.parametrize("noise_scale", [None, 1.8, float("nan")])
def test_unknown_native_noise_cannot_be_bypassed_by_graphene_recovery(noise_scale):
    image = native_camera_fixture()
    with pytest.raises(ValueError, match="camera/observation condition is unsupported"):
        segment_rgb(image, "Gr", objective=100, revisit=True, **native_arguments(noise_scale))


@pytest.mark.parametrize("noise_scale", [0., .375, 1.])
def test_shared_camera_support_retains_actual_substrate_and_excludes_coloured_interior(noise_scale):
    image = native_camera_fixture(320, 240, noise_scale=noise_scale)
    assert native_substrate_support(image, noise_scale).all()
    coloured = native_camera_fixture(320, 240, rectangle=(100, 70, 220, 170), noise_scale=noise_scale)
    support = native_substrate_support(coloured, noise_scale)
    assert not support[120, 160] and support[30, 30]


def test_small_raster_cannot_claim_the_native_material_profile():
    image = native_camera_fixture(320, 240)
    with pytest.raises(ValueError, match="actual native analysis raster"):
        segment_rgb(image, "hBN", objective=100, revisit=True, **native_arguments())


def test_high_mag_recovery_keeps_roi_truncation_explicit_inside_full_frame():
    material, _ = material_policy("Gr", 100)
    image = np.tile(rgb_from_lab(material.target_bg_lab_for_mag(100)), (300, 400, 1))
    image[60:240, 50:200] = rgb_from_lab((46., 16.2, .4))
    recovery = CenteredRevisitRecovery(material, "Gr", 100, 2000, 10_000_000)
    result = recovery._center_color_fallback(image)
    assert result is not None
    mask, candidate = result
    assert mask.any() and not mask[:, :80].any()  # Lost material is inside the full frame.
    support = candidate["support"]
    assert support["support_roi_px"] == [80, 0, 320, 300]
    assert support["whole_body_status"] == "unsupported"
    assert support["whole_body_reason"] == "centered_recovery_roi_clipped_or_unobserved"
    assert support["operator_guard_native_px"] == 3


@pytest.fixture(scope="module")
def native_geometry_bytes():
    encoded = io.BytesIO()
    Image.new("RGB", V1_MATERIAL_SENSOR_PX, V1_NATIVE_SUBSTRATE_RGB).save(encoded, format="PNG")
    return encoded.getvalue()


def geometry_frame(data, objective=20):
    renderer = MicroscopeRenderer(None, material_capture_profile=V1_MATERIAL_CAPTURE_PROFILE)
    geometry = renderer.exposure_geometry(objective, [.65, .43])
    calibration = {**geometry, "u_axis": "+world_x", "v_axis": "-world_y",
        "observation_condition": {"observation_noise_scale": 1.},
        "photometry_profile": renderer.photometry_profile}
    return Frame("fixture", 0., data=data, meta={"objective": objective,
        "fov_mm": geometry["fov_mm"], "calibration": calibration})


@pytest.mark.parametrize("objective", [20, 100])
def test_native_analyzer_geometry_guard_allows_actual_sensor_roundoff(native_geometry_bytes, monkeypatch, objective):
    frame = geometry_frame(native_geometry_bytes, objective)
    def empty_result(rgb, material, **kwargs):
        selected, policy = material_policy(material, kwargs["objective"])
        return RGBRecognition(np.zeros(rgb.shape[:2], dtype=np.uint16), (), policy, kwargs["image_sha256"], kwargs["objective"])
    monkeypatch.setattr("qumus_simlab.perception.rgb_material.segment_rgb", empty_result)
    assert RGBMaterialAnalyzer().find_flakes(frame, "hBN") == []


@pytest.mark.parametrize("change", ["pitch", "pitch_and_fov", "missing_fov", "objective", "axes",
                                  "missing_pitch", "calibration_fov", "raster"])
def test_native_analyzer_rejects_metadata_only_rescaling_before_segmentation(native_geometry_bytes, monkeypatch, change):
    frame = geometry_frame(native_geometry_bytes)
    frame.meta = deepcopy(frame.meta)
    calibration = frame.meta["calibration"]
    if change in ("pitch", "pitch_and_fov"):
        calibration["mm_per_pixel"] = [v*2 for v in calibration["mm_per_pixel"]]
    if change == "pitch_and_fov":
        frame.meta["fov_mm"] = [v*2 for v in frame.meta["fov_mm"]]
        calibration["fov_mm"] = frame.meta["fov_mm"]
    if change == "missing_fov":
        frame.meta.pop("fov_mm")
    if change == "objective":
        frame.meta["objective"] = 50
    if change == "axes":
        calibration["v_axis"] = "+world_y"
    if change == "missing_pitch":
        calibration.pop("mm_per_pixel")
    if change == "calibration_fov":
        calibration["fov_mm"] = [1., 1.]
    if change == "raster":
        calibration["image_size_px"] = [480, 320]
    def must_not_segment(*args, **kwargs):
        raise AssertionError("Invalid native geometry reached material segmentation")
    monkeypatch.setattr("qumus_simlab.perception.rgb_material.segment_rgb", must_not_segment)
    with pytest.raises(ValueError, match="Native RGB capture|capture/calibration dimensions"):
        RGBMaterialAnalyzer().find_flakes(frame, "hBN")


def test_centred_capture_keeps_a_large_monolayer_and_still_lists_its_neighbours():
    # Seed 8 chip_005_f022: a 3870 um2 monolayer is 116000 px at 20X, over
    # the 100000 px scan ceiling. The pickup neighbourhood view is centred on
    # it (as v1's revisit capture is), so it takes v1's 10% revisit ceiling.
    canvas = Canvas(3840, 2160, V1_NATIVE_SUBSTRATE_RGB)
    tint = _v1_material_tint("Gr", 1, 20)
    canvas.rect(1620, 980, 2220, 1180, tint)  # 600 x 200 px, centred
    canvas.rect(2600, 1500, 2700, 1560, tint)  # a small neighbour
    _camera(canvas, 733, 1., V1_SEARCH_CAMERA_PROFILE)
    image = np.frombuffer(canvas.px, dtype=np.uint8).reshape(2160, 3840, 3).copy()
    def found(**kwargs):
        # (nearest drawn rectangle, area relative to it) per candidate
        drawn = {"large": ([1920, 1080], 120000), "small": ([2650, 1530], 6000)}
        rows = []
        for c in segment_rgb(image, "Gr", objective=20, **kwargs, **native_arguments()).candidates:
            name, (centre, area) = min(drawn.items(), key=lambda kv: math.dist(kv[1][0], c["centroid_px"]))
            assert math.dist(centre, c["centroid_px"]) < 5 and c["area_px"] == pytest.approx(area, rel=.05)
            rows.append(name)
        return sorted(rows)
    assert found() == ["small"]  # scan: the large flake is over the ceiling
    assert found(centered=True) == ["large", "small"]
    assert found(revisit=True) == ["large"]  # revisit: the target only


def test_centred_20x_refind_takes_the_whole_monolayer_beside_thicker_flakes():
    # Seed 8 chip_002: a 1320 um2 monolayer with thicker flakes around it.
    # With the 10X colour at 20X, v1's colour recovery kept an 8 um2 speck.
    renderer = MicroscopeRenderer(None, material_capture_profile=V1_MATERIAL_CAPTURE_PROFILE)
    geometry = renderer.exposure_geometry(20, [1.3, .86])
    flakes = [("mono", 0., 0., 1320, 1), ("bi", .105, .006, 1028, 2), ("a", .019, .142, 596, 6),
              ("b", -.019, .076, 497, 5), ("c", -.085, -.028, 665, 5), ("d", .014, .118, 3549, 27),
              ("e", -.016, .025, 531, 31)]
    meta = {"on_chip": "c1", "objective": 20, "fov_mm": geometry["fov_mm"], "sharpness_chip": 10.5,
            "sharpness": 10.5, "observation_noise_scale": 1., "purpose": "analysis_capture",
            "flakes": [{"sim_id": i, "dx_mm": x, "dy_mm": y, "area_um2": a, "material": "Gr", "layers": n}
                       for i, x, y, a, n in flakes],
            "calibration": {**geometry, "u_axis": "+world_x", "v_axis": "-world_y",
                            "observation_condition": {"observation_noise_scale": 1.},
                            "photometry_profile": renderer.photometry_profile}}
    image = np.asarray(Image.open(io.BytesIO(MicroscopeRenderer._compose(meta, size=V1_MATERIAL_SENSOR_PX))).convert("RGB"))
    [target] = segment_rgb(image, "Gr", objective=20, revisit=True, **native_arguments()).candidates
    assert target["thickness_label"] == "monolayer"
    assert target["area_px"]*.182**2 == pytest.approx(1320, rel=.15)
    assert math.dist(target["centroid_px"], [1920, 1080]) < 20
