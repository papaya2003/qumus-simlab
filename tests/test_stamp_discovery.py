"""R017 pure policy/observation controls; no World, services or device jobs."""

import hashlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from qumus_simlab.config import simulation_profile, with_stamp_discovery
from qumus_simlab.core.polygons import cap_surface_point
from qumus_simlab.devices.base import Frame
from qumus_simlab.records.discovery import (
    PHYSICS_PROFILE,
    discovery_policy,
    map_centre,
    settling_envelope_mm,
    validate_policy,
)
from qumus_simlab.testbench.locator import SimMicroscopeAnalyzer
from qumus_simlab.testbench.material_adapter import MODEL_VERSION


def test_fresh_linked_only_profile_preserves_white_bands_and_legacy_mechanics():
    old = simulation_profile("hallbar")
    new = with_stamp_discovery(simulation_profile("hallbar-linked"))
    assert old["stamp"]["dome_radius_mm"] == 1.5
    assert "discovery" not in old["stamp"]
    assert new["stamp"]["dome_radius_mm"] == 12
    assert new["stamp"]["cap_support_radius_mm"] == 5
    assert new["stamp"]["physics_profile"] == PHYSICS_PROFILE
    # Jialei 2026-11-03: the linked Hall-bar SIM has zero stamp drift and no
    # contact twist; every other dynamics key keeps the legacy value.
    steady = {"lateral_drift_mm_s": [1e-8, -1e-8], "approach_drift_mm_per_mm": [0.0, 0.0],
              "contact_slip_deg_s": 0}
    assert new["stamp"]["dynamics"] == {**old["stamp"]["dynamics"], **steady}
    assert new["stamp"]["dynamics"]["contact_front_tau_s"] == 0.8
    from qumus_simlab.testbench.optics import _RGB_BANDS_NM

    assert _RGB_BANDS_NM == ((610.0, 45.0), (540.0, 40.0), (460.0, 35.0))


def test_fixed_envelope_covers_complete_settling_not_only_early_patch():
    result = settling_envelope_mm(discovery_policy())
    assert result["settled_front_radius_mm"] == pytest.approx(0.3550937604535954)
    assert result["minimum_clear_radius_mm"] == pytest.approx(1.0893208703650599)
    assert result["admitted"]
    assert result["raster_radial_guard_mm"] == pytest.approx(2**0.5 * 0.065)
    assert result["relative_stage_position_radius_mm"] > 0.022
    assert not settling_envelope_mm(discovery_policy(), max_command_mm=0.00975)["admitted"]
    bad = discovery_policy()
    bad["clear_radius_mm"] = 0.9
    with pytest.raises(ValueError):
        validate_policy(bad)


@pytest.mark.parametrize("point", [(0.1, 0.1), (0.1, 9.9), (9.9, 0.1), (9.9, 9.9), (5.0, 5.0)])
def test_fixed_proposal_allows_both_axes_at_corners(point):
    dx, dy = map_centre(point, (10.0, 10.0), 0.0)
    target = (point[0] + dx, point[1] + dy)
    assert all(1.2 <= value <= 8.8 for value in target)
    assert (dx * dx + dy * dy) ** 0.5 >= 1.5 - 1e-12
    if point == (0.1, 0.1):
        assert dx > 0 and dy > 0


def test_cap_eligibility_uses_five_mm_footprint_not_twelve_mm_curvature():
    assert cap_surface_point((0.0, 0.0), (20.0, 0.0), (5.0, 5.0), 5.0) is None
    assert cap_surface_point((0.0, 0.0), (0.0, 0.0), (5.0, 5.0), 5.0) == (0.0, 0.0)
    point = cap_surface_point((8.0, 0.0), (0.0, 0.0), (10.0, 10.0), 5.0)
    assert point == (5.0, 0.0)
    # Circle-edge intersection remains available when naive nearest-rectangle
    # point is outside the cap, rather than inventing a twelve-mm branch.
    assert cap_surface_point((7.0, 4.0), (0.0, 0.0), (4.0, 4.0), 5.0) is not None


def observed_frame():
    image = Image.new("RGB", (480, 320), (122, 96, 168))
    output = io.BytesIO()
    image.save(output, format="PNG")
    polys = [
        {
            "material": m,
            "outline_mm": [
                [x - 0.02, -0.02],
                [x + 0.02, -0.02],
                [x + 0.02, 0.02],
                [x - 0.02, 0.02],
            ],
            "geometry_source": MODEL_VERSION,
            "area_um2": 1600.0,
            "visible_area_um2": 1600.0,
        }
        for m, x in [("hBN", -0.3), ("Gr", 0.3)]
    ]
    from qumus_simlab.testbench.render import microscope_photometry_profile

    return Frame(
        "microscope",
        12.0,
        data=output.getvalue(),
        meta={
            "objective": 5,
            "sharpness_chip": 8.0,
            "on_chip": "source",
            "flakes": polys,
            "fov_mm": [2.6, 1.72],
            "calibration": {
                "source": "synthetic",
                "image_size_px": [480, 320],
                "stamp_discovery": discovery_policy(),
                "mm_per_pixel": [2.6 / 480, 1.72 / 320],
                "u_axis": "+world_x",
                "v_axis": "-world_y",
                "photometry_profile": microscope_photometry_profile(),
            },
        },
    )


def observation(frame):
    analyzer = SimMicroscopeAnalyzer.__new__(SimMicroscopeAnalyzer)
    analyzer.cfg = {}
    # A deliberately unusable World object proves this boundary does not query
    # live state/randomness instead of the frozen exposure.
    analyzer.world = SimpleNamespace()
    return analyzer.discovery_obstacles(frame)


def test_mixed_material_one_exact_exposure_observation_is_stable():
    frame = observed_frame()
    result = observation(frame)
    assert {p["material"] for p in result["obstacles"]} == {"hBN", "Gr"}
    assert result["conditional_visible_completeness"] and result["unknown_outside_roi"]
    assert result["image_sha256"] == hashlib.sha256(frame.data).hexdigest()
    assert observation(frame) == result


@pytest.mark.parametrize(
    "fault", ["blur", "clipped", "missing_profile", "unknown", "stamp", "saturated"]
)
def test_unknown_or_unsupported_map_cannot_be_certified_background(fault):
    frame = observed_frame()
    if fault == "blur":
        frame.meta["sharpness_chip"] = 5.99
    elif fault == "clipped":
        frame.meta["flakes"][0]["visible_area_um2"] = 1000
    elif fault == "missing_profile":
        frame.meta["calibration"].pop("stamp_discovery")
    elif fault == "unknown":
        frame.meta["flakes"][0]["geometry_source"] = None
    elif fault == "stamp":
        frame.meta["stamp"] = {"film": True}
    else:
        image = Image.new("RGB", (480, 320), (122, 96, 168))
        image.putpixel((240, 160), (255, 0, 0))
        out = io.BytesIO()
        image.save(out, format="PNG")
        frame.data = out.getvalue()
    with pytest.raises(ValueError):
        observation(frame)


def test_frame_edge_flake_is_its_in_frame_part():
    """A flake cut by the 5x map edge (seed 10, 4ff2e59) is an obstacle up to
    that edge; the clear disk must stay inside the maps' supported rectangles."""
    from qumus_simlab.core.polygons import area_um2, intersection, rectangle

    frame = observed_frame()
    edge = [[1.28, -0.02], [1.32, -0.02], [1.32, 0.02], [1.28, 0.02]]
    inside = intersection([tuple(p) for p in edge], rectangle(2.6, 1.72))
    frame.meta["flakes"][1].update(outline_mm=edge, visible_area_um2=area_um2(inside))
    result = observation(frame)
    clipped = [o for o in result["obstacles"] if o["material"] == "Gr"]
    assert len(clipped) == 1 and clipped[0]["complete"] is False and clipped[0]["clipped_at"] == "frame_edge"
    assert max(x for x, _ in clipped[0]["outline_mm"]) <= 1.3 + .01
    assert next(o for o in result["obstacles"] if o["material"] == "hBN")["complete"] is True


def test_quieter_stage_stays_inside_the_declared_stage_bound():
    from qumus_simlab.records.discovery import validate_generator_config

    config = with_stamp_discovery(simulation_profile("hallbar-linked"))
    config["stage"]["sigma_mm"] = 0.0
    assert validate_generator_config(config) == discovery_policy()
    config["stage"]["sigma_mm"] = -0.001
    with pytest.raises(ValueError):
        validate_generator_config(config)


@pytest.mark.parametrize("key,steady,faster", [
    ("lateral_drift_mm_s", [0.0, 0.0], [0.0, -0.00002]),
    ("approach_drift_mm_per_mm", [0.0, 0.0], [-0.003, -0.002]),
])
def test_steadier_stamp_stays_inside_the_declared_drift_bound(key, steady, faster):
    from qumus_simlab.records.discovery import validate_generator_config

    config = with_stamp_discovery(simulation_profile("hallbar-linked"))
    config["stamp"]["dynamics"][key] = steady
    assert validate_generator_config(config) == discovery_policy()
    config["stamp"]["dynamics"][key] = faster  # larger or reversed drift is not inside the bound
    with pytest.raises(ValueError):
        validate_generator_config(config)


@pytest.mark.parametrize(
    "fault",
    ["curvature", "tau", "drift", "tilt", "apex_prior", "noise", "stage_noise", "focus_speed"],
)
def test_declared_policy_cannot_mask_changed_actual_generators(fault):
    from qumus_simlab.records.discovery import validate_generator_config

    config = with_stamp_discovery(simulation_profile("hallbar-linked"))
    assert validate_generator_config(config) == discovery_policy()
    if fault == "curvature":
        config["stamp"]["dome_radius_mm"] = 15
    elif fault == "tau":
        config["stamp"]["dynamics"]["contact_front_tau_s"] = 0.2
    elif fault == "drift":
        config["stamp"]["dynamics"]["lateral_drift_mm_s"] = [0.00004, -0.00001]
    elif fault == "tilt":
        config["microscope"]["focus_tilt_per_mm"] = 0.02
    elif fault == "apex_prior":
        config["stamp"]["apex_sigma_mm"] = 0.10
    elif fault == "noise":
        config["microscope"]["observation_noise_scale"] = 2.0
    elif fault == "stage_noise":
        config["stage"]["sigma_mm"] = 0.004
    else:
        config["motion"]["focus_speed_mm_s"] = 0.5
    with pytest.raises(ValueError):
        validate_generator_config(config)
