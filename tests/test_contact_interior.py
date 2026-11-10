"""Portable RGB controls for the declared synthetic contact-interior contract.

These generated optical fields are unit controls, not measured devices or a
source-feasibility experiment. The classifier receives only RGB images, signed
pixel calibration, a public complete outline, and the full public nm band.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.optical_profiles import (
    HBN_LABEL_NM_PER_LAYER,
    SYNTHETIC_CONTACT_PHOTOMETRY,
    SYNTHETIC_HBN_SLAB,
    flake_tint,
)
from qumus_simlab.perception.contact_coverage import InteriorCalibration, contact_interior_evidence
from qumus_simlab.perception.tracking import PixelCalibration

SIZE = (480, 320)
PIXELS = PixelCalibration(2.6/SIZE[0], -1.72/SIZE[1])
OUTLINE = ((235., 157.), (245., 157.), (245., 163.), (235., 163.))
MARGIN_MM = .1752
EMPTY = InteriorCalibration()
THIN = InteriorCalibration(labels_nm=("0-10",))


def optical_pair(*, layers=0, center=(240., 160.), radius=100.):
    """Pillow RGB exposures under the public affine photometry and gain law.

    A central uniform Gr3 optical field isolates hypotheses from source shape
    recognition. Ordinary substrate around the rim supplies its declared palette.
    The small complete body is the independently supplied outline.
    A carried layer is an image-generation control, never a classifier argument.
    """
    width, height = SIZE
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    reference = np.empty((height, width, 3), dtype=float)
    reference[:] = (125, 102, 171)
    reference[np.hypot(xx-240, yy-160) <= 60.] = flake_tint("Gr", 3)
    current = reference.copy()
    optics = SYNTHETIC_CONTACT_PHOTOMETRY
    distance = np.hypot(xx-center[0], yy-center[1])
    rim = distance <= radius+max(2., radius*.06)
    interior = distance <= radius
    current[rim] = (1-optics.front_opacity)*current[rim]+optics.front_opacity*np.array(optics.front_color)
    current[interior] = ((1-optics.interior_opacity)*current[interior]
                         +optics.interior_opacity*np.array(optics.interior_color))
    if layers:
        opacity = SYNTHETIC_HBN_SLAB.projection_weight(layers)
        tint = np.array(flake_tint("hBN", layers))
        reference = (1-opacity)*reference+opacity*tint
        current = (1-opacity)*current+opacity*tint
    def raster(values):
        return Image.fromarray(np.rint(values*gain[..., None]).astype(np.uint8), "RGB")
    return raster(reference), raster(current)


def measure(reference, current, *, optical=THIN, outline=OUTLINE, calibration=PIXELS, margin=MARGIN_MM):
    return contact_interior_evidence(reference, current, calibration, outline,
                                     margin_mm=margin, optical=optical)


def test_public_zero_to_ten_nanometres_admits_transparent_and_every_layer_1_through_30():
    expected_layers = [n for n in range(1, 100) if 0 <= n*HBN_LABEL_NM_PER_LAYER < 10]
    assert expected_layers == list(range(1, 31))  # nm labels must never become a 0–10-layer cap.
    expected = [(0., (0, 0, 0)), *[(SYNTHETIC_HBN_SLAB.projection_weight(n), flake_tint("hBN", n))
                                  for n in expected_layers]]
    assert THIN.generators() == expected
    assert EMPTY.generators() == [(0., (0, 0, 0))]
    assert THIN.generators()[-1][0] == pytest.approx(.03936697184550111)


def test_a_different_bounded_nm_band_has_its_actual_layer_generators():
    layers = [n for n in range(1, 100) if 10 <= n*HBN_LABEL_NM_PER_LAYER < 20]
    assert layers == list(range(31, 61))
    generators = InteriorCalibration(labels_nm=("10-20",)).generators()
    assert generators == [(0., (0, 0, 0)), *[(SYNTHETIC_HBN_SLAB.projection_weight(n), flake_tint("hBN", n))
                                           for n in layers]]


@pytest.mark.parametrize("layers,optical", [(0, EMPTY), (0, THIN), (1, THIN), (10, THIN), (30, THIN)])
def test_small_complete_body_inside_closed_contact_is_supported(layers, optical):
    result = measure(*optical_pair(layers=layers), optical=optical)
    assert result["covered"] is True, result
    assert result["supported_pixels"] > 0 and result["holes_filled"] is False
    assert result["minimum_outline_clearance_mm"] > result["effective_required_margin_mm"]
    guard = math.hypot(PIXELS.mm_per_pixel_x, PIXELS.mm_per_pixel_y)
    assert result["required_margin_mm"] == MARGIN_MM
    assert result["registration_guard_mm"] == pytest.approx(guard)
    assert result["effective_required_margin_mm"] == pytest.approx(MARGIN_MM+guard)
    assert result["coverage_is_transfer_proof"] is False
    assert result["camera_bounds"]["gain_ratio_max"] > 1.
    assert len(result["supported_mask_sha256"]) == 64
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("optical", [EMPTY, THIN])
def test_unchanged_images_never_supply_contact_interior(optical):
    reference, _ = optical_pair()
    result = measure(reference, reference.copy(), optical=optical)
    assert result["covered"] is False and result["supported_pixels"] == 0
    assert result["holes_filled"] is False


def test_one_unchanged_pixel_pocket_inside_the_body_is_not_filled():
    reference, current = optical_pair(layers=30)
    current.putpixel((240, 160), reference.getpixel((240, 160)))
    result = measure(reference, current)
    assert result["covered"] is False and result["holes_filled"] is False
    assert result["unknown_inside_front_pixels"] > 0
    assert result["ambiguous_contact_unchanged_pixels"]+result["model_unsupported_pixels"] > 0


def test_a_visible_dark_crack_does_not_become_a_filled_contact_disc():
    reference, current = optical_pair()
    from PIL import ImageDraw
    ImageDraw.Draw(current).line((228, 160, 252, 160), fill=(30, 22, 40), width=1)
    result = measure(reference, current)
    assert result["covered"] is False and result["holes_filled"] is False


@pytest.mark.parametrize("direction", [(1, 0), (0, 1), (1, 1), (1, -1)])
def test_fractional_registration_mixture_is_in_the_full_unchanged_neighbor_hull(direction):
    reference, current = optical_pair()
    # No single reference sample equals this current RGB/chroma. The half-pixel
    # mixture is unchanged, while it also fits contact at the centre sample.
    # The unknown cell lies inside the source body, which needs strictly
    # supported cells; the margin band admits such contact-consistent cells (#136).
    x, y = 240, 160
    def gain(px, py):
        return (1-.14*((px-SIZE[0]/2)/(SIZE[0]/2))**2)*(1-.14*((py-SIZE[1]/2)/(SIZE[1]/2))**2)
    centre_gain = gain(x, y)
    reference.putpixel((x, y), tuple(round(value*centre_gain) for value in (128, 108, 173)))
    corrected = []
    for sign, color in ((-1, (100, 80, 140)), (1, (158, 154, 182))):
        px, py = x+sign*direction[0], y+sign*direction[1]
        neighbor_gain = gain(px, py)
        rgb = tuple(round(value*neighbor_gain) for value in color)
        reference.putpixel((px, py), rgb)
        corrected.append(np.array(rgb)*centre_gain/neighbor_gain)
    current.putpixel((x, y), tuple(round(value) for value in (corrected[0]+corrected[1])/2))
    result = measure(reference, current)
    assert result["covered"] is False and result["holes_filled"] is False
    assert result["ambiguous_contact_unchanged_pixels"] > 0


@pytest.mark.parametrize("offset", [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
@pytest.mark.parametrize("clipped", [(0, 122, 200), (122, 122, 255)])
def test_any_clipped_reference_stencil_sample_remains_unknown(offset, clipped):
    reference, current = optical_pair()
    reference.putpixel((240+offset[0], 160+offset[1]), clipped)
    result = measure(reference, current)
    assert result["covered"] is False and result["holes_filled"] is False
    assert result["unknown_inside_front_pixels"] > 0


@pytest.mark.parametrize("value", [(255, 255, 255), (0, 0, 0), (127, 121, 255)])
def test_current_clipping_is_unknown_even_inside_a_closed_front(value):
    reference, current = optical_pair()
    current.putpixel((240, 160), value)
    result = measure(reference, current)
    assert result["covered"] is False and result["holes_filled"] is False


def test_complete_outline_and_margin_are_checked_instead_of_its_centre_only():
    pair = optical_pair()
    assert measure(*pair)["covered"] is True
    wide = ((235., 157.), (345., 157.), (345., 163.), (235., 163.))
    result = measure(*pair, outline=wide)
    assert result["covered"] is False
    assert measure(*pair, margin=.65)["covered"] is False


def test_observed_front_is_shrunk_before_it_can_supply_body_cells():
    pair = optical_pair()
    boundary_body = ((338.1, 159.8), (338.3, 159.8), (338.3, 160.2), (338.1, 160.2))
    result = measure(*pair, optical=EMPTY, outline=boundary_body, margin=0.)
    front = result["front"]
    assert front["has_contact"] and front["interface_uncertainty_px"] > 0
    # This body plus the one-pixel diagonal guard fits the unshrunk radius.
    # Refusal therefore needs the observed interface uncertainty, not just a
    # body vertex already outside the observed circle.
    assert ((338.3-front["centroid_px"][0])*PIXELS.mm_per_pixel_x
            +result["registration_guard_mm"] < front["radius_px"]*PIXELS.mm_per_pixel_x)
    assert result["covered"] is False
    assert result["front_domain_pixels"] < math.pi*front["radius_px"]**2


def test_clipped_front_cannot_authorize_a_complete_contact_domain():
    result = measure(*optical_pair(center=(35., 160.)))
    assert result["covered"] is False and result["supported_pixels"] == 0


@pytest.mark.parametrize("calibration", [PixelCalibration(0., .005), PixelCalibration(.005, 0.),
    PixelCalibration(float("nan"), -.005), PixelCalibration(.005, float("inf")),
    PixelCalibration(True, -.005), PixelCalibration(.005, False), PixelCalibration("0.005", -.005)])
def test_invalid_signed_calibration_is_rejected(calibration):
    with pytest.raises(ValueError):
        measure(*optical_pair(), calibration=calibration)


@pytest.mark.parametrize("labels", [("unknown",), (">40",), ("0-10 nm",), ("0-10 layers",), ("1-10",)])
def test_unknown_unbounded_or_misunit_public_bands_are_rejected(labels):
    with pytest.raises(ValueError):
        InteriorCalibration(labels_nm=labels).generators()


@pytest.mark.parametrize("field", ["profile", "camera_response", "contact", "carried_hbn", "raster", "source_mode"])
def test_unrecognized_optical_profile_is_rejected(field):
    with pytest.raises(ValueError, match="Unsupported"):
        measure(*optical_pair(), optical=replace(THIN, **{field: "unrecognized-profile"}))


def sharp_source_pair(*, bubble=None, neighbour=None):
    """A visible flat-tint source with hard edges, as the SIM renderer draws it.

    Its own edge cells see both substrate and source in their 3x3 reference
    stencil, so per-channel slabs alone cannot exclude 'unchanged' there (#136).
    An optional bubble keeps a small disc out of contact (reference RGB).
    """
    width, height = SIZE
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    reference = np.empty((height, width, 3), dtype=float)
    reference[:] = (125, 105, 170)
    body = (abs(xx-240) <= 5) & (abs(yy-160) <= 3)
    reference[body] = (118, 162, 197)
    if neighbour is not None:
        reference[(abs(xx-neighbour[0]) <= 2) & (abs(yy-neighbour[1]) <= 4)] = (147, 178, 165)
    current = reference.copy()
    optics = SYNTHETIC_CONTACT_PHOTOMETRY
    distance = np.hypot(xx-240, yy-120)
    rim = distance <= 100+6
    interior = distance <= 100
    current[rim] = (1-optics.front_opacity)*current[rim]+optics.front_opacity*np.array(optics.front_color)
    current[interior] = ((1-optics.interior_opacity)*current[interior]
                         +optics.interior_opacity*np.array(optics.interior_color))
    if bubble is not None:
        pocket = np.hypot(xx-bubble[0], yy-bubble[1]) <= 3
        current[pocket] = reference[pocket]
    def raster(values):
        return Image.fromarray(np.rint(values*gain[..., None]).astype(np.uint8), "RGB")
    return raster(reference), raster(current)


SHARP_OUTLINE = ((235., 157.), (245., 157.), (245., 163.), (235., 163.))


def test_a_visible_hard_edged_source_under_a_complete_front_is_covered():
    """#136: edge cells fit contact and no joint mixture of the unchanged
    alternatives reproduces all channels, so the source's own edge is no pocket."""
    result = measure(*sharp_source_pair(), optical=EMPTY, outline=SHARP_OUTLINE)
    assert result["covered"] is True, result
    assert result["edge_resolved_pixels"] > 0 and result["holes_filled"] is False
    assert result["unchanged_rule"] == "channel_slabs_then_joint_mixture_v1"


@pytest.mark.parametrize("bubble", [(240, 160), (240, 178), (222, 160)])
def test_a_non_contact_bubble_inside_the_source_or_margin_still_blocks(bubble):
    result = measure(*sharp_source_pair(bubble=bubble), optical=EMPTY, outline=SHARP_OUTLINE)
    assert result["covered"] is False and result["holes_filled"] is False


def test_a_neighbour_flake_inside_the_margin_band_does_not_block():
    """Owner decision B (#136): a neighbour inside the margin may be co-picked.
    Its contact-consistent edge cells stay unsupported but do not block the band."""
    result = measure(*sharp_source_pair(neighbour=(222, 178)), optical=EMPTY, outline=SHARP_OUTLINE)
    assert result["covered"] is True, result
    assert result["margin_rule"] == "strict_body_contact_consistent_margin_v1"
    assert result["body_coverage"]["covered"] is True and result["holes_filled"] is False
