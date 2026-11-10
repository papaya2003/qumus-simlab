"""Contact-region pickup coverage (#136, D13).

Owner decisions (Jialei, 2026-11-02): judge by strategy, not by frozen colour
limits, and do not assume a circular front (a tall neighbour or an imperfect
PDMS dome). Every pixel of the outline plus margin must be observed contact,
judged relative to this exposure's own contact colour and the unchanged
reference; any resolvable non-contact pocket (bubble, lagging front) blocks.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from test_contact_interior import (
    OUTLINE,
    PIXELS,
    SHARP_OUTLINE,
    SIZE,
    optical_pair,
    sharp_source_pair,
)

from qumus_simlab.core.optical_profiles import SYNTHETIC_CONTACT_PHOTOMETRY
from qumus_simlab.perception.contact_coverage import contact_region_coverage
from qumus_simlab.testbench.render import MicroscopeRenderer

MARGIN_MM = .1752


def measure(reference, current, outline=OUTLINE):
    return contact_region_coverage(reference, current, PIXELS, outline, margin_mm=MARGIN_MM)


def test_a_complete_front_over_a_hard_edged_source_is_covered():
    result = measure(*sharp_source_pair(), outline=SHARP_OUTLINE)
    assert result["covered"] is True, result
    assert result["blocking_pockets"] == 0 and result["holes_filled"] is False
    assert result["region_contact_fraction"] == 1.


@pytest.mark.parametrize("bubble", [(240, 160), (240, 178), (222, 160)])
def test_a_non_contact_bubble_inside_the_source_or_margin_blocks(bubble):
    result = measure(*sharp_source_pair(bubble=bubble), outline=SHARP_OUTLINE)
    assert result["covered"] is False
    assert result["reason"] == "A non-contact pocket lies inside the outline or margin"
    assert result["blocking_pockets"] >= 1


def test_a_neighbour_flake_inside_the_margin_does_not_block():
    """Owner decision B (#136): a neighbour that contacts with the source is no bubble."""
    result = measure(*sharp_source_pair(neighbour=(222, 178)), outline=SHARP_OUTLINE)
    assert result["covered"] is True, result


@pytest.mark.parametrize("layers", [0, 1, 10, 30])
def test_carried_layers_do_not_change_the_relative_decision(layers):
    result = measure(*optical_pair(layers=layers))
    assert result["covered"] is True, result


def test_an_unchanged_frame_never_covers():
    reference, _ = optical_pair()
    result = measure(reference, reference.copy())
    assert result["covered"] is False and result["found"] is False


def test_a_front_that_does_not_yet_reach_the_margin_does_not_cover():
    result = measure(*optical_pair(center=(240., 120.), radius=70.))
    assert result["covered"] is False and result["blocking_pockets"] >= 1


def test_a_front_clipped_by_the_camera_support_still_covers_a_region_in_view():
    """The v1-style offset contact centre lies near the image edge, so the
    front leaves the view before it holds the source plus margin (#136)."""
    result = measure(*optical_pair(center=(240., 95.), radius=125.))
    assert result["front"]["failure_reason"] == "boundary_clipped"
    assert result["covered"] is True, result


def _shaped_pair(contact_mask, *, bubble=None):
    """Uniform field with an arbitrary contact shape (declared photometry)."""
    reference, _ = optical_pair()
    a = np.asarray(reference, dtype=float)
    width, height = SIZE
    yy, xx = np.mgrid[:height, :width]
    gain = ((1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2))[..., None]
    optics = SYNTHETIC_CONTACT_PHOTOMETRY
    alpha = (1-optics.front_opacity)*(1-optics.interior_opacity)
    offset = (np.array(optics.front_color)*optics.front_opacity*(1-optics.interior_opacity)
              + np.array(optics.interior_color)*optics.interior_opacity)
    mask = contact_mask(xx, yy)
    if bubble is not None:
        mask &= np.hypot(xx-bubble[0], yy-bubble[1]) > 3
    b = np.where(mask[..., None], alpha*a+gain*offset, a)
    return reference, Image.fromarray(np.clip(np.rint(b), 0, 255).astype(np.uint8), "RGB")


def test_a_non_circular_front_that_holds_the_region_covers():
    """An elongated contact (imperfect dome) needs no circle fit."""
    def ellipse(xx, yy):
        return ((xx-240)/170)**2+((yy-150)/75)**2 <= 1
    result = measure(*_shaped_pair(ellipse))
    assert result["covered"] is True, result


def test_a_front_indented_into_the_margin_blocks():
    """A tall neighbour holds the film off one side: the lagging front cuts
    into the margin although a circle through the rest would hold it."""
    def notched(xx, yy):
        return (np.hypot(xx-240, yy-150) <= 110) & ~((xx > 255) & (abs(yy-160) < 12))
    result = measure(*_shaped_pair(notched))
    assert result["covered"] is False and result["blocking_pockets"] >= 1


def test_an_outline_whose_margin_leaves_the_camera_support_does_not_cover():
    outline = tuple((x, y-120.) for x, y in OUTLINE)
    result = measure(*optical_pair(center=(240., 60.), radius=100.), outline=outline)
    assert result["covered"] is False
    assert result["reason"] == "Outline and margin are not inside the observed camera support"


GR_MM = [[-.03, -.02], [.03, -.02], [.03, .02], [-.03, .02]]
HBN_MM = [[-.07, -.05], [.07, -.05], [.07, .05], [-.07, .05]]


def _rendered(contact, layers):
    metadata = {"on_chip": "fixture", "fov_mm": [2.6, 1.72], "objective": 5,
                "sharpness_chip": 10.5, "sharpness_stamp": 10.5, "stamp": {"film": True},
                "stamp_texture_seed": "fixture:d13", "contact": contact,
                "flakes": [{"sim_id": "gr", "material": "Gr", "layers": 3, "outline_mm": GR_MM,
                            "dx_mm": 0, "dy_mm": 0}],
                "stamp_flakes": [{"sim_id": "h", "material": "hBN", "layers": layers, "outline_mm": HBN_MM,
                                  "dx_mm": 0, "dy_mm": 0}],
                "observation_noise_scale": 0}
    data = MicroscopeRenderer(None, size=(480, 320)).render_bytes("fixture", metadata)
    return Image.open(io.BytesIO(data)).convert("RGB")


@pytest.mark.parametrize("layers", [20, 45, 75, 105])  # 0-10, 10-20, 20-30, 30-40 nm carried hBN
def test_rendered_carried_hbn_of_any_bounded_thickness_is_judged_relatively(layers):
    """SIM renderer pixels (no mock): a thick carried hBN over the Gr no longer
    blocks a complete front, and the same unchanged exposure still blocks."""
    outline = [(240+x/PIXELS.mm_per_pixel_x, 160+y/PIXELS.mm_per_pixel_y) for x, y in GR_MM]
    reference = _rendered(None, layers)
    current = _rendered({"dx_mm": 0, "dy_mm": .05, "radius_mm": .5}, layers)
    covered = measure(reference, current, outline=outline)
    assert covered["covered"] is True, covered
    assert covered["contact_separation_dn"] > 3*covered["contact_fit_noise_dn"]+2
    assert covered["contact_region_bounds_px"]
    unchanged = measure(reference, reference.copy(), outline=outline)
    assert unchanged["covered"] is False


def test_a_bubble_under_a_thick_carried_hbn_still_blocks():
    outline = [(240+x/PIXELS.mm_per_pixel_x, 160+y/PIXELS.mm_per_pixel_y) for x, y in GR_MM]
    reference = _rendered(None, 105)
    current = np.asarray(_rendered({"dx_mm": 0, "dy_mm": .05, "radius_mm": .5}, 105)).copy()
    yy, xx = np.mgrid[:320, :480]
    pocket = np.hypot(xx-240, yy-160) <= 3
    current[pocket] = np.asarray(reference)[pocket]
    result = measure(reference, Image.fromarray(current, "RGB"), outline=outline)
    assert result["covered"] is False and result["blocking_pockets"] >= 1


@pytest.mark.parametrize("where,covered", [((262, 172), True), ((243, 162), False)])
def test_a_co_picked_carried_flake_beside_the_source_does_not_block(where, covered):
    """a518 seed 2 (Jialei 2026-11-03: a co-picked neighbour hBN is normal).
    A thick carried flake inside the margin changes colour under contact in
    its own way; it is no unchanged bubble, so beside the source it does not
    block. The same patch on the source outline still blocks."""
    reference, current = _shaped_pair(lambda xx, yy: np.hypot(xx-240, yy-150) <= 110)
    b = np.asarray(current, dtype=float).copy()
    yy, xx = np.mgrid[:SIZE[1], :SIZE[0]]
    patch = (abs(xx-where[0]) <= 3) & (abs(yy-where[1]) <= 2)
    # A blurred thick flake: no single carried colour repeats often enough to
    # be learned as contact, as in the a518 seed-2 exposure.
    ramp = (xx[patch]-where[0]+3)[:, None]*np.array([3., 3., -3.])
    b[patch] = np.asarray(reference, dtype=float)[patch]+np.array([13., 23., -18.])+ramp
    result = measure(reference, Image.fromarray(np.clip(np.rint(b), 0, 255).astype(np.uint8), "RGB"))
    assert result["covered"] is covered, result
    if covered:
        assert result["carried_pockets"] == 1
    else:
        assert result["blocking_pockets"] >= 1
