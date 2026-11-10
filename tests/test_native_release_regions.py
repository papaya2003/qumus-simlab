"""Region topology survives release opacity/containment without source hulls."""
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.optical_profiles import RELEASE_SUPPORT_ASSUMPTIONS
from qumus_simlab.perception.contact_coverage import (
    InteriorCalibration,
    _release_opaque_masks,
    release_support_evidence,
)
from qumus_simlab.perception.tracking import PixelCalibration


def region(*, holes=False, island=False):
    parts = [{"exterior": [[-.10, -.10], [.10, -.10], [.10, .10], [-.10, .10]],
              "holes": [[[-.05, -.05], [-.05, .05], [.05, .05], [.05, -.05]]] if holes else []}]
    if island:
        parts.append({"exterior": [[-.003, -.003], [.003, -.003], [.003, .003], [-.003, .003]], "holes": []})
    return {"version": "polygon-region-v1", "components": parts}


def test_native_opaque_exemption_keeps_real_hole_and_island_with_existing_psf_guard():
    loaded = [{"material": "Gr", "region_mm": region(holes=True, island=True)}]
    before = deepcopy(loaded)
    reference, current, union = _release_opaque_masks(loaded, (240, 240), PixelCalibration(.001, -.001), 0., 0.)
    assert current[120, 120] and current[120, 200]
    assert not current[120, 145] and not reference[120, 145] and not union[120, 145]
    assert loaded == before


def test_release_native_region_reaches_rgb_support_without_outline_alias(monkeypatch):
    import qumus_simlab.perception.contact_coverage as module

    loaded = [{"flake_id": "Gr1", "material": "Gr", "region_mm": region(holes=True)}]
    monkeypatch.setattr(module, "_release_strict_cells", lambda a, b, optical, unknown:
                        (np.zeros((320, 480), dtype=bool), {"has_contact": False}))
    result = release_support_evidence(Image.new("RGB", (480, 320), (120, 100, 170)),
        Image.new("RGB", (480, 320), (120, 100, 170)), PixelCalibration(2.6/480, -1.72/320), loaded,
        optical=InteriorCalibration(), reference_transport={"load_envelope_mm": 0.},
        current_transport={"load_envelope_mm": 0.}, profile=RELEASE_SUPPORT_ASSUMPTIONS)
    assert not result["conditional_modeled_support"]
    assert result["reason"] == "Insufficient strict observed interior"
    assert result["recognition_tail_budget_applies"] is False
    assert result["owner_error_budget"]["inclusive_error_budget_mm"] == .001
    assert "outline_mm" not in loaded[0]


def test_native_release_class_must_match_actual_frozen_bounded_labels():
    loaded = [{"flake_id": "hBN1", "material": "hBN", "thickness_label": "0-10", "region_mm": region()},
              {"flake_id": "Gr1", "material": "Gr", "region_mm": region()}]
    with pytest.raises(ValueError, match="frozen observed"):
        release_support_evidence(Image.new("RGB", (480, 320)), Image.new("RGB", (480, 320)),
            PixelCalibration(2.6/480, -1.72/320), loaded, optical=InteriorCalibration(("10-20",)),
            reference_transport={"load_envelope_mm": 0.}, current_transport={"load_envelope_mm": 0.},
            profile=RELEASE_SUPPORT_ASSUMPTIONS)


def _square(cx, cy, half):
    return [[cx-half, cy-half], [cx+half, cy-half], [cx+half, cy+half], [cx-half, cy+half]]


@pytest.mark.parametrize("label,opaque", [("0-10", False), ("10-20", True), ("20-30", True)])
def test_thick_carried_hbn_is_opaque_unknown_not_seen_through(monkeypatch, label, opaque):
    """D13 principle at release: LIMITS stay frozen; a carried hBN class the
    camera cannot see through is unknown, and 0-10 nm keeps its see-through law."""
    import qumus_simlab.perception.contact_coverage as module

    calibration = PixelCalibration(2.6/480, -1.72/320)
    extra = {"material": "hBN", "thickness_label": "20-30",
             "region_mm": {"version": "polygon-region-v1", "components": [{"exterior": _square(.6, .3, .03), "holes": []}]}}
    loaded = [{"flake_id": "hBN1", "material": "hBN", "thickness_label": label, "outline_mm": _square(0, 0, .1)},
              {"flake_id": "Gr1", "material": "Gr", "outline_mm": _square(0, 0, .01)}]
    seen = {}

    def strict(before, after, optical, unknown):
        seen.update(optical=optical, unknown=unknown)
        return np.zeros((320, 480), dtype=bool), {"has_contact": False}

    def region(reference, current, calibration, vertices, masks, *args):
        seen.update(optical=InteriorCalibration(), unknown=masks[2])
        return {"carried_hbn_unknown": {}}

    monkeypatch.setattr(module, "_release_strict_cells", strict)
    monkeypatch.setattr(module, "_release_region_support", region)
    result = release_support_evidence(Image.new("RGB", (480, 320), (120, 100, 170)),
        Image.new("RGB", (480, 320), (120, 100, 170)), calibration, loaded,
        optical=InteriorCalibration((label,)), reference_transport={"load_envelope_mm": 0.},
        current_transport={"load_envelope_mm": 0.}, profile=RELEASE_SUPPORT_ASSUMPTIONS, unknown_flakes=[extra])
    assert module.release_carried_hbn_opaque(InteriorCalibration((label,)), 2) is opaque
    # 25 px from the centre: inside the 0.1 mm hBN, outside the Gr and its 13 px guard.
    hbn_px = (240+25, 160)
    extra_px = (round(240+.6/calibration.mm_per_pixel_x), round(160+.3/calibration.mm_per_pixel_y))
    assert bool(seen["unknown"][hbn_px[1], hbn_px[0]]) is opaque
    assert bool(seen["unknown"][extra_px[1], extra_px[0]]) is opaque
    assert ("carried_hbn_unknown" in result) is opaque
    assert seen["optical"] == (InteriorCalibration() if opaque else InteriorCalibration((label,)))
    if not opaque:
        assert result["opaque_unknown_mask_sha256"] == module.hashlib.sha256(
            _release_opaque_masks(loaded, (480, 320), calibration, 0., 0.)[2].tobytes()).hexdigest()


def _release_frames(*, bubble=False):
    """A purple wafer with gold leads; the contact covers a frame-clipped disc."""
    from qumus_simlab.core.optical_profiles import SYNTHETIC_CONTACT_PHOTOMETRY as photo

    ref = np.zeros((320, 480, 3), dtype=float)
    ref[:] = (120, 101, 162)
    ref[150:170, 120:360] = (186, 165, 104)   # a lead under the stamp
    ref[60:260, 236:244] = (186, 165, 104)    # a crossing lead
    yy, xx = np.mgrid[:320, :480]
    gain = ((1-.14*((xx-240)/240)**2)*(1-.14*((yy-160)/160)**2))[..., None]
    slope = (1-photo.front_opacity)*(1-photo.interior_opacity)
    offset = (np.array(photo.front_color)*photo.front_opacity*(1-photo.interior_opacity)
              + np.array(photo.interior_color)*photo.interior_opacity)
    contact = (xx-240)**2+(yy-120)**2 <= 150**2   # clipped at the top of the frame
    if bubble:
        contact &= (xx-270)**2+(yy-195)**2 > 6**2   # inside the hull + margin
    cur = ref.copy()
    cur[contact] = slope*ref[contact]+(gain*offset)[contact]
    ring = ((xx-240)**2+(yy-120)**2 > 150**2) & ((xx-240)**2+(yy-120)**2 <= 153**2)
    cur[ring] = (90, 80, 120)
    cur[146:174, 235:245] = (60, 70, 140)     # the opaque loaded stack
    return [Image.fromarray(np.clip(v, 1, 254).astype(np.uint8)) for v in (ref, cur)]


@pytest.mark.parametrize("bubble", [False, True])
def test_opaque_release_uses_the_observed_contact_region_over_electrodes(bubble):
    """D13 at release: a far-side clipped front over gold leads admits the
    opaque load only when its region holds the hull plus margin with no pocket."""
    loaded = [{"flake_id": "hBN1", "material": "hBN", "thickness_label": "20-30", "outline_mm": _square(0, 0, .025)},
              {"flake_id": "Gr1", "material": "Gr", "outline_mm": _square(0, 0, .01)}]
    ref, cur = _release_frames(bubble=bubble)
    result = release_support_evidence(ref, cur, PixelCalibration(2.6/480, -1.72/320), loaded,
        optical=InteriorCalibration(("20-30",)), reference_transport={"load_envelope_mm": 0.},
        current_transport={"load_envelope_mm": .03}, profile=RELEASE_SUPPORT_ASSUMPTIONS)
    assert result["front"]["front_bounds_from_fitted_circle"] is True
    assert bool(result["region_coverage"]["region_inside_observed_support"])
    assert result["conditional_modeled_support"] is (not bubble)
    # The accepted record is re-derived after storage and compared as stored.
    import json
    assert json.loads(json.dumps(result, default=str))["region_coverage"] == result["region_coverage"]
    if bubble:
        assert result["region_coverage"]["blocking_pockets"] >= 1
