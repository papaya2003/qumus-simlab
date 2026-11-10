"""SIM renderer pixels (no mock): a thin faint Gr sliver stays registrable.

a518 seed 2 stopped at the second-source refind because register_specimen
thinned the boundary in raster order; an even stride kept only one side of
each row, so a thin sliver lost a whole direction and failed even against
its own 20x reference.
"""
from __future__ import annotations

import io
import math

import pytest
from PIL import Image

from qumus_simlab.devices.base import Frame
from qumus_simlab.perception.tracking import RegistrationConfig, register_specimen
from qumus_simlab.testbench.render import MicroscopeRenderer

SCALE = .65/480  # native 20x mm per pixel
# The a518 seed-2 Gr outline (pixels about its centre), a 6 x 40 px sliver.
SLIVER_PX = [(-10.5, -19.5), (-6.8, 3.9), (2.1, 13.3), (10.5, 19.5), (9.9, 7.8), (7.4, -1.6), (.4, -12.9)]
WIDE_PX = [(-14., -8.), (-4., 9.), (12., 11.), (15., -3.), (3., -12.)]


def _rendered(outline_px, *, rotation_deg=0., shift_px=(0., 0.), present=True, view="ref"):
    angle = math.radians(rotation_deg)
    turned = [(x*math.cos(angle)-y*math.sin(angle), x*math.sin(angle)+y*math.cos(angle)) for x, y in outline_px]
    flakes = [{"sim_id": "gr", "material": "Gr", "layers": 2,
               "outline_mm": [((x+shift_px[0])*SCALE, -(y+shift_px[1])*SCALE) for x, y in turned],
               "dx_mm": shift_px[0]*SCALE, "dy_mm": -shift_px[1]*SCALE}] if present else []
    metadata = {"on_chip": f"fixture:{view}", "fov_mm": [.65, .43], "objective": 20, "sharpness_chip": 10.5,
                "flakes": flakes, "observation_noise_scale": 1.}
    data = MicroscopeRenderer(None, size=(480, 320)).render_bytes("fixture", metadata)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _frame(image, tmp_path, name):
    path = tmp_path/f"{name}.png"
    image.save(path)
    return Frame("microscope", 0., path=str(path), meta={})


def _register(reference, current, *, nominal_deg):
    polygon = [(240+x, 160+y) for x, y in SLIVER_PX]
    return register_specimen(reference, current, polygon_px=polygon, config=RegistrationConfig(
        max_translation_px=40, max_rotation_deg=1., nominal_rotation_deg=nominal_deg,
        min_quality=.72, min_ambiguity=.02))


def test_faint_gr_sliver_registers_against_itself_and_a_noisy_rotated_refind(tmp_path):
    reference = _frame(_rendered(SLIVER_PX), tmp_path, "reference")
    assert _register(reference, reference, nominal_deg=0.).ok
    # Seed 2: the chip turned 153.89 deg and the source moved 11 px, 4 px.
    turned = _register(reference, _frame(_rendered(SLIVER_PX, rotation_deg=-153.89, view="turned"),
                                         tmp_path, "turned"), nominal_deg=153.89)
    refind = _register(reference, _frame(_rendered(SLIVER_PX, rotation_deg=-153.89, shift_px=(11., 4.),
                                                   view="refind"), tmp_path, "refind"), nominal_deg=153.89)
    assert turned.ok and refind.ok, (turned, refind)
    assert (refind.dx_px-turned.dx_px, refind.dy_px-turned.dy_px) == pytest.approx((11., 4.), abs=1.)


@pytest.mark.parametrize("wrong", ["absent", "other_shape", "other_rotation"])
def test_a_wrong_specimen_still_fails(tmp_path, wrong):
    reference = _frame(_rendered(SLIVER_PX), tmp_path, "reference")
    if wrong == "absent":
        current, nominal = _rendered(SLIVER_PX, present=False, view="refind"), 0.
    elif wrong == "other_shape":
        current, nominal = _rendered(WIDE_PX, view="refind"), 0.
    else:
        current, nominal = _rendered(SLIVER_PX, rotation_deg=-153.89, view="refind"), 0.
    assert not _register(reference, _frame(current, tmp_path, "current"), nominal_deg=nominal).ok
