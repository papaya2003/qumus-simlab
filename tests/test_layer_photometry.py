"""Declared synthetic slab optics, tested through the public raster renderer.

These fixtures do not calibrate a real microscope or validate PDMS/PC optics.
The retained checks cover the declared renderer model and physical layer order.
"""

import math

import pytest

from qumus_simlab.devices.base import Frame
from qumus_simlab.perception.tracking import load_pixels
from qumus_simlab.testbench.render import SYNTHETIC_HBN_SLAB, MicroscopeRenderer, SyntheticHBNSlab

SIZE = (480, 320)
FOV_MM = (.65, .43)
# A fully resolved six-sided source footprint, in fixture world/image axes.
SOURCE_OUTLINE_MM = ((.0025, .0105), (-.0045, .0085), (-.0085, -.0045),
                     (-.003, -.0115), (.005, -.0085), (.008, .0045))
FOREGROUND_OUTLINE_MM = ((-.11, -.08), (.12, -.07), (.10, .10), (-.09, .11))


def source_flake(dx_px=0, dy_px=0):
    dx, dy = dx_px*FOV_MM[0]/SIZE[0], -dy_px*FOV_MM[1]/SIZE[1]
    return dict(sim_id="layer-photometry-source", material="Gr", layers=6,
                area_um2=217., dx_mm=dx, dy_mm=dy,
                outline_mm=[(x+dx, y+dy) for x, y in SOURCE_OUTLINE_MM])


def foreground(*, material="hBN", layers=96):
    return dict(sim_id="layer-photometry-stamp", material=material, layers=layers,
                area_um2=30000., dx_mm=0., dy_mm=0., outline_mm=FOREGROUND_OUTLINE_MM)


def scene(*, source=None, carried=None, sharp_stamp=.5):
    return dict(objective=20, fov_mm=FOV_MM, on_chip="layer-photometry-chip",
                sharpness_chip=10.5, sharpness_stamp=sharp_stamp,
                flakes=[source if source is not None else source_flake()],
                stamp={"film": True}, stamp_flakes=[foreground()] if carried is None else carried)


def exposure(metadata):
    return Frame("scope", 0., data=MicroscopeRenderer(None, size=SIZE).render_bytes("scope", metadata))


def test_slab_weight_has_declared_lossless_limits_and_96_layer_value():
    slab = SyntheticHBNSlab()
    assert slab.source == "synthetic"
    assert slab.wavelength_um == .55 and slab.layer_spacing_um == .000333
    assert slab.projection_weight(0) == 0.
    assert slab.projection_weight(96) == pytest.approx(.2579786749780012)
    assert 0 < slab.projection_weight(2) < slab.projection_weight(96)
    assert all(0 <= slab.projection_weight(layers) < 1 for layers in (1, 2, 96, 1000, 10000))
    # A lossless slab has interference minima; opacity must not be forced to
    # grow monotonically with layer count as if this were an absorption law.
    n = math.sqrt(1+slab.sellmeier_a/(1-(slab.sellmeier_pole_um/slab.wavelength_um)**2))
    half_wave_layers = round(slab.wavelength_um/(2*n*slab.layer_spacing_um))
    assert slab.projection_weight(half_wave_layers) < .001
    assert slab == SYNTHETIC_HBN_SLAB


def test_top_first_stamp_order_renders_lower_layer_before_transmitted_hbn():
    hbn = foreground()
    graphite = foreground(material="Gr", layers=200) | {"sim_id": "lower-graphite"}
    top_hbn = exposure(scene(carried=[hbn, graphite], sharp_stamp=10.5))
    top_graphite = exposure(scene(carried=[graphite, hbn], sharp_stamp=10.5))
    graphite_only = exposure(scene(carried=[graphite], sharp_stamp=10.5))
    centre = (SIZE[0]//2, SIZE[1]//2)
    # An opaque top flake hides the lower slab; reversing the physical order
    # leaves the transmitted hBN contribution visible over the same graphite.
    assert load_pixels(top_graphite).getpixel(centre) == load_pixels(graphite_only).getpixel(centre)
    assert load_pixels(top_hbn).getpixel(centre) != load_pixels(top_graphite).getpixel(centre)
