"""Finite chip surfaces remain visible when only an edge crosses the field."""

from __future__ import annotations

import copy
import hashlib
import io
import math

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library.stacking import _image_calibration
from qumus_simlab.perception.tracking import raster_focus_score
from qumus_simlab.testbench.lab import build_sim_lab
from qumus_simlab.testbench.render import MicroscopeRenderer
from tests.conftest import quiet_bench


def scene(x, y, rotation=0, *, view=(0, 0)):
    return {"on_chip": None, "chip_corners": [], "flakes": [], "objective": 10,
            "fov_mm": [1.3, .86], "sharpness_chip": 10.5, "observation_noise_scale": 0,
            "specimen_texture": {"profile": "chip-local-procedural-v1", "seed": "fixture:chip",
                "chip_id": "chip", "pose": {"x": x, "y": y, "rotation": rotation},
                "size_mm": [10, 10], "view_center_world_mm": list(view)}}


def render(packet):
    return MicroscopeRenderer(None, size=(480, 320)).render_bytes("fixture", packet)


@pytest.mark.parametrize("rotation", [0, 23, 75, 135])
def test_crossing_edge_without_center_corner_or_flake_is_visible_and_focusable(rotation):
    angle = math.radians(rotation)
    nx, ny = math.cos(angle), math.sin(angle)
    packet = scene(5.2*nx, 5.2*ny, rotation)
    frozen = copy.deepcopy(packet)
    image = Image.open(io.BytesIO(render(packet)))

    def pixel_at(distance):
        return image.getpixel((round(240+distance*nx*480/1.3),
                               round(160-distance*ny*320/.86)))

    assert max(pixel_at(-.3)) < 30  # Dark stage outside the finite surface.
    assert max(pixel_at(.3)) > 90  # Surface beyond the crossing edge.
    sharp_score = raster_focus_score(image)
    blurred = {**packet, "sharpness_chip": .5}
    assert sharp_score > raster_focus_score(Image.open(io.BytesIO(render(blurred)))) > 0
    assert packet == frozen


def test_reported_normal_condition_corner_miss_still_projects_the_intersecting_surface():
    # The final R001 probe has its nearest vertex below the image while the
    # finite left edge and chip interior intersect the right side of the FOV.
    packet = scene(680.0446690864862, 424.30640324221326, .496745,
                   view=(675, 420))
    points = MicroscopeRenderer._specimen_outline(packet["specimen_texture"])
    assert all(abs(x) > .65 or abs(y) > .43 for x, y in points)
    image = Image.open(io.BytesIO(render(packet)))
    assert max(image.getpixel((80, 160))) < 30
    assert max(image.getpixel((400, 160))) > 90
    assert raster_focus_score(image) > 0


@pytest.mark.parametrize("x,y,rotation", [(6, 0, 0), (5.7, 5.7, 45), (0, 6, 90)])
def test_disjoint_chip_is_blank_even_when_rotated_bounding_box_overlaps(x, y, rotation):
    packet = scene(x, y, rotation)
    blank = {key: value for key, value in packet.items() if key != "specimen_texture"}
    assert render(packet) == render(blank)
    assert raster_focus_score(Image.open(io.BytesIO(render(packet)))) == 0


def test_apparent_optical_shift_moves_surface_and_texture_together():
    packet = scene(5.2, 0)
    shifted = copy.deepcopy(packet)
    shifted["specimen_texture"]["view_center_world_mm"] = [3, -2]
    shifted["specimen_texture"]["apparent_shift_xy_mm"] = [3, -2]
    assert render(shifted) == render(packet)


def test_unaffected_centered_texture_and_legacy_corner_keep_exact_raster():
    center = scene(0, 0, 17)
    center["on_chip"] = "chip"
    legacy = {"on_chip": "chip_legacy", "chip_corners": [{"chip": "chip_legacy", "corner": "bl",
        "dx_mm": -.2, "dy_mm": -.1, "chip_rotation": 17}], "objective": 10,
        "fov_mm": [1.3, .86], "sharpness_chip": 8, "flakes": [], "observation_noise_scale": 0}
    renderer = MicroscopeRenderer(None, size=(128, 84))

    def pixels(data):
        with Image.open(io.BytesIO(data)) as image:
            return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()

    # Frozen immediately before R001; affected partial-chip packets intentionally
    # change on re-render. Previously stored raw image bytes are never rewritten.
    # Decoded pixels are compared: the PNG zlib level changed in #203 (same
    # pixels as the bytes frozen before, b652341c... and 40f45afa...).
    assert pixels(renderer.render_bytes("fixture", center)) == (
        "4251ac9f14609f413dad8d414578d8df9283166c2a4284fbfe2a0cbe33da0b13")
    # The legacy packet is defocused (sharpness 8): re-frozen when the integer
    # 1/k downsample became a continuous Gaussian defocus (#133).
    assert pixels(renderer.render_bytes("fixture", legacy)) == (
        "a5c7c3eaf56791d0930c48d83e967a5152f595c4842e470b8b1f9d09602df4e9")


def test_previous_chip_surface_identity_cannot_authorize_a_current_optical_reference():
    lab = build_sim_lab(SimClock(), config=quiet_bench())
    scope = lab.devices.get("microscope")
    original = scope.capture()
    _image_calibration(original)
    metadata = copy.deepcopy(original.meta)
    assert metadata["calibration"]["photometry_profile"].pop("chip_surface") == (
        "synthetic-chip-footprint-v2")
    legacy = Frame(original.device_id, original.timestamp, data=original.data, meta=metadata)
    with pytest.raises(NeedsHumanError, match="capture calibration"):
        _image_calibration(legacy)
    assert "chip_surface" in original.meta["calibration"]["photometry_profile"]


@pytest.mark.parametrize("rotation,view", [(0, (0, 0)), (17, (0, 0)), (75, (4.9, 0)), (120, (4.9, 0))])
def test_one_focus_plane_gives_one_contiguous_raster_peak(rotation, view):
    """#133: integer 1/k downsampling once let a nearer plane score below a
    farther one, so a single plane produced two separated autofocus peaks."""
    packet = scene(0, 0, rotation, view=view)
    packet["on_chip"] = "chip"
    sharpness = [10.5-i*.1 for i in range(101)]
    scores = [raster_focus_score(Image.open(io.BytesIO(render({**packet, "sharpness_chip": s}))))
              for s in sharpness]
    peak = max(scores)
    top = [i for i, score in enumerate(scores) if score >= peak*.98]  # autofocus minimum_peak_gain
    assert top == list(range(top[0], top[-1]+1)) and top[0] == 0
    assert max(scores[top[-1]+1:]) < .95*peak
    assert scores[-1] < .2*peak
