"""Exposure metadata freezes the declared optics without changing mechanics."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from qumus_simlab.core.clock import SimClock
from qumus_simlab.testbench.lab import build_sim_lab
from qumus_simlab.testbench.optics import STAMP_STRUCTURE_PROFILE, WHITE_LIGHT_PROFILE
from qumus_simlab.testbench.persist import mechanics_signature
from qumus_simlab.testbench.render import (
    STAMP_PSF_PROFILE,
    SYNTHETIC_CONTACT_PHOTOMETRY,
    SYNTHETIC_HBN_SLAB,
    MicroscopeRenderer,
)
from tests.conftest import display_matches, quiet_bench
from tests.test_motion_preview_evidence import fixture


@pytest.mark.parametrize("file_backed", [False, True])
def test_capture_and_both_previews_bind_the_declared_profile(tmp_path, file_backed):
    lab = build_sim_lab(SimClock(), config=quiet_bench(), image_dir=tmp_path if file_backed else None)
    scope = lab.devices.get("microscope")
    scope.renderer.size = (128, 84)
    expected = {"source": "synthetic", "contact": SYNTHETIC_CONTACT_PHOTOMETRY.calibration_id,
                "fringes": WHITE_LIGHT_PROFILE,
                "carried_hbn": SYNTHETIC_HBN_SLAB.calibration_id, "raster": STAMP_PSF_PROFILE,
                "chip_surface": "synthetic-chip-footprint-v2",
                "stamp_structure": STAMP_STRUCTURE_PROFILE}
    for frame in (scope.capture(), scope.preview(), scope.preview_snapshot()):
        assert frame.meta["calibration"]["photometry_profile"] == expected
        assert frame.meta["captured_at"] == frame.timestamp
        assert frame.meta["calibration"]["image_size_px"] == [frame.width, frame.height] == [128, 84]
        assert frame.meta["calibration"]["photometry_profile"] is not scope.renderer.photometry_profile


def test_profile_change_cannot_relabel_a_capture_or_rewrite_delayed_pixels(tmp_path):
    class MutableProfileRenderer(MicroscopeRenderer):
        @property
        def photometry_profile(self):
            return self.profile

    lab = build_sim_lab(SimClock(), config=quiet_bench())
    scope = lab.devices.get("microscope")
    renderer = MutableProfileRenderer(tmp_path, size=(128, 84))
    renderer.profile = dict(scope.renderer.photometry_profile)
    scope.renderer = renderer
    original = scope.capture()
    preview = scope.preview_snapshot()
    frozen, pixels = copy.deepcopy(original.meta), original.data
    preview_meta = copy.deepcopy(preview.meta)
    preview_path = Path(preview.saved_path())
    preview_pixels = preview_path.read_bytes()
    signature, mechanics = scope.preview_signature(), mechanics_signature(lab)
    physical = lab.world.evaluation_snapshot()

    renderer.profile["fringes"] = "synthetic-white-light-fixture-v2"
    current = scope.capture()
    assert current.meta["calibration"] != original.meta["calibration"]
    assert current.meta["calibration"]["version"] == original.meta["calibration"]["version"]
    assert scope.preview_signature() != signature
    assert mechanics_signature(lab) == mechanics
    assert lab.world.evaluation_snapshot() == physical
    assert original.meta == frozen and preview.meta == preview_meta
    assert Path(original.saved_path()).read_bytes() == pixels
    scope.preview()
    scope.preview_snapshot()
    assert preview_path.read_bytes() == preview_pixels
    assert original.meta == frozen and preview.meta == preview_meta


def test_motion_event_saves_the_same_profile_binding_as_its_exposure(tmp_path):
    _, _, store, gate, _, scope = fixture(tmp_path)
    gate.microscope("microscope").set_focus(.2)
    previews = [event.payload["microscope_preview"] for event in store.events_after(0, limit=1000)
                if event.kind == "lab.frame" and "microscope_preview" in event.payload]
    assert previews
    saved = previews[-1]
    assert saved["immutable"] is True and saved["verification"] is False
    assert saved["calibration"] == scope.capture().meta["calibration"]
    assert Path(saved["image"]).is_file()


def test_legacy_exposure_pixels_and_default_noise_are_unchanged():

    packet = {"on_chip": "compat", "fov_mm": [.65, .43], "objective": 20,
              "sharpness_chip": 10.5, "flakes": [], "chip_corners": []}
    legacy = MicroscopeRenderer._compose(packet, size=(128, 84))
    # Frozen from the c1a0600 renderer, before condition/texture consumer changes.
    assert _pixel_sha(legacy) == "66195ffdc5782749bc54f977857af79c9772843ab5392ff7f0e0d025bbe9508c"  # bytes 1c0b22f0
    assert MicroscopeRenderer._compose({**packet, "observation_noise_scale": 1}, size=(128, 84)) == legacy


def test_zero_camera_noise_preserves_deterministic_illumination():
    from qumus_simlab.testbench.render import Canvas, _camera

    canvas = Canvas(4, 4, (100, 100, 100))
    _camera(canvas, 123, 0)
    assert canvas.px[:3] == bytes((73, 73, 73))
    middle = (2*4+2)*3
    assert canvas.px[middle:middle+3] == bytes((100, 100, 100))


@pytest.fixture
def ideal_optical_lab(tmp_path):
    from qumus_simlab.config import ideal_stacking_profile
    from qumus_simlab.runtime.store import RunStore

    clock = SimClock()
    store = RunStore(":memory:", clock=clock)
    config = ideal_stacking_profile()
    config["seed"] = 8
    lab = build_sim_lab(clock, config=config, image_dir=tmp_path, store=store)
    scope = lab.devices.get("microscope")
    yield lab, clock, scope
    store.close()


def test_actual_ideal_capture_freezes_condition_for_pixels_and_previews(ideal_optical_lab):
    lab, _, scope = ideal_optical_lab
    scope.renderer.size = (128, 84)
    original, preview = scope.capture(), scope.preview_snapshot()
    frozen, pixels = copy.deepcopy(original.meta), original.data
    signature = scope.preview_signature()
    for frame in (original, preview, scope.preview()):
        assert frame.meta["simulation_condition"] == "ideal_stacking_v1"
        assert frame.meta["observation_noise_scale"] == 0
        assert frame.meta["calibration"]["observation_condition"] == {
            "simulation_condition": "ideal_stacking_v1", "observation_noise_scale": 0}
    lab.world.model["microscope"]["observation_noise_scale"] = 1
    assert scope.preview_signature() != signature
    assert scope.capture().data != pixels
    assert original.meta == frozen
    assert scope.renderer.render_bytes("microscope", original.meta) == pixels
    assert Path(original.saved_path()).read_bytes() == pixels
    assert preview.meta["observation_noise_scale"] == 0


def _projected_fixture(ideal_optical_lab, plane, objective):
    from qumus_simlab.core.polygons import area_um2
    from qumus_simlab.core.types import Pose
    from qumus_simlab.testbench.world import SimFlake, StampState

    lab, clock, scope = ideal_optical_lab
    chip = lab.world.objects["chip_001"]
    cx, cy = scope.center
    chip.pose, chip.on, chip.held_by = Pose(x=cx+.08, y=cy-.03, rotation=90), "transfer_stage", None
    chip.focus_offset = 0
    outline = ((-.03, -.02), (.04, -.02), (.01, .04))
    flake = SimFlake("projection", "Gr", 3, .02, .01, area_um2(outline), outline)
    chip.flakes = [flake] if plane == "chip" else []
    if plane == "carried":
        lab.world.objects["stamp_01"].held_by = "transfer_arm"
        state = StampState(5.3, 0, 0, 165, optic_shift=.3)
        # Nonzero old source-local coordinates must not be added to the saved offset.
        flake.x_mm, flake.y_mm = .6, -.7
        state.stack = [(flake, .03, -.02)]
        lab.world.stamps["stamp_01"] = state
        origin = lab.bench["transfer_arm"]["origin"]
        lab.world.manip.x, lab.world.manip.y = cx-.04-origin["x"], cy+.06-origin["y"]
        lab.world.manip.z = 5.45
    scope.set_objective(objective)
    scope.set_focus(.45 if plane == "carried" else 0)
    return lab, clock, scope


@pytest.mark.parametrize("objective", [5, 20])
@pytest.mark.parametrize("plane", ["chip", "carried"])
def test_state_to_actual_pixels_preserves_rotations_offsets_scale_and_one_y_flip(ideal_optical_lab, plane, objective):
    import io

    from PIL import Image, ImageChops

    lab, clock, scope = _projected_fixture(ideal_optical_lab, plane, objective)
    frame = scope.capture()
    field = "flakes" if plane == "chip" else "stamp_flakes"
    shape = frame.meta[field][0]
    # Independent numeric geometry: R90*(.02,.01)+(.08,-.03), or apex+saved offset.
    centre = (.07, -.01) if plane == "chip" else (-.01, .04)
    vertices = [(.09, -.04), (.09, .03), (.03, 0)] if plane == "chip" else [(-.04, .02), (.03, .02), (0, .08)]
    assert (shape["dx_mm"], shape["dy_mm"]) == pytest.approx(centre, abs=1e-12)
    for actual, expected in zip(shape["outline_mm"], vertices, strict=True):
        assert actual == pytest.approx(expected, abs=1e-12)
    baseline = copy.deepcopy(frame.meta)
    baseline[field] = []
    with Image.open(io.BytesIO(frame.data)) as image, Image.open(io.BytesIO(scope.renderer.render_bytes("microscope", baseline))) as blank:
        delta = ImageChops.difference(image, blank).convert("L")
    weights = list(delta.get_flattened_data())
    mass = sum(weights)
    assert mass > 0
    measured = (sum((i % delta.width+.5)*v for i, v in enumerate(weights))/mass,
                sum((i // delta.width+.5)*v for i, v in enumerate(weights))/mass)
    fw, fh = frame.meta["fov_mm"]
    expected = (delta.width/2+sum(x for x, _ in vertices)/3/fw*delta.width,
                delta.height/2-sum(y for _, y in vertices)/3/fh*delta.height)
    assert measured == pytest.approx(expected, abs=1.5)
    before = (lab.world.evaluation_snapshot(), lab.world.rng.getstate(), lab.world._observation_rng.getstate(), clock.now())
    for _ in range(2):
        assert scope.renderer.render_bytes("microscope", frame.meta) == frame.data
        assert display_matches(Path(scope.preview_snapshot().saved_path()).read_bytes(), frame.data)
        assert display_matches(Path(scope.preview().saved_path()).read_bytes(), frame.data)
    assert before == (lab.world.evaluation_snapshot(), lab.world.rng.getstate(), lab.world._observation_rng.getstate(), clock.now())


def _texture_plane(texture, size=(400, 400), fov=(2, 2)):
    from PIL import Image

    from qumus_simlab.testbench.render import Canvas

    canvas = Canvas(*size, (160, 160, 160))
    MicroscopeRenderer._specimen_residue(canvas, texture, lambda x, y: True,
        size[0]/fov[0], size[1]/fov[1], *fov)
    return Image.frombytes("RGB", size, bytes(canvas.px))


def test_specimen_texture_translates_rotates_and_zooms_in_its_own_frame():
    from PIL import Image, ImageChops, ImageFilter

    packet = {"seed": "7:chip_001", "pose": {"x": 0, "y": 0, "rotation": 0},
              "size_mm": [4, 4], "view_center_world_mm": [0, 0]}
    base = _texture_plane(packet)
    assert ImageChops.difference(base, Image.new("RGB", base.size, (160, 160, 160))).getbbox()
    translated = _texture_plane({**packet, "pose": {"x": .125, "y": .125, "rotation": 0}})
    assert translated.crop((25, 0, 400, 375)).tobytes() == base.crop((0, 25, 375, 400)).tobytes()
    zoom = _texture_plane(packet, size=(200, 200), fov=(1, 1))
    assert zoom.tobytes() == base.crop((100, 100, 300, 300)).tobytes()
    rotated = _texture_plane({**packet, "pose": {"x": 0, "y": 0, "rotation": 90}})
    expected = base.transpose(Image.Transpose.ROTATE_90)
    def mask(image):
        return ImageChops.difference(image, Image.new("RGB", image.size, (160, 160, 160))).convert("L").point(lambda p: 255 if p else 0)
    actual_mask, expected_mask = mask(rotated), mask(expected)
    # Scanline rasterization of rotated polygon edges has one-pixel support.
    assert ImageChops.subtract(actual_mask, expected_mask.filter(ImageFilter.MaxFilter(3))).getbbox() is None
    assert ImageChops.subtract(expected_mask, actual_mask.filter(ImageFilter.MaxFilter(3))).getbbox() is None


def test_removing_a_flake_does_not_regenerate_unrelated_specimen_pixels(ideal_optical_lab):
    import io

    from PIL import Image

    _, _, scope = _projected_fixture(ideal_optical_lab, "chip", 20)
    frame = scope.capture()
    without = copy.deepcopy(frame.meta)
    without["flakes"] = []
    with Image.open(io.BytesIO(frame.data)) as before, Image.open(io.BytesIO(scope.renderer.render_bytes("microscope", without))) as after:
        before_pixels, after_pixels = list(before.get_flattened_data()), list(after.get_flattened_data())
    # Known projected triangle is x=[.03,.09], y=[-.04,.03].
    sx, sy = frame.meta["calibration"]["mm_per_pixel"]
    changed = []
    for i, (a, b) in enumerate(zip(before_pixels, after_pixels, strict=True)):
        if a != b:
            x, y = (i % frame.width+.5-frame.width/2)*sx, -(i // frame.width+.5-frame.height/2)*sy
            assert .03-sx <= x <= .09+sx and -.04-sy <= y <= .03+sy
            changed.append(i)
    assert changed


def test_torn_film_texture_uses_its_own_frozen_seed():
    import random

    from qumus_simlab.testbench.render import Canvas

    packet = {"stamp": {"film": False}, "contact": {"dx_mm": 0, "dy_mm": 0, "radius_mm": .1},
              "stamp_texture_seed": "7:stamp_01"}
    a, b = Canvas(160, 120, (120, 120, 120)), Canvas(160, 120, (120, 120, 120))
    MicroscopeRenderer._stamp(a, packet, 8.5, 400, 400, random.Random(1))
    MicroscopeRenderer._stamp(b, packet, 8.5, 400, 400, random.Random(99))
    assert a.px == b.px


def _pixel_sha(data: bytes) -> str:
    """sha256 of decoded RGB pixels: the PNG zlib level changed in #203, the
    pixels did not (the old byte hashes are kept in the comments)."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
