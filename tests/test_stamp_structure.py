"""Frozen support geometry and state-driven optical keyframes; no process recipe."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library.stacking import _image_calibration
from qumus_simlab.testbench.optics import (
    LEGACY_STAMP_STRUCTURE_PROFILE,
    STAMP_STRUCTURE_PROFILE,
    StampStructure,
    stamp_structure_packet,
)
from qumus_simlab.testbench.render import Canvas, MicroscopeRenderer
from qumus_simlab.testbench.world import SimFlake
from tests.conftest import display_matches
from tests.test_white_light_optics import optical_lab


def test_stamp_structure_is_frozen_separate_from_apex_and_bound_to_calibration(tmp_path):
    lab, clock, gate, scope = optical_lab(tmp_path)
    state = lab.world.stamps["stamp_01"]
    state.apex_dx, state.apex_dy = .08, -.04
    frame = scope.capture()
    packet = frame.meta["stamp_structure"]
    structure = StampStructure.from_packet(packet)
    assert structure.cap_center_mm == structure.aperture_center_mm == (0, 0)
    assert structure.cap_radius_mm == structure.aperture_radius_mm == 5
    assert structure.aperture_shape == "diamond" and structure.aperture_radius_semantics == "inradius"
    assert packet["profile_id"] == STAMP_STRUCTURE_PROFILE
    assert frame.meta["calibration"]["stamp_structure_model"]["definition"]["aperture_shape"] == "diamond"
    gap = frame.meta["optical_gap"]
    assert (gap["centre_dx_mm"], gap["centre_dy_mm"]) != structure.cap_center_mm
    assert frame.meta["calibration"]["photometry_profile"]["stamp_structure"] == STAMP_STRUCTURE_PROFILE
    frozen, pixels = copy.deepcopy(frame.meta), frame.data
    signature = scope.preview_signature()
    lab.world.model["stamp"]["structure"]["aperture_offset_mm"] = [.25, -.1]
    assert scope.preview_signature() != signature
    newer = scope.capture()
    assert newer.meta["calibration"] != frame.meta["calibration"]
    assert newer.meta["stamp_structure"]["aperture_center_mm"] == pytest.approx([.25, -.1])
    assert frame.meta == frozen
    assert scope.renderer.render_bytes("microscope", frozen) == pixels
    before = copy.deepcopy(newer.meta["scene"]), clock.now()
    gate.microscope("microscope").set_focus(.3)
    after = scope.capture()
    assert after.meta["stamp_structure"] == newer.meta["stamp_structure"]
    assert after.meta["scene"]["stamp_axes_xyz_mm"] == before[0]["stamp_axes_xyz_mm"]
    assert after.meta["scene"]["sample_pose"] == before[0]["sample_pose"]
    assert clock.now() > before[1]  # normal drift/contact relaxation can evolve during focus travel


def test_missing_structure_profile_cannot_reuse_a_current_optical_reference(tmp_path):
    _, _, _, scope = optical_lab(tmp_path)
    original = scope.capture()
    metadata = copy.deepcopy(original.meta)
    metadata["calibration"]["photometry_profile"].pop("stamp_structure")
    legacy = Frame(original.device_id, original.timestamp, data=original.data, meta=metadata)
    with pytest.raises(NeedsHumanError, match="capture calibration"):
        _image_calibration(legacy)
    assert "stamp_structure" in original.meta["calibration"]["photometry_profile"]


def test_legacy_stamp_packet_without_structure_keeps_original_pixels():
    metadata = {"on_chip": "legacy", "fov_mm": [.65, .43], "objective": 20,
        "sharpness_chip": 8, "sharpness_stamp": 4, "stamp": {"film": True},
        "stamp_texture_seed": "fixture:legacy", "contact": {"dx_mm": .04, "dy_mm": -.02, "radius_mm": .09},
        "stamp_flakes": [], "flakes": [], "observation_noise_scale": 0}
    image = MicroscopeRenderer(None, size=(128, 84)).render_bytes("fixture", metadata)
    # Frozen from c881be9 before structure rendering; no historical backfill.
    # Re-frozen for the Gaussian chip defocus (#133): sharpness_chip 8 is
    # slightly defocused, so only the defocus kernel changed these pixels.
    # Re-frozen again when the contact front took the chip-plane defocus
    # (a518 seed 2 discovery autofocus); the in-focus hash below is unchanged.
    assert _pixel_sha(image) == "552f0908e5f2d7a00dca6df250aebe586423d5eeea534a849307ffa11d825ce7"  # bytes b72edc86
    # At best chip focus no defocus is applied; this hash is unchanged from
    # c881be9-era rendering, so the legacy packet identity is still guarded.
    image = MicroscopeRenderer(None, size=(128, 84)).render_bytes("fixture", {**metadata, "sharpness_chip": 10.5})
    assert _pixel_sha(image) == "8faa0233048c7658070f505dd7de769506e7c2a2cb92c87cd17214b19d76fdf7"  # bytes 561fb3f2


def test_legacy_implicit_circle_packet_keeps_original_pixels():
    packet = stamp_structure_packet({"stamp_slide": {"size": [40, 25], "dome_inset_mm": 7}},
        {"profile_id": LEGACY_STAMP_STRUCTURE_PROFILE, "cap_radius_mm": 5, "aperture_radius_mm": 5,
         "aperture_offset_mm": [0, 0]}, (4.9, 0), (0, 0))
    structure = StampStructure.from_packet(packet)
    assert structure.aperture_shape == "circle" and structure.aperture_radius_semantics == "radius"
    metadata = {"on_chip": "legacy", "fov_mm": [2.6, 1.72], "objective": 5,
        "sharpness_chip": 8, "sharpness_stamp": 4, "stamp": {"film": True},
        "stamp_texture_seed": "fixture:legacy-circle", "contact": {"dx_mm": .04, "dy_mm": -.02, "radius_mm": .09},
        "stamp_flakes": [], "flakes": [], "observation_noise_scale": 0, "stamp_structure": packet}
    pixels = MicroscopeRenderer(None, size=(128, 84)).render_bytes("fixture", metadata)
    # Frozen immediately before the explicit diamond profile was introduced.
    # Re-frozen for the Gaussian chip defocus (#133) and the chip-plane
    # contact defocus, as above.
    assert _pixel_sha(pixels) == "556325ec901557361a67d106954b3bd72904b795f68402c58a4c32cae66a375e"  # bytes 5fc07c3f
    # In-focus pixels are identical before and after the defocus change.
    pixels = MicroscopeRenderer(None, size=(128, 84)).render_bytes("fixture", {**metadata, "sharpness_chip": 10.5})
    assert _pixel_sha(pixels) == "2f624ff97a40050dd137c63ff569ec4ebafc4faecfd321dd65b018cbc7c69b59"  # bytes 4b489d93


def test_diamond_inradius_and_circular_cap_are_distinct():
    packet = stamp_structure_packet({"stamp_slide": {"size": [40, 25], "dome_inset_mm": 7}},
        {"profile_id": STAMP_STRUCTURE_PROFILE, "cap_radius_mm": 5, "aperture_radius_mm": 5,
         "aperture_shape": "diamond", "aperture_radius_semantics": "inradius",
         "aperture_offset_mm": [0, 0]}, (0, 0), (0, 0))
    structure = StampStructure.from_packet(packet)
    assert structure.cap_radius_mm == 5  # Still a circular cap, not the diamond vertex extent.
    canvas = Canvas(400, 400, (120, 90, 150))
    MicroscopeRenderer._paper_mask(canvas, structure, 20, 20)
    def pixel(x, y):
        at = (round(200-y*20)*400+round(200+x*20))*3
        return tuple(canvas.px[at:at+3])
    assert pixel(4.8, 0) == pixel(6, 0) == (120, 90, 150)  # Open outside the round cap too.
    assert pixel(3.8, 3.8) == pixel(7.5, 0) == (184, 179, 163)


@pytest.mark.parametrize("field,value", [("cap_radius_mm", -1), ("aperture_radius_mm", float("nan")),
                                         ("frame", "world"), ("source", "measured"),
                                         ("aperture_shape", "circle"), ("aperture_radius_semantics", "circumradius"),
                                         ("glass_outline_mm", [[0, 0]]*4)])
def test_invalid_structure_cannot_silently_render(field, value):
    packet = stamp_structure_packet({"stamp_slide": {"size": [40, 25], "dome_inset_mm": 7}},
        {"profile_id": STAMP_STRUCTURE_PROFILE, "cap_radius_mm": 5, "aperture_radius_mm": 5,
         "aperture_shape": "diamond", "aperture_radius_semantics": "inradius",
         "aperture_offset_mm": [0, 0]}, (10, 20), (10, 20))
    packet[field] = value
    with pytest.raises(ValueError):
        MicroscopeRenderer(None).render_bytes("fixture", {"stamp": {"film": True}, "stamp_structure": packet})


def test_opaque_paper_hides_source_and_contact_without_fabricating_transfer(tmp_path):
    _, _, _, scope = optical_lab(tmp_path)
    metadata = copy.deepcopy(scope.capture().meta)
    metadata.update(objective=5, fov_mm=[2.6, 1.72], observation_noise_scale=0,
                    contact=None, flakes=[], stamp_flakes=[])
    metadata["stamp_structure"]["aperture_center_mm"] = [5/math.sqrt(2), 5/math.sqrt(2)]
    metadata["stamp_structure"]["cap_center_mm"] = [5/math.sqrt(2), 5/math.sqrt(2)]
    renderer = MicroscopeRenderer(None, size=(320, 214))
    plain = Image.open(io.BytesIO(renderer.render_bytes("fixture", metadata)))
    contact = copy.deepcopy(metadata)
    contact["contact"] = {"dx_mm": -.5, "dy_mm": 0, "radius_mm": .1}
    contact["stamp_flakes"] = [{"sim_id": "hidden_gr", "material": "Gr", "layers": 1,
        "dx_mm": -.5, "dy_mm": 0, "area_um2": 10000,
        "outline_mm": [[-.56, -.04], [-.44, -.04], [-.5, .06]]}]
    contact["flakes"] = copy.deepcopy(contact["stamp_flakes"])
    obscured = Image.open(io.BytesIO(renderer.render_bytes("fixture", contact)))
    assert plain.crop((60, 75, 130, 140)).tobytes() == obscured.crop((60, 75, 130, 140)).tobytes()
    assert metadata["stamp"]["film"] is True
    assert metadata["flakes"] == [] and len(contact["stamp_flakes"]) == len(contact["flakes"]) == 1


def test_clock_driven_keyframes_keep_failed_pickup_and_replay_exact_pixels(tmp_path):
    """An explicit synthetic fixture exercises real device/Clock state changes."""
    lab, clock, gate, scope = optical_lab(tmp_path / "capture")
    scope.renderer.size = (320, 214)
    state, chip = lab.world.stamps["stamp_01"], lab.world.objects["chip_001"]
    state.optic_shift, state.apex_dx, state.apex_dy = .3, .08, -.04
    chip.flakes.append(SimFlake("keyframe_gr", "Gr", 1, 0, 0, 10000,
        outline_mm=((- .06, -.04), (.06, -.04), (0, .07))))
    holder = gate.manipulator("transfer_arm")
    over, park = lab.bench["transfer_arm"]["over_stage"], lab.bench["transfer_arm"]["park"]
    output = Path(os.environ.get("QUMUS_C3_KEYFRAME_DIR", str(tmp_path / "keyframes")))
    output.mkdir(parents=True, exist_ok=True)
    frames = []

    def capture(name):
        frame = scope.capture()
        frozen, pixels = copy.deepcopy(frame.meta), frame.data
        assert scope.renderer.render_bytes("microscope", frozen) == pixels
        preview = scope.preview_snapshot()
        assert display_matches(Path(preview.saved_path()).read_bytes(), pixels)
        assert scope.renderer.render_bytes("microscope", {**frozen, "captured_at": frame.timestamp+100}) == pixels
        (output / f"{name}.png").write_bytes(pixels)
        frames.append({"name": name, "timestamp": frame.timestamp, "meta": frozen,
                       "sha256": hashlib.sha256(pixels).hexdigest()})
        return frame

    holder.move_axis_to("z", 6.5)
    holder.move_axis_to("x", park["x"])
    scope.set_objective(20)
    scope.set_focus(0)
    capture("01-source-high-power")
    scope.set_objective(5)
    holder.move_axis_to("x", over["x"]+4.9/math.sqrt(2))
    holder.move_axis_to("y", over["y"]+4.9/math.sqrt(2))
    capture("02-aperture-edge-entering")
    holder.move_axis_to("x", over["x"])
    holder.move_axis_to("y", over["y"])
    blurred = capture("03-through-stamp-defocus")
    scope.set_focus(.3)
    focused = capture("04-wafer-refocus")
    assert focused.meta["sharpness_chip"] > blurred.meta["sharpness_chip"]
    contact_z = lab.world.manip.z + lab.world.apex()[3]
    # Enter the declared synthetic white-light coherence range; this fixture
    # selects a physical gap, never a requested image or process verdict.
    holder.move_axis_to("z", contact_z+.0002)
    rings = capture("05-approach-rings")
    assert rings.meta["optical_gap"]["signed_apex_gap_mm"] > 0
    holder.move_axis_to("z", contact_z-.003)
    clock.sleep(.5)
    small = capture("06-initial-contact")
    holder.move_axis_to("z", contact_z-.07)
    clock.sleep(1)
    expanded = capture("07-expanded-contact")
    assert expanded.meta["contact"]["radius_mm"] > small.meta["contact"]["radius_mm"]
    clock.sleep(2)
    capture("08-contact-hold")
    lab.world.faults.pickup_fail_next = 1
    holder.move_axis_to("z", 6.5)
    capture("09-lift-with-pickup-fault")
    assert chip.flakes and not state.stack and lab.world.faults.pickup_fail_next == 0
    holder.move_axis_to("x", park["x"])
    scope.set_objective(20)
    scope.set_focus(0)
    failed = capture("10-source-remains-failed-pickup")
    assert any(flake["sim_id"] == "keyframe_gr" for flake in failed.meta["flakes"])
    manifest = {"source": "synthetic fixture, not an accepted process run",
        "structure_profile": STAMP_STRUCTURE_PROFILE,
        "limitation": "Diamond aperture has synthetic 5 mm inradius and exceeds centred 5x FOV; clear field does not establish centre.",
        "frames": frames}
    (output / "keyframes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (1020, 1100), "#f4f5f7")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=13)
    draw.text((12, 10), "Synthetic stamp optics | actual Clock/device states | failed pickup retained", fill="#172338", font=font)
    for index, item in enumerate(frames):
        x, y = 10+(index % 3)*340, 42+(index // 3)*260
        sheet.paste(Image.open(output / f"{item['name']}.png"), (x, y))
        draw.text((x, y+217), item["name"], fill="#172338", font=font)
        elapsed = item["timestamp"]-frames[0]["timestamp"]
        draw.text((x, y+234), f"t+{elapsed:.2f}s | focus={item['meta']['focus_z']:.3f}mm",
                  fill="#48576d", font=font)
    for index, item in enumerate((frames[4], frames[5]), 10):
        x, y = 10+(index % 3)*340, 42+(index // 3)*260
        metadata = item["meta"]
        source = Image.open(output / f"{item['name']}.png")
        gap, (fw, fh) = metadata["optical_gap"], metadata["fov_mm"]
        cx, cy = round(160+gap["centre_dx_mm"]*320/fw), round(107-gap["centre_dy_mm"]*214/fh)
        detail = source.crop((cx-40, cy-27, cx+40, cy+27)).resize((320, 214), Image.Resampling.NEAREST)
        sheet.paste(detail, (x, y))
        draw.text((x, y+217), f"{item['name']} | 4x pixel detail", fill="#172338", font=font)
        draw.text((x, y+234), f"saved gap {gap['signed_apex_gap_mm']*1000:.3f} um",
                  fill="#48576d", font=font)
    sheet.save(output / "stamp-optics-keyframes.png")


def _pixel_sha(data: bytes) -> str:
    """sha256 of decoded RGB pixels: the PNG zlib level changed in #203, the
    pixels did not (the old byte hashes are kept in the comments)."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
