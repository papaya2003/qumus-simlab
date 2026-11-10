"""v1 camera-image algorithms on rendered frames, measured against the scene."""
import io
import math

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE
from qumus_simlab.devices.base import Frame
from qumus_simlab.perception import v1_vision
from qumus_simlab.perception.observations import RGBMicroscopeAnalyzer
from qumus_simlab.testbench.render import MicroscopeRenderer

pytest.importorskip("cv2")

W, H = 1780, 1180  # half the native sensor, same field of view: fast and v1-like


class _NoTruth:
    """An optical provider that fails if any ported method still asks the simulator."""

    def __getattr__(self, name):
        raise AssertionError(f"truth provider called: {name}")


def _frame(objective, *, corners=(), flakes=(), texture=None, sharpness=10.5):
    fw = 1.2996 * 10 / objective
    fh = fw * H / W
    meta = {"on_chip": "c1", "objective": objective, "fov_mm": [fw, fh],
            "sharpness_chip": sharpness, "sharpness": sharpness,
            "chip_corners": list(corners), "flakes": list(flakes),
            **({"specimen_texture": texture} if texture else {}),
            "calibration": {"material_capture_profile": V1_MATERIAL_CAPTURE_PROFILE,
                            "mm_per_pixel": [fw / W, fh / H], "image_size_px": [W, H],
                            "u_axis": "+world_x", "v_axis": "-world_y"}}
    return Frame("microscope", 0.0, data=MicroscopeRenderer._compose(meta, size=(W, H)), meta=meta)


def _rgb(frame):
    return np.asarray(Image.open(io.BytesIO(frame.data)).convert("RGB"))


def test_blur_score_is_v1_tenengrad_and_ranks_focus():
    corner = [{"chip": "c1", "corner": "bl", "dx_mm": -.1, "dy_mm": -.1, "chip_rotation": 1.}]
    sharp, soft = _frame(10, corners=corner), _frame(10, corners=corner, sharpness=3.)
    analyzer = RGBMicroscopeAnalyzer(_NoTruth())
    assert analyzer.sharpness(sharp) > analyzer.sharpness(soft)
    # Same formula as v1 camera_controller.blur_score_bgr on a known array.
    ramp = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
    # Sobel x of a 4 DN/px ramp is 32 inside; the two reflected border columns give 0.
    assert v1_vision.blur_score(ramp, shrink=1) == pytest.approx(math.log(32 ** 2 * 62 / 64), abs=1e-4)


@pytest.mark.parametrize("corner,dx,dy,rotation", [
    ("bl", -.12, -.08, 1.5), ("br", .1, -.1, -2.), ("tl", -.05, .12, .5), ("tr", .15, .06, -1.)])
def test_corner_is_found_from_pixels(corner, dx, dy, rotation):
    # A 16 x 26 mm chip placed so that the named corner sits at (dx, dy) from the view centre.
    hx = {"bl": -8, "tl": -8, "br": 8, "tr": 8}[corner]
    hy = {"bl": -13, "br": -13, "tl": 13, "tr": 13}[corner]
    c, s = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    pose = {"x": dx - (c * hx - s * hy), "y": dy - (s * hx + c * hy), "rotation": rotation}
    frame = _frame(10, texture={"seed": corner, "pose": pose, "size_mm": [16., 26.],
                                "view_center_world_mm": [0., 0.], "residue": False})
    analyzer = RGBMicroscopeAnalyzer(_NoTruth())
    found = analyzer.find_corner(frame, corner)
    assert math.hypot(found.dx_mm - dx, found.dy_mm - dy) < .002  # 1.46 um pixels here
    other = {"bl": "tr", "br": "tl", "tl": "br", "tr": "bl"}[corner]
    assert analyzer.find_corner(frame, other) is None


def test_corner_detector_names_the_visible_chip_corner():
    # Chip in the top-right of the screen: its bottom-left corner is in view.
    rgb = np.full((600, 900, 3), 20, np.uint8)
    rgb[:350, 400:] = (120, 140, 210)
    result = v1_vision.detect_edge_and_motion(rgb)
    found = v1_vision.corner_from_detector(result, rgb.shape)
    assert found.chip_corner == "bl"
    assert found.xy_px == pytest.approx((400, 350), abs=2)


def _flakes(view_x, view_y):
    flakes = [("a", 0., 0., 1500, "hBN", 80), ("b", .12, .05, 800, "hBN", 80),
              ("c", -.15, -.06, 2500, "Gr", 1), ("d", .05, -.12, 400, "hBN", 40)]
    return [{"sim_id": i, "dx_mm": x - view_x, "dy_mm": y - view_y, "area_um2": a,
             "material": m, "layers": n} for i, x, y, a, m, n in flakes]


def test_refind_template_recovers_the_reference_centre():
    reference = _frame(20, flakes=_flakes(0., 0.))
    shift = (.018, -.011)
    current = _frame(20, flakes=_flakes(*shift))
    match = v1_vision.refind_template(_rgb(reference), _rgb(current))
    fw, fh = current.meta["fov_mm"]
    x = (match.centre_px[0] - W / 2) * fw / W
    y = -(match.centre_px[1] - H / 2) * fh / H
    assert math.hypot(x + shift[0], y + shift[1]) < .001
    assert abs(match.angle_deg) <= 1


def test_refind_template_refuses_a_featureless_reference():
    blank = np.full((400, 600, 3), 128, np.uint8)
    with pytest.raises(ValueError, match="no valid result"):
        v1_vision.refind_template(blank, blank)


def test_analysis_reads_the_in_memory_q95_jpeg_and_writes_nothing():
    # v1 saved JPEG q95 and segmented that file; the sim exposure carries the
    # same JPEG bytes in Frame.data and perception reads them without a save.
    corner = [{"chip": "c1", "corner": "bl", "dx_mm": -.12, "dy_mm": -.08, "chip_rotation": 1.5}]
    png = _frame(10, corners=corner)
    buffer = io.BytesIO()
    Image.fromarray(_rgb(png)).save(buffer, format="JPEG", quality=95)

    def no_save():
        raise AssertionError("analysis must not write the exposure")

    jpeg = Frame("microscope", 0.0, data=buffer.getvalue(), meta=png.meta, save=no_save)
    analyzer = RGBMicroscopeAnalyzer(_NoTruth())
    decoded = np.asarray(Image.open(io.BytesIO(jpeg.data)).convert("RGB"))
    assert analyzer.sharpness(jpeg) == pytest.approx(v1_vision.blur_score(decoded))
    assert analyzer.sharpness(jpeg) != analyzer.sharpness(png)  # the JPEG pixels, not the PNG ones
    assert analyzer.find_corner(jpeg, "bl") is not None


def _corner_bgr(rotation, dx, dy):
    # Bottom-left corner of a 16 x 26 mm chip at (dx, dy) from the view centre.
    c, s = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    pose = {"x": dx - (c * -8 + s * 13), "y": dy - (s * -8 - c * 13), "rotation": rotation}
    frame = _frame(10, texture={"seed": "w:chip", "pose": pose, "size_mm": [16., 26.],
                                "view_center_world_mm": [0., 0.], "residue": False})
    return np.ascontiguousarray(_rgb(frame)[:, :, ::-1])


@pytest.mark.parametrize("searched,now", [(.3, 1.7), (1., -.5)])
def test_corner_rotation_is_v1_rough_refind_angle_from_one_corner(searched, now):
    # The chip came back rotated and shifted; v1's theta_cw is clockwise on
    # screen, so the world rotation now is the searched one minus theta_cw.
    found = v1_vision.corner_rotation(_corner_bgr(searched, -.12, -.08), _corner_bgr(now, -.07, -.11))
    assert found.found and found.inliers >= 50
    assert searched - math.degrees(found.theta_cw_rad) == pytest.approx(now, abs=.1)


def test_corner_rotation_reports_a_photo_without_features():
    blank = np.full((400, 600, 3), 128, np.uint8)
    found = v1_vision.corner_rotation(blank, blank)
    assert not found.found and found.theta_cw_rad is None and found.loss_reason
