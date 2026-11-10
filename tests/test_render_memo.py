"""Unchanged descent frames are reused byte for byte; changing ones are drawn."""
from __future__ import annotations

from qumus_simlab.testbench.render import MicroscopeRenderer
from tests.test_white_light_optics import optical_lab


def test_far_descent_frames_repeat_and_equal_a_fresh_render(tmp_path):
    lab, clock, gate, scope = optical_lab(tmp_path)
    arm = gate.manipulator("transfer_arm")
    touch = 5.30  # the fixture's dome touches the chip at holder z 5.3
    arm.move_axis_to("z", touch + 0.5)
    fresh = MicroscopeRenderer(None, size=scope.renderer.size,
                               material_capture_profile=scope.renderer.material_capture_profile)
    hits0 = scope.renderer.memo_hits
    frames = []
    for k in range(6):  # 10 um steps, far above the white-light ring range
        arm.move_axis_to("z", touch + 0.5 - 0.01 * k)
        frame = scope.capture()
        frames.append(frame)
        assert frame.data == fresh._render_bytes(frame.meta)  # the memo never changes pixels
    assert scope.renderer.memo_hits - hits0 >= 4


def test_pixel_key_drops_the_gap_only_beyond_the_ring_range():
    from qumus_simlab.testbench.optics import WHITE_LIGHT_MAX_GAP_MM, WHITE_LIGHT_PROFILE
    from qumus_simlab.testbench.render import _pixel_key

    def meta(gap, contact=False, **extra):
        return {"objective": 20, "fov_mm": [1.3, 0.86], "captured_at": extra.get("t", 0.0),
                "optical_gap": {"signed_apex_gap_mm": gap, "dome_radius_mm": 1.5,
                                "centre_dx_mm": 0.0, "centre_dy_mm": 0.0,
                                "current_contact_radius_mm": 0.0, "has_contact": contact,
                                "profile_id": WHITE_LIGHT_PROFILE}}

    far = WHITE_LIGHT_MAX_GAP_MM
    assert _pixel_key(meta(far + 0.1)) == _pixel_key(meta(far + 0.2, t=5.0))
    assert _pixel_key(meta(far * 0.5)) != _pixel_key(meta(far * 0.25))
    assert _pixel_key(meta(far + 0.1)) != _pixel_key(meta(far * 0.5))
    assert _pixel_key({**meta(far + 0.1), "flakes": [1]}) != _pixel_key(meta(far + 0.1))


def test_identical_consecutive_frames_share_one_file(tmp_path):
    from qumus_simlab.testbench.render import _Writer

    w = _Writer(tmp_path)
    a, b = w.write("microscope", b"same"), w.write("microscope", b"same")
    c = w.write("microscope", b"other")
    assert a == b != c and len(list(tmp_path.iterdir())) == 2
