"""Analysis pixels and inexpensive focus exposures retain honest calibration."""
import io

import pytest
from PIL import Image

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.optical_profiles import (
    SYNTHETIC_SUBSTRATE_RGB,
    V1_MATERIAL_CAPTURE_PROFILE,
    V1_MATERIAL_SENSOR_PX,
)
from qumus_simlab.devices.sim.devices import SimMicroscope
from qumus_simlab.testbench.render import MicroscopeRenderer


@pytest.mark.parametrize("layers,low_rgb,high_rgb", [
    (1, (110, 89, 162), (113, 84, 167)),
    (2, (114, 77, 159), (113, 76, 164)),
])
def test_gr_forward_reference_shares_the_revisit_camera_colour_at_20x(layers, low_rgb, high_rgb):
    from qumus_simlab.testbench.render import _v1_material_tint

    # Frozen donor LAB anchors translated to the declared substrate, RGB8.
    # v1 LiveSegment: the 20X and 100X revisit captures share one thin-Gr
    # colour calibration; the 10X search camera has its own.
    assert _v1_material_tint("Gr", layers, 10, SYNTHETIC_SUBSTRATE_RGB) == low_rgb
    assert _v1_material_tint("Gr", layers, 20, SYNTHETIC_SUBSTRATE_RGB) == high_rgb
    assert _v1_material_tint("Gr", layers, 100, SYNTHETIC_SUBSTRATE_RGB) == high_rgb
    renderer = MicroscopeRenderer(None, material_capture_profile=V1_MATERIAL_CAPTURE_PROFILE)
    assert renderer.photometry_profile["material_reference_selection"] == "v1-objective-references-revisit20-gr-v3"


def test_native_donor_background_preserves_hbn_reference_contrast():
    from qumus_simlab.testbench.render import _v1_material_tint

    # 75 layers * .33 nm is the donor's 20–30 nm class. These are direct
    # LAB-to-RGB8 conversions, not colours optimized against a classifier.
    assert _v1_material_tint("hBN", 75, 20) == (128, 187, 127)
    assert _v1_material_tint("hBN", 75, 100) == (79, 143, 159)
    assert _v1_material_tint("Gr", 1, 20) == _v1_material_tint("Gr", 1, 100)
    assert _v1_material_tint("Gr", 1, 100) == (157, 86, 102)
    with pytest.raises(ValueError, match="profile"):
        MicroscopeRenderer(None, material_capture_profile="v1-reference-material-capture-v1")


def test_native_chip_pixels_and_substrate_support_share_the_new_camera_domain():
    import numpy as np

    from qumus_simlab.core.optical_profiles import native_substrate_support

    meta = {"on_chip": "empty-chip", "objective": 20, "fov_mm": [3840*.000182, 2160*.000182],
            "sharpness_chip": 10.5, "observation_noise_scale": 0,
            "calibration": {"material_capture_profile": V1_MATERIAL_CAPTURE_PROFILE}}
    raw = MicroscopeRenderer._compose(meta, size=V1_MATERIAL_SENSOR_PX)
    rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    assert native_substrate_support(rgb, 0).mean() > .9
    meta["calibration"] = {}
    legacy = np.asarray(Image.open(io.BytesIO(MicroscopeRenderer._compose(meta, size=V1_MATERIAL_SENSOR_PX)))
                        .convert("RGB"))
    assert native_substrate_support(legacy, 0).mean() < .01


def test_native_material_capture_is_rasterized_at_its_declared_scale():
    renderer = MicroscopeRenderer(None, material_capture_profile=V1_MATERIAL_CAPTURE_PROFILE)
    microscope = SimMicroscope("microscope", SimClock(), renderer=renderer)
    microscope.set_objective(20)
    frame = microscope.capture()
    focus = microscope.capture_for_focus()
    assert Image.open(io.BytesIO(frame.data)).size == (3840, 2160)
    assert (frame.width, frame.height) == (3840, 2160)
    assert frame.meta["calibration"]["mm_per_pixel"] == pytest.approx([.000182, .000182])
    assert frame.meta["fov_mm"] == pytest.approx([3840*.000182, 2160*.000182])
    assert Image.open(io.BytesIO(focus.data)).size == (480, 320)
    assert focus.meta["fov_mm"] == frame.meta["fov_mm"]
    assert focus.meta["calibration"]["mm_per_pixel"] == pytest.approx([3840*.000182/480, 2160*.000182/320])
    assert focus.meta["purpose"] == "focus_capture"
    assert frame.meta["purpose"] == "analysis_capture"


def test_full_sensor_neighbourhood_preserves_ordinary_5x_optical_protocols():
    renderer = MicroscopeRenderer(None, material_capture_profile=V1_MATERIAL_CAPTURE_PROFILE)
    microscope = SimMicroscope("microscope", SimClock(), renderer=renderer)
    microscope.set_objective(5)
    ordinary = microscope.capture()
    wide = microscope.capture_for_neighbourhood()
    after = microscope.capture()
    # ordinary 5X views are the v1 camera's stacking field at the analysis raster (#205)
    assert (ordinary.width, ordinary.height) == (960, 540)
    assert ordinary.meta["fov_mm"] == pytest.approx([3840*.00073, 2160*.00073])
    assert ordinary.meta["calibration"]["camera_profile"] == "v1-search-camera-measured-v1"
    assert after.data == ordinary.data and after.meta["calibration"] == ordinary.meta["calibration"]
    assert Image.open(io.BytesIO(wide.data)).size == (3265, 2160)
    assert wide.meta["fov_mm"] == [2.6, 1.72]
    assert "camera_profile" not in wide.meta["calibration"]
    assert wide.meta["calibration"]["mm_per_pixel"] == pytest.approx([2.6/3265, 1.72/2160])
    assert wide.meta["calibration"]["neighbourhood_capture_profile"] == "full-sensor-5x-neighbourhood-v1"
    assert "material_capture_profile" not in wide.meta["calibration"]
    assert wide.meta["purpose"] == "neighbourhood_capture"


def test_legacy_capture_does_not_gain_a_native_profile():
    microscope = SimMicroscope("microscope", SimClock(), renderer=MicroscopeRenderer(None))
    frame, focus = microscope.capture(), microscope.capture_for_focus()
    assert (frame.width, frame.height) == (480, 320)
    assert "material_capture_profile" not in frame.meta["calibration"]
    assert frame.data == focus.data



def test_completed_focus_stroke_reads_back_its_exact_target():
    """a518 seed 8: restoring 0.4991666666666687 from 1.4526713387901302 read
    back 0.49916666666666876, and the release support refused the 'changed'
    optical context of a focus no command had changed."""
    microscope = SimMicroscope("microscope", SimClock())
    microscope.set_focus(1.4526713387901302)
    microscope.set_focus(0.4991666666666687)
    assert microscope.get_focus() == 0.4991666666666687
